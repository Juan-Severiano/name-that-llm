"""Prepara/verifica o candidato Gaussian local; nenhuma operação de Kaggle.

  PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py preparar
  PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py paridade
  PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py stress --linhas 5000
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py amostra --device mps

O comando amostra carrega encoders; os demais usam apenas matrizes já existentes.
"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from name_that_llm import gaussiano as g

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/gaussiano_candidato"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


def cache(df):
    vistas={name:np.load(ROOT/"data/artefatos"/f"emb_{name}.npy",allow_pickle=False).astype(np.float32)
            for name in g.VISTAS if name!="StyleDistance_L12"}
    style=ROOT/"runs/experimentos_styledistance/train_embeddings.npy"
    metadata=json.loads((style.parent/"model_manifest.json").read_text())
    if metadata["source_csv_sha256"]!=sha(ROOT/"data/train.csv") or metadata["source_ids_sha256"]!=g.checksum_ids(df.id):
        raise ValueError("proveniência/ordem StyleDistance incompatível")
    if sha(style)!=metadata["embedding_sha256"]:
        raise ValueError("checksum StyleDistance incompatível")
    vistas["StyleDistance_L12"]=np.asarray(np.load(style,mmap_mode="r",allow_pickle=False)[:,12],dtype=np.float32)
    sem=np.asarray(np.load(ROOT/"data/emb/BAAI__bge-small-en-v1.5.treino.npy",mmap_mode="r",allow_pickle=False)[:,12],dtype=np.float32)
    return g.Matrizes(df.id.to_numpy(dtype=str),vistas,sem)


def preparar():
    df=pd.read_csv(ROOT/"data/train.csv",keep_default_na=False)
    if json.loads((ROOT/"data/artefatos/rotulos.json").read_text())!=df.model.tolist():
        raise ValueError("rótulos dos caches diferem da ordem do treino")
    matrizes=cache(df)
    # As matrizes legadas não têm IDs próprios; comparar com as mesmas fatias dos caches-fonte.
    sources={}
    for name,spec in g.VISTAS.items():
        if name=="StyleDistance_L12":
            continue
        path=ROOT/"data/emb"/(spec["backbone"].replace("/","__")+".treino.npy")
        E=np.load(path,mmap_mode="r",allow_pickle=False)
        selected=np.concatenate([E[:,c] for c in spec["camadas"]],axis=1).astype(np.float32)
        if not np.array_equal(selected,matrizes.vistas[name]):
            raise ValueError(f"cache de {name} difere do artefato na mesma ordem")
        sources[name]={"source_sha256":sha(path),"shape":list(selected.shape)}
    started=time.monotonic()
    model=g.fit(df,matrizes)
    model.metadados["fontes"]={"train_csv_sha256":sha(ROOT/"data/train.csv"),"embeddings":sources,
                               "legacy_limit":"caches antigos não têm IDs próprios; conferidos labels/shape/fatias de origem; checksum registra a ordem atual"}
    g.salvar(model,OUT/"artefatos")
    # Seleção para PARIDADE numérica, sem medir acurácia ou consultar labels na inferência.
    indices=np.flatnonzero(df.prompt_id.isin(df.prompt_id.drop_duplicates().iloc[:2]))
    test=df.iloc[indices][["id","text"]].reset_index(drop=True)
    inputs=g.Matrizes(matrizes.ids[indices],{k:x[indices] for k,x in matrizes.vistas.items()},matrizes.semantica[indices])
    before=g.prever(model,test,inputs)
    after=g.prever(g.carregar(OUT/"artefatos"),test,inputs)
    np.testing.assert_array_equal(before.probabilidades,after.probabilidades)
    np.testing.assert_array_equal(before.grupos,after.grupos)
    np.testing.assert_array_equal(before.submissao.model,after.submissao.model)
    arrays={"ids":inputs.ids,"textos":test.text.to_numpy(dtype=str),"semantica":inputs.semantica}
    arrays.update({f"vista{i}":inputs.vistas[k] for i,k in enumerate(g.VISTAS)})
    np.savez_compressed(OUT/"paridade_entrada.npz",**arrays)
    np.savez_compressed(OUT/"paridade_referencia_191.npz",probabilidades=before.probabilidades,grupos=before.grupos,predicoes=before.submissao.model.to_numpy(dtype=str))
    save("preparo.json",{"linhas_treino":len(df),"classes":model.classes.tolist(),"treino_ids_sha256":g.checksum_ids(df.id),
                          "segundos_fit_export":time.monotonic()-started,"linhas_paridade":len(test),
                          "paridade_reload_maxdiff":float(np.max(np.abs(before.probabilidades-after.probabilidades))),
                          "refit_no_reload":False,"notebook_modificado":False})
    print(json.dumps(json.loads((OUT/"preparo.json").read_text()),ensure_ascii=False),flush=True)


def paridade():
    import sklearn
    model=g.carregar(OUT/"artefatos")
    with np.load(OUT/"paridade_entrada.npz",allow_pickle=False) as f:
        test=pd.DataFrame({"id":f["ids"],"text":f["textos"]})
        features=g.Matrizes(f["ids"],{k:f[f"vista{i}"] for i,k in enumerate(g.VISTAS)},f["semantica"])
    # Provar que carregar/prever não chamam ajuste, inclusive em outra versão do sklearn.
    g.fit=lambda *a,**k:(_ for _ in ()).throw(AssertionError("refit proibido na paridade"))
    g._ajustar_vista=g.fit
    result=g.prever(model,test,features)
    with np.load(OUT/"paridade_referencia_191.npz",allow_pickle=False) as ref:
        delta=float(np.max(np.abs(result.probabilidades-ref["probabilidades"])))
        np.testing.assert_allclose(result.probabilidades,ref["probabilidades"],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(result.grupos,ref["grupos"])
        np.testing.assert_array_equal(result.submissao.model.to_numpy(dtype=str),ref["predicoes"])
    r={"sklearn":sklearn.__version__,"sklearn_file":sklearn.__file__,"linhas":len(test),"maxdiff":delta,
       "clusters_iguais":True,"predicoes_iguais":True,"refit":False,"labels_fornecidos_na_entrada":False}
    save(f"paridade_sklearn_{sklearn.__version__}.json",r)
    print(json.dumps(r),flush=True)


def stress(n):
    model=g.carregar(OUT/"artefatos")
    df=pd.read_csv(ROOT/"data/train.csv",usecols=["id","text"],keep_default_na=False)
    features=cache(df)
    rows=np.arange(n)%len(df)
    test=df.iloc[rows][["id","text"]].reset_index(drop=True)
    test["id"]=[f"stress-{i:06d}" for i in range(n)]
    features=g.Matrizes(test.id.to_numpy(dtype=str),{k:x[rows] for k,x in features.vistas.items()},features.semantica[rows])
    start=time.monotonic()
    result=g.prever(model,test,features)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=="darwin" else 1024)
    unique,counts=np.unique(result.grupos,return_counts=True)
    report={"linhas":n,"segundos":time.monotonic()-start,"rss_pico_mb":rss,"classes_permitidas":bool(set(result.submissao.model)<=set(model.classes)),
            "ids_completos_ordenados":result.submissao.id.tolist()==test.id.tolist(),"ids_unicos":bool(result.submissao.id.is_unique),
            "probabilidades_finitas":bool(np.isfinite(result.probabilidades).all()),"max_erro_soma":float(np.max(np.abs(result.probabilidades.sum(axis=1)-1))),
            "clusters":len(unique),"maior_grupo":int(counts.max()),"labels_lidos":False,"gpu":False,
            "nota":"stress de desempenho/contrato, textos e features repetidos com novos IDs; não é nova avaliação de acurácia"}
    save(f"stress_{n}.json",report)
    print(json.dumps(report),flush=True)


def amostra(device):
    # Importações pesadas ocorrem SOMENTE neste comando, após agendamento de GPU.
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(1)
    df=pd.read_csv(ROOT/"data/test.csv",keep_default_na=False)
    model=g.carregar(OUT/"artefatos")
    views={};semantic=None
    for name,spec in g.VISTAS.items():
        path=ROOT/"data/modelos"/("styledistance" if name=="StyleDistance_L12" else name)
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        encoder=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False).to(device).eval()
        batch=tok(df.text.tolist(),padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
        with torch.inference_mode():
            output=encoder(**batch,output_hidden_states=True)
            hidden=torch.stack(output.hidden_states,dim=1)
            mask=batch["attention_mask"][:,None,:,None].to(hidden.dtype)
            means=((hidden*mask).sum(dim=2)/mask.sum(dim=2)).float().cpu().numpy().astype(np.float16)
        views[name]=np.concatenate([means[:,c] for c in spec["camadas"]],axis=1)
        if name=="bge-small-en-v1.5": semantic=means[:,12]
        del encoder,batch,output,hidden,mask
        if device=="mps":torch.mps.empty_cache()
    result=g.prever(model,df[["id","text"]],g.Matrizes(df.id.to_numpy(dtype=str),views,semantic))
    result.submissao.to_parquet(OUT/"amostra_submission.parquet",index=False)
    np.savez_compressed(OUT/"amostra_probabilidades.npz",probas=result.probabilidades,grupos=result.grupos)
    np.savez_compressed(OUT/"amostra_matrizes.npz",ids=df.id.to_numpy(dtype=str),semantica=semantic,
                        **{f"vista{i}":views[k] for i,k in enumerate(g.VISTAS)})
    save("amostra.json",{"linhas":len(df),"device":device,"ids_ordenados":result.submissao.id.tolist()==df.id.tolist(),
                          "finitos":bool(np.isfinite(result.probabilidades).all()),"diagnostico":result.diagnostico,
                          "vistas_concluidas":list(views),"normas_l2_vistas":{k:np.linalg.norm(v.astype(np.float64),axis=1).tolist() for k,v in views.items()},
                          "formas_vistas":{k:list(v.shape) for k,v in views.items()},"somas_probas":result.probabilidades.sum(axis=1).tolist(),
                          "predicoes":result.submissao.model.tolist(),"offline":True,"submissao_kaggle":False})
    print((OUT/"amostra.json").read_text(),flush=True)


def preparar_robusto():
    from name_that_llm import gaussiano_robusto as r
    from name_that_llm.pipeline import montar_tfidf,lr_embeddings
    from sklearn.linear_model import LogisticRegression
    df=pd.read_csv(ROOT/"data/train.csv",keep_default_na=False)
    matrices=cache(df);gaussian=g.carregar(OUT/"artefatos")
    started=time.monotonic()
    robust=r.fit(df,matrices,gaussian,ROOT/"data/artefatos")
    r.salvar(robust,OUT/"robusto")
    # Usar o smoke já guardado: este comando não carrega qualquer encoder/GPU.
    sample=pd.read_csv(ROOT/"data/test.csv",keep_default_na=False)
    with np.load(OUT/"amostra_matrizes.npz",allow_pickle=False) as f:
        sf=g.Matrizes(f["ids"],{k:f[f"vista{i}"] for i,k in enumerate(g.VISTAS)},f["semantica"])
    native=[];diffs={}
    tfidf=montar_tfidf(LogisticRegression(C=20,max_iter=2000)).fit(df,df.model.to_numpy(dtype=str))
    native.append(tfidf.predict_proba(sample))
    diffs['tfidf']=float(np.max(np.abs(native[-1]-r._tfidf(sample.text.tolist(),robust))))
    for name in r.BASES:
        pipe=lr_embeddings().fit(matrices.vistas[name],df.model.to_numpy(dtype=str))
        native.append(pipe.predict_proba(sf.vistas[name].astype(np.float32)))
        port=r._linear(sf.vistas[name].astype(np.float32),robust.lineares[name])
        diffs[name]=float(np.max(np.abs(native[-1]-port)))
    expected=np.mean(native,axis=0);expected/=expected.sum(axis=1,keepdims=True)
    baseline=r.prever_baseline(robust,sample[['id','text']],sf)
    np.testing.assert_allclose(baseline,expected,rtol=1e-6,atol=1e-7)
    result=r.prever(robust,sample[['id','text']],sf)
    reloaded=r.prever(r.carregar(OUT/"robusto"),sample[['id','text']],sf)
    np.testing.assert_array_equal(result.probabilidades,reloaded.probabilidades)
    np.testing.assert_array_equal(result.grupos,reloaded.grupos)
    pd.testing.assert_frame_equal(result.submissao,reloaded.submissao)
    # Paridade entre versões também utiliza o pool de 36 linhas previamente salvo.
    with np.load(OUT/"paridade_entrada.npz",allow_pickle=False) as f:
        test=pd.DataFrame({"id":f["ids"],"text":f["textos"]})
        features=g.Matrizes(f["ids"],{k:f[f"vista{i}"] for i,k in enumerate(g.VISTAS)},f["semantica"])
    reference=r.prever(robust,test,features)
    base_ref=r.prever_baseline(robust,test,features)
    np.savez_compressed(OUT/"robusto_paridade_referencia_191.npz",probabilidades=reference.probabilidades,baseline=base_ref,
                        grupos=reference.grupos,predicoes=reference.submissao.model.to_numpy(dtype=str))
    result.submissao.to_parquet(OUT/"robusto_amostra_submission.parquet",index=False)
    np.savez_compressed(OUT/"robusto_amostra_probabilidades.npz",probas=result.probabilidades,baseline=baseline,grupos=result.grupos)
    report={"segundos":time.monotonic()-started,"linhas_treino":len(df),"classes":gaussian.classes.tolist(),
            "mean5_portatil_vs_pipelines_maxdiff":float(np.max(np.abs(expected-baseline))),"por_componente":diffs,
            "paridade_reload_maxdiff":0.,"sem_refit_no_reload":True,"sample_ids_model_valido":result.submissao.id.tolist()==sample.id.tolist(),
            "sample_predicoes":result.submissao.model.tolist(),"sample_diagnostico":result.diagnostico,
            "artefatos_puros_alterados":False,"gpu":False,"submissao_kaggle":False}
    save("robusto_preparo.json",report);print(json.dumps(report,ensure_ascii=False),flush=True)


def paridade_robusta():
    import sklearn
    from name_that_llm import gaussiano_robusto as r
    model=r.carregar(OUT/"robusto")
    with np.load(OUT/"paridade_entrada.npz",allow_pickle=False) as f:
        test=pd.DataFrame({"id":f["ids"],"text":f["textos"]})
        features=g.Matrizes(f["ids"],{k:f[f"vista{i}"] for i,k in enumerate(g.VISTAS)},f["semantica"])
    r.fit=lambda *a,**k:(_ for _ in ()).throw(AssertionError("refit proibido"))
    g.fit=r.fit;g._ajustar_vista=r.fit
    result=r.prever(model,test,features);baseline=r.prever_baseline(model,test,features)
    with np.load(OUT/"robusto_paridade_referencia_191.npz",allow_pickle=False) as ref:
        delta=float(np.max(np.abs(result.probabilidades-ref["probabilidades"])))
        db=float(np.max(np.abs(baseline-ref["baseline"])))
        np.testing.assert_allclose(result.probabilidades,ref["probabilidades"],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(baseline,ref["baseline"],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(result.grupos,ref["grupos"])
        np.testing.assert_array_equal(result.submissao.model.to_numpy(dtype=str),ref["predicoes"])
    report={"sklearn":sklearn.__version__,"linhas":len(test),"maxdiff":delta,"baseline_maxdiff":db,"clusters_iguais":True,
            "predicoes_iguais":True,"refit":False,"labels_de_teste_fornecidos":False,"gpu":False}
    save(f"robusto_paridade_sklearn_{sklearn.__version__}.json",report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("preparar","paridade","stress","amostra"))
    p.add_argument("--linhas",type=int,default=5000)
    p.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    p.add_argument("--com-fallback",action="store_true",help="preparar/paridade do candidato robusto separado")
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.command=="preparar":preparar_robusto() if args.com_fallback else preparar()
        elif args.command=="paridade":paridade_robusta() if args.com_fallback else paridade()
        elif args.command=="stress":stress(args.linhas)
        else:amostra(args.device)


if __name__=="__main__":main()
