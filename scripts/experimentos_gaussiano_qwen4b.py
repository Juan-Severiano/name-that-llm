"""Hipótese Qwen3-4B-Base separada, sem alterar as cinco vistas vencedoras.

  .venv/bin/python scripts/experimentos_gaussiano_qwen4b.py prepare
  .venv/bin/python scripts/experimentos_gaussiano_qwen4b.py selftest
  .venv/bin/python scripts/experimentos_gaussiano_qwen4b.py run

run bloqueia até cache completo, verifica identidade/hashes e usa os três folds
do manifesto. Seleção das seis camadas × cinco pesos ocorre somente no dev.
Labels do holdout só são lidos depois de salvar selection_frozen.json.
"""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[name]="1"

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/experimentos_gaussiano_qwen4b"
CACHE=ROOT/"runs/experimentos_qwen4b"
LAYERS=[2,4,8,12,20,28]
ALPHAS=[0.,.25,.5,.75,1.]
MODEL="Qwen/Qwen3-4B-Base"
REVISION="906bfd4b4dc7f14ee4320094d8b41684abff8539"
REPRESENTATION={"keep_layers":28,"max_len":384,"append_eos":True,
                "pooling":"masked mean excluding token0, including EOS","final_norm":"Identity",
                "dtype_model":"bfloat16","dtype_features":"float16","use_cache":False,"batch_size":1,
                "eos_token_id":151643}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def lines_sha(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name;temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n");temp.replace(path)


def research():
    # Importar apenas código local próprio: nenhuma célula do notebook público é executada.
    path=ROOT/"scripts/experimentos_prompt_removal.py"
    spec=importlib.util.spec_from_file_location("own_prompt_gaussian_math",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def data_protocol():
    csv=ROOT/"data/train.csv"
    df=pd.read_csv(csv,usecols=["id","text","prompt_id"],keep_default_na=False)
    protocol=json.loads((ROOT/"runs/validacao_pesada/split.json").read_text())
    if sha(csv)!=protocol["arquivo_sha256"]:raise ValueError("CSV mudou depois do protocolo")
    dev=np.asarray(protocol["dev_indices"]);holdout=np.asarray(protocol["holdout_indices"])
    classes=np.asarray(protocol["classes"],dtype=str)
    folds=[(np.asarray(f["treino_indices"]),np.asarray(f["valid_indices"])) for f in protocol["inner_folds"]]
    if len(folds)!=3:raise ValueError("são necessários os três folds congelados")
    seen=[]
    for train,valid in folds:
        if not set(train)<=set(dev) or not set(valid)<=set(dev) or set(train)&set(valid):raise ValueError("fold usa holdout ou sobrepõe linhas")
        if set(df.prompt_id.iloc[train])&set(df.prompt_id.iloc[valid]):raise ValueError("prompt compartilhado entre treino e validação")
        seen.extend(valid.tolist())
    if sorted(seen)!=dev.tolist():raise ValueError("OOF não cobre desenvolvimento uma única vez")
    return df,dev,holdout,classes,folds


def labels_for(indices,all_ids,classes,csv=None):
    """skiprows exclui linhas externas antes de materializar seus labels no DataFrame."""
    csv=Path(csv) if csv is not None else ROOT/"data/train.csv"
    keep=set(map(int,indices))
    df=pd.read_csv(csv,usecols=["id","model"],keep_default_na=False,
                   skiprows=lambda row:row>0 and row-1 not in keep)
    if df.id.tolist()!=np.asarray(all_ids)[indices].tolist():raise ValueError("ordem dos labels difere dos IDs")
    mapping={c:i for i,c in enumerate(classes)}
    if not set(df.model)<=set(classes):raise ValueError("label fora das classes canônicas")
    return np.array([mapping[c] for c in df.model],dtype=int)


def verify_cache(df,cache=CACHE,csv=None):
    cache=Path(cache);csv=Path(csv) if csv is not None else ROOT/"data/train.csv"
    required=["features.npy","ids.npy","indices_originais.npy","concluidas.npy","manifesto.json","extracao_completa.json"]
    absent=[name for name in required if not (cache/name).exists()]
    if absent:raise RuntimeError("cache Qwen4B incompleto: "+", ".join(absent))
    done=np.load(cache/"concluidas.npy",allow_pickle=False)
    if done.shape!=(len(df),) or done.dtype.kind!="b" or not done.all():raise RuntimeError("extração parcial; nenhuma avaliação permitida")
    ids=np.load(cache/"ids.npy",allow_pickle=False)
    np.testing.assert_array_equal(ids,df.id.to_numpy(dtype=str))
    np.testing.assert_array_equal(np.load(cache/"indices_originais.npy",allow_pickle=False),np.arange(len(df)))
    m=json.loads((cache/"manifesto.json").read_text())
    checks={"modelo":MODEL,"revision":REVISION,"layers":LAYERS,"shape":[len(df),6,2560],**REPRESENTATION,
            "ids_sha256":lines_sha(df.id),"textos_sha256":lines_sha(df.text),"train_csv_sha256":sha(csv)}
    for k,v in checks.items():
        if m.get(k)!=v or type(m.get(k)) is not type(v):raise ValueError(f"manifesto Qwen4B incompatível: {k}")
    features=np.load(cache/"features.npy",mmap_mode="r",allow_pickle=False)
    if features.shape!=(len(df),6,2560) or features.dtype!=np.float16 or not np.isfinite(features).all():raise ValueError("features Qwen4B inválidas/não finitas")
    info={"manifesto":m,"manifest_sha256":sha(cache/"manifesto.json"),"feature_sha256":sha(cache/"features.npy"),
          "row_ids_sha256":lines_sha(df.id),"labels_lidos_na_verificacao":False}
    return features,info


def prepare():
    df,dev,holdout,classes,folds=data_protocol()
    expected={"hypothesis":"frozen Qwen4B layer Gaussian + unchanged original meanGaussian5 reference",
              "model":MODEL,"revision":REVISION,"layers":LAYERS,"alpha_new_view":ALPHAS,
              "dimension":128,"temperature":3.,"cluster_threshold":.60,
              "expected_pooling":"mean without token0, EOS included; keep28/36; final norm identity; max384",
              "representation_contract":REPRESENTATION,
              "train_csv_sha256":sha(ROOT/"data/train.csv"),"ids_sha256":lines_sha(df.id),
              "classes":classes.tolist(),"dev_rows":len(dev),"holdout_rows":len(holdout),
              "candidates": [{"layer":layer,"alpha_qwen4b":a} for layer in LAYERS for a in ALPHAS],
              "selection":"balanced accuracy on the three manifest dev folds; ties favor lower new-view weight then earlier layer",
              "no_holdout_labels_until":"selection_frozen.json exists",
              "baseline":"mean_d128 in runs/experimentos_prompt_removal/inner_oof.npz; no power/LoRA/robust-policy variant",
              "caveat":"Same historically exposed training data; this campaign holdout is diagnostic, not pristine external evidence.",
              "folds":[{"train_rows":len(t),"valid_rows":len(v),"train_ids_sha":lines_sha(df.id.iloc[t]),"valid_ids_sha":lines_sha(df.id.iloc[v])} for t,v in folds]}
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=expected:
        old=json.loads(path.read_text())
        prior_schema={k:v for k,v in expected.items() if k!="representation_contract"}
        # Enriquecimento não estatístico do contrato antes de qualquer seleção.
        if old!=prior_schema or (OUT/"selection_frozen.json").exists():
            raise RuntimeError("protocolo existente difere; não sobrescrever seleção")
    save("protocol.json",expected)
    np.savez_compressed(OUT/"split.npz",dev=dev,holdout=holdout,classes=classes,
                        **{f"fold{i}_{kind}":arr for i,(t,v) in enumerate(folds) for kind,arr in (("train",t),("valid",v))})
    print(json.dumps({"protocol":str(path),"candidates":30,"dev_rows":len(dev),"holdout_rows":len(holdout),"labels_read":False}),flush=True)


def run():
    if (OUT/"holdout.json").exists():
        print("Seleção/holdout já congelados; não executar novamente.",flush=True);return
    prepare()
    df,dev,holdout,classes,folds=data_protocol()
    features,cache_info=verify_cache(df)
    if (OUT/"verified_inputs.json").exists() and json.loads((OUT/"verified_inputs.json").read_text())!=cache_info:raise RuntimeError("cache mudou entre etapas")
    save("verified_inputs.json",cache_info)
    original=ROOT/"runs/experimentos_prompt_removal"
    parent=json.loads((original/"protocol.json").read_text())
    if parent["train_sha256"]!=sha(ROOT/"data/train.csv"):raise ValueError("baseline OOF veio de outro treino")
    with np.load(original/"inner_oof.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["indices"],dev);np.testing.assert_array_equal(f["classes"],classes)
        baseline=f["mean_d128"]
    labels_dev=labels_for(dev,df.id.to_numpy(),classes)
    y=np.full(len(df),-1,dtype=int);y[dev]=labels_dev
    math=research()
    semantic=np.asarray(np.load(ROOT/"data/emb/BAAI__bge-small-en-v1.5.treino.npy",mmap_mode="r",allow_pickle=False)[:,12],dtype=np.float32)
    if semantic.shape[0]!=len(df) or not np.isfinite(semantic).all():raise ValueError("cache semântico incompatível")
    out={layer:np.full((len(dev),len(classes)),np.nan) for layer in LAYERS}
    lookup={int(v):i for i,v in enumerate(dev)}
    with threadpool_limits(limits=1):
        for fold,(train,valid) in enumerate(folds):
            if not np.array_equal(np.unique(y[train]),np.arange(len(classes))):raise ValueError("classe ausente no treino interno")
            groups=math.infer_groups(df.text.iloc[valid].tolist(),semantic[valid])
            for j,layer in enumerate(LAYERS):
                X=np.asarray(features[:,j],dtype=np.float32)
                reducer,Z=math.reduction(X[train],df.prompt_id.iloc[train].to_numpy(),max_dimension=128)
                head=math.fit_gaussian(Z,y[train],df.prompt_id.iloc[train].to_numpy(),len(classes))
                Zvalid=reducer["scaler"].transform(X[valid])@reducer["V"]
                p=math.predict_gaussian(Zvalid,head,groups)
                out[layer][[lookup[int(i)] for i in valid]]=p
                print(f"fold{fold+1} layer{layer}: {math.metrics(y[valid],p,classes)['balanced_accuracy']:.4f}",flush=True)
    ranking=[]
    for layer in LAYERS:
        if not np.isfinite(out[layer]).all():raise ValueError("OOF incompleto")
        for alpha in ALPHAS:
            p=(1-alpha)*baseline+alpha*out[layer]
            ranking.append({"layer":layer,"alpha_qwen4b":alpha,**math.metrics(labels_dev,p,classes)})
    ranking.sort(key=lambda r:(-r["balanced_accuracy"],r["alpha_qwen4b"],r["layer"]))
    winner=ranking[0]
    save("development.json",{"ranking":ranking,"baseline_ba":math.metrics(labels_dev,baseline,classes)["balanced_accuracy"],
                             "baseline_source_sha256":sha(original/"inner_oof.npz")})
    np.savez_compressed(OUT/"inner_oof.npz",**{f"layer{k}":v for k,v in out.items()},baseline=baseline,indices=dev,classes=classes)
    save("selection_frozen.json",{"winner":winner,"holdout_labels_read":False,"cache_feature_sha256":cache_info["feature_sha256"]})
    print(f"SELEÇÃO CONGELADA layer={winner['layer']} alpha={winner['alpha_qwen4b']} dev={winner['balanced_accuracy']:.4f}",flush=True)
    # Nenhum label de holdout foi usado/materializado acima.
    with np.load(original/"holdout_predictions.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["indices"],holdout);np.testing.assert_array_equal(f["classes"],classes)
        base=f["probas"]
    alpha=winner["alpha_qwen4b"]
    if alpha:
        j=LAYERS.index(winner["layer"]);X=np.asarray(features[:,j],dtype=np.float32)
        groups=math.infer_groups(df.text.iloc[holdout].tolist(),semantic[holdout])
        with threadpool_limits(limits=1):
            reducer,Z=math.reduction(X[dev],df.prompt_id.iloc[dev].to_numpy(),max_dimension=128)
            head=math.fit_gaussian(Z,y[dev],df.prompt_id.iloc[dev].to_numpy(),len(classes))
            new=math.predict_gaussian(reducer["scaler"].transform(X[holdout])@reducer["V"],head,groups)
    else:new=base.copy()
    p=(1-alpha)*base+alpha*new
    labels_holdout=labels_for(holdout,df.id.to_numpy(),classes)
    result={"winner":winner,"selected":math.expanded_metrics(labels_holdout,p,classes),"baseline_gaussian5":math.expanded_metrics(labels_holdout,base,classes),
            "paired_vs_baseline":math.paired_prompt(labels_holdout,p,base,df.prompt_id.iloc[holdout].to_numpy(),classes),
            "old_five_view_config_modified":False,"source_of_model":"frozen pretrained Base; no new supervised extraction", "no_more_holdout_tuning":True}
    np.savez_compressed(OUT/"holdout_predictions.npz",probas=p,baseline=base,new_view=new,indices=holdout,classes=classes)
    save("holdout.json",result)
    print(json.dumps({"winner":winner["layer"],"alpha":alpha,"holdout":result["selected"]["balanced_accuracy"],
                      "baseline":result["baseline_gaussian5"]["balanced_accuracy"],"delta_ci":result["paired_vs_baseline"]["delta_ci95"]}),flush=True)


def selftest():
    directory=OUT/"selftest_cache";directory.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame({"id":["a","b","c"],"text":["one","two","three"],"model":["m0","HOLDOUT_NOT_READ","m1"]})
    csv=directory/"fake.csv";df.to_csv(csv,index=False)
    np.save(directory/"features.npy",np.zeros((3,6,2560),dtype=np.float16),allow_pickle=False)
    np.save(directory/"ids.npy",df.id.to_numpy(dtype=str),allow_pickle=False)
    np.save(directory/"indices_originais.npy",np.arange(3),allow_pickle=False)
    np.save(directory/"concluidas.npy",np.ones(3,dtype=bool),allow_pickle=False)
    (directory/"extracao_completa.json").write_text("{}")
    metadata={"modelo":MODEL,"revision":REVISION,"layers":LAYERS,"shape":[3,6,2560],**REPRESENTATION,"ids_sha256":lines_sha(df.id),
              "textos_sha256":lines_sha(df.text),"train_csv_sha256":sha(csv)}
    (directory/"manifesto.json").write_text(json.dumps(metadata))
    verify_cache(df,directory,csv)
    mutations={"keep_layers":36,"max_len":64,"append_eos":False,"pooling":"CLS","final_norm":"RMSNorm",
               "dtype_model":"float32","dtype_features":"float32","use_cache":True,"batch_size":2,"eos_token_id":151645}
    for field,bad_value in mutations.items():
        changed=dict(metadata);changed[field]=bad_value
        (directory/"manifesto.json").write_text(json.dumps(changed))
        try:verify_cache(df,directory,csv);raise AssertionError(f"representação incorreta aceita: {field}")
        except ValueError as exc:
            if field not in str(exc):raise
    changed=dict(metadata)|mutations
    (directory/"manifesto.json").write_text(json.dumps(changed))
    try:verify_cache(df,directory,csv);raise AssertionError("repro combinado aceito")
    except ValueError:pass
    (directory/"manifesto.json").write_text(json.dumps(metadata))
    labels=labels_for(np.array([0,2]),df.id.to_numpy(),np.array(["m0","m1"]),csv)
    np.testing.assert_array_equal(labels,[0,1])
    np.save(directory/"concluidas.npy",np.array([True,False,True]),allow_pickle=False)
    try:verify_cache(df,directory,csv);raise AssertionError("extração parcial aceita")
    except RuntimeError:pass
    np.save(directory/"concluidas.npy",np.ones(3,dtype=bool),allow_pickle=False)
    np.save(directory/"ids.npy",df.id.to_numpy(dtype=str)[::-1],allow_pickle=False)
    try:verify_cache(df,directory,csv);raise RuntimeError("IDs trocados aceitos")
    except AssertionError:pass
    np.save(directory/"ids.npy",df.id.to_numpy(dtype=str),allow_pickle=False)
    features=np.load(directory/"features.npy");features[2,5,0]=np.nan;np.save(directory/"features.npy",features,allow_pickle=False)
    try:verify_cache(df,directory,csv);raise AssertionError("NaN aceito")
    except ValueError:pass
    save("selftest.json",{"cache_partial_rejected":True,"reordered_ids_rejected":True,"nan_rejected":True,
                          "representation_fields_rejected_individually":list(mutations),"combined_bad_representation_rejected":True,
                          "only_requested_labels_materialized":True,"gpu":False})
    print("selftest passou",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","selftest","run"));args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):{"prepare":prepare,"selftest":selftest,"run":run}[args.command]()


if __name__=="__main__":main()
