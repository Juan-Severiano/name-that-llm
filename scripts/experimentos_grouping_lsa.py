"""Uma hipótese de grouping textual LSA, sem cap, assignment ou quotas.

O mecanismo TF-IDF/SVD vem da leitura do notebook público de yangkuangou;
nenhum código remoto é executado. A ordem canônica vem exclusivamente dos textos.
Head fixo: meanGaussian cinco vistas d128/T3. Só avaliar holdout se melhorar dev.

  PYTHONPATH=src .venv/bin/python scripts/experimentos_grouping_lsa.py selftest
  PYTHONPATH=src .venv/bin/python scripts/experimentos_grouping_lsa.py run
  PYTHONPATH=src /tmp/name-that-llm-sklearn161-20261103/bin/python scripts/experimentos_grouping_lsa.py verificar-versao
"""
import os
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[variable]="1"

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster,linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from name_that_llm import gaussiano as g

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/experimentos_grouping_lsa"
SEED=20261015
SIMILARITY=.18
TFIDF={"stop_words":"english","ngram_range":(1,1),"min_df":1,"max_df":1.,"sublinear_tf":True,
       "max_features":60000,"strip_accents":"unicode","lowercase":True,"analyzer":"word",
       "token_pattern":r"(?u)\b\w\w+\b","binary":False,"norm":"l2","use_idf":True,"smooth_idf":True}
SVD={"algorithm":"randomized","n_iter":5,"n_oversamples":10,"power_iteration_normalizer":"auto","random_state":SEED,"tol":0.}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/name;tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n");tmp.replace(path)


def own_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def grouping(texts):
    """Somente textos; sem IDs, prompts, classes, embeddings ou teto de tamanho."""
    n=len(texts)
    if n<2:return np.arange(n,dtype=int)
    order=np.array(sorted(range(n),key=lambda i:(hashlib.sha256(texts[i].encode()).digest(),texts[i])))
    canonical=[texts[i] for i in order]
    try:x=TfidfVectorizer(**TFIDF,dtype=np.float32).fit_transform(canonical)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):raise
        return np.arange(n,dtype=int)
    if min(x.shape)>3:
        dim=min(128,x.shape[0]-1,x.shape[1]-1)
        x=normalize(TruncatedSVD(n_components=dim,**SVD).fit_transform(x))
        similarity=x@x.T
    else:similarity=(x@x.T).toarray()
    distances=np.clip(1-similarity,0,2);np.fill_diagonal(distances,0)
    tree=linkage(squareform(distances,checks=False),method="average")
    canonical_groups=fcluster(tree,1-SIMILARITY,criterion="distance")
    result=np.empty(n,dtype=int);result[order]=canonical_groups
    return result


def predict_groups(model,features,groups):
    p=np.mean([g._probabilidades(features.vistas[k],model.parametros[k],groups) for k in g.VISTAS],axis=0)
    if not np.isfinite(p).all():raise ValueError("probabilidades não finitas")
    return p/p.sum(axis=1,keepdims=True)


def subset(all_features,indices):
    return g.Matrizes(all_features.ids[indices],{k:x[indices] for k,x in all_features.vistas.items()},all_features.semantica[indices])


def read_data():
    helper=own_module("own_qwen_protocol",ROOT/"scripts/experimentos_gaussiano_qwen4b.py")
    df,dev,holdout,classes,folds=helper.data_protocol()
    prep=own_module("own_gaussian_prepare",ROOT/"scripts/preparar_gaussiano.py")
    features=prep.cache(df)
    return helper,df,dev,holdout,classes,folds,features


def run():
    if (OUT/"complete.json").exists():print("Experimento já congelado; não repetir seleção/holdout.",flush=True);return
    helper,df,dev,holdout,classes,folds,features=read_data()
    math=own_module("own_gaussian_research",ROOT/"scripts/experimentos_prompt_removal.py")
    protocol={"hypothesis":"single fixed uncapped LSA128 grouping replaces mixed BGE/content grouping; Gaussian heads unchanged",
              "source":"https://www.kaggle.com/code/yangkuangou/name-that-llm-the-prompt-recovery-trap",
              "public_source_sha256":sha(ROOT/"data/public_notebooks/prompt_recovery_trap/source.txt"),
              "tfidf":TFIDF,"tfidf_dtype":"float32","svd":SVD,"svd_dimensions":"min(128,n_rows-1,n_terms-1), only when min(shape)>3",
              "canonical_order":"SHA256(text UTF8), then full text; restore caller order afterwards",
              "similarity_threshold":SIMILARITY,"distance_threshold":1-SIMILARITY,"linkage":"average",
              "hard_cap":False,"assignment":False,"class_quotas":False,"classifier":g.CONFIG,
              "gate":"evaluate holdout only if pooled dev balanced accuracy strictly exceeds original meanGaussian5 OOF",
              "train_sha256":sha(ROOT/"data/train.csv"),"dev_rows":len(dev),"holdout_rows":len(holdout),
              "historical_exposure":"Exploratory hypothesis on historically exposed data; not pristine external evidence.",
              "large_pool_if_dev_win":"diagnostic 150 prompts/2699 texts; five row permutations; sklearn1.6/1.9 parity required before promotion"}
    save("protocol.json",protocol)
    ydev=helper.labels_for(dev,df.id.to_numpy(),classes)
    y=np.full(len(df),-1,dtype=int);y[dev]=ydev
    with np.load(ROOT/"runs/experimentos_prompt_removal/inner_oof.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f['indices'],dev);np.testing.assert_array_equal(f['classes'],classes);baseline=f['mean_d128']
    pnew=np.full_like(baseline,np.nan);lookup={int(v):i for i,v in enumerate(dev)};cluster_metrics=[];control_deltas=[]
    with threadpool_limits(limits=1):
        for fold,(train,valid) in enumerate(folds):
            train_frame=df.iloc[train].copy();train_frame['model']=classes[y[train]]
            model=g.fit(train_frame,subset(features,train));valid_features=subset(features,valid)
            groups=grouping(df.text.iloc[valid].tolist())
            predictions=predict_groups(model,valid_features,groups)
            rows=[lookup[int(i)] for i in valid]
            control=g.prever(model,df.iloc[valid][['id','text']],valid_features).probabilidades
            np.testing.assert_allclose(control,baseline[rows],rtol=1e-10,atol=1e-12)
            control_deltas.append(float(np.max(np.abs(control-baseline[rows]))))
            pnew[rows]=predictions
            diagnostic=math.cluster_diagnostics(groups,df.prompt_id.iloc[valid].to_numpy());cluster_metrics.append(diagnostic)
            np.savez_compressed(OUT/f"fold{fold}_groups.npz",groups=groups,indices=valid)
            print(json.dumps({"fold":fold,"BA":math.metrics(y[valid],predictions,classes)['balanced_accuracy'],"clusters":diagnostic}),flush=True)
    if not np.isfinite(pnew).all():raise ValueError("OOF incompleto")
    original=math.metrics(ydev,baseline,classes);new=math.metrics(ydev,pnew,classes)
    win=new['balanced_accuracy']>original['balanced_accuracy']+1e-12
    save("development.json",{"baseline":original,"lsa":new,"clusters":cluster_metrics,"same_head_control_maxdiff":control_deltas,
                             "improved":win,"paired":math.paired_prompt(ydev,pnew,baseline,df.prompt_id.iloc[dev].to_numpy(),classes)})
    np.savez_compressed(OUT/"inner_oof.npz",baseline=baseline,lsa=pnew,indices=dev,classes=classes)
    save("selection_frozen.json",{"use_lsa":win,"holdout_labels_read":False,"variant_count":1,"new_ba":new['balanced_accuracy'],"baseline_ba":original['balanced_accuracy']})
    if not win:
        save("complete.json",{"dev_improved":False,"holdout_evaluated":False})
        print(json.dumps({"dev_baseline":original['balanced_accuracy'],"dev_lsa":new['balanced_accuracy'],"holdout_evaluated":False}),flush=True);return
    # A única hipótese já foi congelada; nenhuma busca sobre o holdout.
    with threadpool_limits(limits=1):
        frame=df.iloc[dev].copy();frame['model']=classes[ydev]
        model=g.fit(frame,subset(features,dev));hf=subset(features,holdout)
        texts=df.text.iloc[holdout].tolist();groups=grouping(texts);p=predict_groups(model,hf,groups)
    g.salvar(model,OUT/"head_gaussian_fixed")
    with np.load(ROOT/"runs/experimentos_prompt_removal/holdout_predictions.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f['indices'],holdout);np.testing.assert_array_equal(f['classes'],classes);pb=f['probas']
    yh=helper.labels_for(holdout,df.id.to_numpy(),classes)
    result={"baseline":math.expanded_metrics(yh,pb,classes),"lsa":math.expanded_metrics(yh,p,classes),
            "paired_vs_baseline":math.paired_prompt(yh,p,pb,df.prompt_id.iloc[holdout].to_numpy(),classes),
            "clusters":math.cluster_diagnostics(groups,df.prompt_id.iloc[holdout].to_numpy())}
    save("holdout.json",result)
    np.savez_compressed(OUT/"holdout_predictions.npz",probas=p,baseline=pb,groups=groups,indices=holdout,classes=classes)
    np.savez_compressed(OUT/"parity_inputs.npz",ids=hf.ids,textos=np.asarray(texts,dtype=str),semantica=hf.semantica,
                        **{f"vista{i}":hf.vistas[k] for i,k in enumerate(g.VISTAS)})
    np.savez_compressed(OUT/"parity_reference191.npz",groups=groups,probas=p,predictions=classes[p.argmax(axis=1)])
    # Escala maior sem alterar parâmetros e sem usar autores/labels de classe.
    started=time.monotonic();full_groups=grouping(df.text.tolist())
    scale={"rows":len(df),"prompts":int(df.prompt_id.nunique()),"seconds":time.monotonic()-started,
           "diagnostic":math.cluster_diagnostics(full_groups,df.prompt_id.to_numpy()),"retuning":False,"permutations":[]}
    np.savez_compressed(OUT/"full_grouping_reference191.npz",ids=df.id.to_numpy(dtype=str),textos=df.text.to_numpy(dtype=str),groups=full_groups)
    rng=np.random.default_rng(SEED)
    for j in range(5):
        permutation=rng.permutation(len(df));permuted=grouping(df.text.iloc[permutation].tolist());restored=np.empty_like(permuted);restored[permutation]=permuted
        same=float(adjusted_rand_score(full_groups,restored))
        if same<1:raise ValueError("LSA não preservou a partição sob permutação")
        scale['permutations'].append({"seed_draw":j,"partition_ari":same,"numeric_groups_equal":bool(np.array_equal(full_groups,restored))})
    save("large_pool_diagnostic.json",scale)
    save("complete.json",{"dev_improved":True,"holdout_evaluated":True,"large_pool_checked":True,"sklearn16_parity_pending":True})
    print(json.dumps({"dev_baseline":original['balanced_accuracy'],"dev_lsa":new['balanced_accuracy'],"holdout_baseline":result['baseline']['balanced_accuracy'],
                      "holdout_lsa":result['lsa']['balanced_accuracy'],"delta_ci":result['paired_vs_baseline']['delta_ci95'],"large_pool":scale['diagnostic']}),flush=True)


def verify_version():
    import sklearn
    model=g.carregar(OUT/"head_gaussian_fixed")
    with np.load(OUT/"parity_inputs.npz",allow_pickle=False) as f:
        texts=f['textos'].tolist();features=g.Matrizes(f['ids'],{k:f[f'vista{i}'] for i,k in enumerate(g.VISTAS)},f['semantica'])
    groups=grouping(texts);p=predict_groups(model,features,groups)
    with np.load(OUT/"parity_reference191.npz",allow_pickle=False) as ref:
        ari=float(adjusted_rand_score(ref['groups'],groups));same_preds=bool(np.array_equal(model.classes[p.argmax(axis=1)],ref['predictions']))
        delta=float(np.max(np.abs(p-ref['probas'])))
        np.testing.assert_array_equal(groups,ref['groups']);np.testing.assert_allclose(p,ref['probas'],rtol=1e-10,atol=1e-12)
    with np.load(OUT/"full_grouping_reference191.npz",allow_pickle=False) as f:
        full=grouping(f['textos'].tolist());full_ari=float(adjusted_rand_score(full,f['groups']));full_same=bool(np.array_equal(full,f['groups']))
        if full_ari<1:raise ValueError("partição full difere entre versões")
    result={"sklearn":sklearn.__version__,"holdout_partition_ari":ari,"max_probability_diff":delta,"predictions_equal":same_preds,
            "full_partition_ari":full_ari,"full_numeric_groups_equal":full_same,"Gaussian_refit":False,"transductive_LSA_fitted":True}
    save(f"parity_sklearn_{sklearn.__version__}.json",result);print(json.dumps(result),flush=True)
    if all((OUT/f"parity_sklearn_{v}.json").exists() for v in ("1.9.1","1.6.1")):
        completion=json.loads((OUT/"complete.json").read_text())
        completion["sklearn16_parity_pending"]=False
        completion["parity_versions"]=["1.9.1","1.6.1"]
        save("complete.json",completion)


def selftest():
    texts=["pizza tomato dough oven"]*25+["galaxy stars orbit planet"]
    original=grouping(texts);counts=np.bincount(original)
    assert counts.max()==25  # Prova concreta de ausência de cap18 ou quota por classe.
    rng=np.random.default_rng(SEED)
    for _ in range(3):
        p=rng.permutation(len(texts));permuted=grouping([texts[i] for i in p]);restored=np.empty_like(permuted);restored[p]=permuted
        assert adjusted_rand_score(original,restored)==1
    assert len(grouping([]))==0 and grouping([""]).tolist()==[0]
    np.testing.assert_array_equal(grouping(["",""]),[0,1])
    save("selftest.json",{"uncapped_group25":True,"text_only_permutation_invariant":True,"empty_and_singletons":True,"gpu":False})
    print("selftest LSA passou",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("selftest","run","verificar-versao"));args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):{"selftest":selftest,"run":run,"verificar-versao":verify_version}[args.command]()


if __name__=="__main__":main()
