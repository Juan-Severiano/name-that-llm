"""Última hipótese fixa: cosseno TF-IDF direto, sem SVD/cap/assignment/quotas.

Somente dev primeiro. Se melhorar: escala/permutação antes de qualquer holdout.
Só um algoritmo novo, mesmo head Gaussian5 d128/T3 e similaridade .18.
"""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[name]="1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster,linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from name_that_llm import gaussiano as g
from experimentos_grouping_lsa import TFIDF,own_module,read_data,predict_groups,subset,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/experimentos_grouping_tfidf"
SIMILARITY=.18
MIN_PAIR_PRECISION=.98
MAX_ARI_DROP=.02


def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/name;tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n");tmp.replace(path)


def grouping(texts):
    n=len(texts)
    if n<2:return np.arange(n,dtype=int)
    order=np.array(sorted(range(n),key=lambda i:(hashlib.sha256(texts[i].encode()).digest(),texts[i])))
    canonical=[texts[i] for i in order]
    try:X=TfidfVectorizer(**TFIDF,dtype=np.float32).fit_transform(canonical)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):raise
        return np.arange(n,dtype=int)
    distance=np.clip(1-(X@X.T).toarray(),0,2);np.fill_diagonal(distance,0)
    tree=linkage(squareform(distance,checks=False),method="average")
    result=np.empty(n,dtype=int);result[order]=fcluster(tree,1-SIMILARITY,criterion="distance")
    return result


def run():
    if (OUT/"complete.json").exists():print("Experimento já concluído; não repetir seleção/holdout.",flush=True);return
    helper,df,dev,holdout,classes,folds,features=read_data()
    math=own_module("own_gaussian_math_tfidf",ROOT/"scripts/experimentos_prompt_removal.py")
    save("protocol.json",{"hypothesis":"single direct cosine TF-IDF grouping; no SVD, no cap, no assignment or class quota",
                          "tfidf":TFIDF,"dtype":"float32","similarity":SIMILARITY,"distance":1-SIMILARITY,"linkage":"average",
                          "canonical_order":"SHA256(text UTF8), then text, restore input order","head":g.CONFIG,
                          "dev_gate":"strictly improve pooled original meanGaussian5 OOF",
                          "scale_gate":{"min_pair_precision":MIN_PAIR_PRECISION,"max_ARI_drop_vs_current":MAX_ARI_DROP,
                                        "note":"diagnostic rejection rules fixed before testing; no inference size constraint"},
                          "holdout_gate":"only after passing dev and scale, freeze once",
                          "train_sha256":sha(ROOT/"data/train.csv"),"historical_exposure":"same previously exposed data, exploratory conditional evaluation"})
    labels=helper.labels_for(dev,df.id.to_numpy(),classes);y=np.full(len(df),-1,dtype=int);y[dev]=labels
    with np.load(ROOT/"runs/experimentos_prompt_removal/inner_oof.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f['indices'],dev);np.testing.assert_array_equal(f['classes'],classes);baseline=f['mean_d128']
    pnew=np.full_like(baseline,np.nan);lookup={int(v):i for i,v in enumerate(dev)};diagnostics=[]
    with threadpool_limits(limits=1):
        for fold,(train,valid) in enumerate(folds):
            frame=df.iloc[train].copy();frame['model']=classes[y[train]]
            model=g.fit(frame,subset(features,train));valid_features=subset(features,valid)
            groups=grouping(df.text.iloc[valid].tolist());p=predict_groups(model,valid_features,groups)
            rows=[lookup[int(i)] for i in valid]
            np.testing.assert_allclose(g.prever(model,df.iloc[valid][['id','text']],valid_features).probabilidades,baseline[rows],rtol=1e-10,atol=1e-12)
            pnew[rows]=p;diagnostics.append(math.cluster_diagnostics(groups,df.prompt_id.iloc[valid].to_numpy()))
            print(json.dumps({"fold":fold,"ba":math.metrics(y[valid],p,classes)['balanced_accuracy'],"groups":diagnostics[-1]}),flush=True)
    original=math.metrics(labels,baseline,classes);new=math.metrics(labels,pnew,classes)
    dev_win=new['balanced_accuracy']>original['balanced_accuracy']+1e-12
    save("development.json",{"baseline":original,"direct_tfidf":new,"cluster_diagnostics":diagnostics,"improved":dev_win})
    np.savez_compressed(OUT/"inner_oof.npz",baseline=baseline,direct_tfidf=pnew,indices=dev,classes=classes)
    if not dev_win:
        save("complete.json",{"dev_improved":False,"scale_evaluated":False,"holdout_evaluated":False})
        print(json.dumps({"dev_baseline":original['balanced_accuracy'],"dev_direct_tfidf":new['balanced_accuracy'],"holdout_evaluated":False}),flush=True);return
    groups_full=grouping(df.text.tolist())
    current,_=g._grupos(df.text.tolist(),features.semantica)
    quality=math.cluster_diagnostics(groups_full,df.prompt_id.to_numpy());reference=math.cluster_diagnostics(current,df.prompt_id.to_numpy())
    permutation_results=[];rng=np.random.default_rng(20261015)
    for iteration in range(5):
        order=rng.permutation(len(df));gp=grouping(df.text.iloc[order].tolist());restored=np.empty_like(gp);restored[order]=gp
        ari=float(adjusted_rand_score(restored,groups_full));permutation_results.append(ari)
        if ari<1:raise ValueError("ordem das linhas mudou partição TF-IDF")
    scale_pass=quality['pair_precision']>=MIN_PAIR_PRECISION and quality['adjusted_rand']>=reference['adjusted_rand']-MAX_ARI_DROP
    save("scale.json",{"rows":len(df),"new":quality,"current":reference,"permutation_ARI":permutation_results,"passed_fixed_gate":scale_pass})
    if not scale_pass:
        save("complete.json",{"dev_improved":True,"scale_evaluated":True,"scale_passed":False,"holdout_evaluated":False})
        print(json.dumps({"dev_baseline":original['balanced_accuracy'],"dev_direct_tfidf":new['balanced_accuracy'],"scale":quality,"holdout_evaluated":False}),flush=True);return
    save("selection_frozen.json",{"use_direct_tfidf":True,"variants_tested":1,"dev_passed":True,"scale_passed":True,"holdout_labels_read":False})
    frame=df.iloc[dev].copy();frame['model']=classes[labels]
    with threadpool_limits(limits=1):
        model=g.fit(frame,subset(features,dev));hf=subset(features,holdout)
        groups=grouping(df.text.iloc[holdout].tolist());p=predict_groups(model,hf,groups)
    with np.load(ROOT/"runs/experimentos_prompt_removal/holdout_predictions.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f['indices'],holdout);pb=f['probas']
    yh=helper.labels_for(holdout,df.id.to_numpy(),classes)
    results={"baseline":math.expanded_metrics(yh,pb,classes),"direct_tfidf":math.expanded_metrics(yh,p,classes),
             "paired":math.paired_prompt(yh,p,pb,df.prompt_id.iloc[holdout].to_numpy(),classes),"clusters":math.cluster_diagnostics(groups,df.prompt_id.iloc[holdout].to_numpy())}
    save("holdout.json",results)
    np.savez_compressed(OUT/"holdout_predictions.npz",probas=p,baseline=pb,groups=groups,indices=holdout,classes=classes)
    save("complete.json",{"dev_improved":True,"scale_passed":True,"holdout_evaluated":True,"candidate_api_changed":False})
    print(json.dumps({"dev":new['balanced_accuracy'],"scale":quality,"holdout":results['direct_tfidf']['balanced_accuracy'],
                      "baseline":results['baseline']['balanced_accuracy'],"delta_ci":results['paired']['delta_ci95']}),flush=True)


def selftest():
    texts=["pizza tomato dough oven"]*25+["galaxy stars orbit planet"]
    original=grouping(texts);assert np.bincount(original).max()==25
    order=np.random.default_rng(20261015).permutation(len(texts));p=grouping([texts[i] for i in order]);restored=np.empty_like(p);restored[order]=p
    assert adjusted_rand_score(original,restored)==1
    assert not len(grouping([])) and grouping([""]).tolist()==[0]
    save("selftest.json",{"group25_no_cap":True,"permutation_invariant":True,"empty_and_singletons":True,"gpu":False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("selftest","run"));args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):{"selftest":selftest,"run":run}[args.command]()


if __name__=="__main__":main()
