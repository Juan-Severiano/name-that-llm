"""Gaussian sobre features LoRA do MESMO fold0: diagnóstico CPU, não CV completa.

  .venv/bin/python scripts/experimentos_gaussiano_lora.py
  .venv/bin/python scripts/experimentos_gaussiano_lora.py --source runs/experimentos_lora/two_stage_host_fold0

Não treina LoRA, não usa GPU, não reutiliza features supervisionadas fora do
fold0 e não avalia holdout. A época LoRA já foi selecionada na mesma validação:
o resultado é exploratório, inclusive seus intervalos por prompt.
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
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/experimentos_gaussiano_lora"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n");temp.replace(path)


def read_labels(indices,df,classes):
    allowed=set(map(int,indices))
    labels=pd.read_csv(ROOT/"data/train.csv",usecols=["id","model"],keep_default_na=False,
                       skiprows=lambda row:row>0 and row-1 not in allowed)
    if labels.id.tolist()!=df.id.iloc[indices].tolist():raise ValueError("ordem de labels/IDs inválida")
    mapping={c:i for i,c in enumerate(classes)}
    if not set(labels.model)<=set(classes):raise ValueError("classe desconhecida")
    return np.array([mapping[c] for c in labels.model],dtype=int)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"runs/experimentos_lora/official_fold0_b3")
    args=parser.parse_args();source=args.source.resolve();destination=OUT/source.name
    if (destination/"resultados.json").exists():
        print("Diagnóstico desse checkpoint já registrado; não repetir seleção/validação.",flush=True);return
    required=["anchor_features.npz","valid_best.npz","best.json","protocolo.json","resultados.json"]
    missing=[name for name in required if not (source/name).exists()]
    if missing:raise RuntimeError("LoRA ainda incompleto; aguardar anchors finais: "+", ".join(missing))
    protocol=json.loads((ROOT/"runs/validacao_pesada/split.json").read_text())
    if protocol["arquivo_sha256"]!=sha(ROOT/"data/train.csv"):raise ValueError("treino mudou após protocolo")
    df=pd.read_csv(ROOT/"data/train.csv",usecols=["id","text","prompt_id"],keep_default_na=False)
    classes=np.asarray(protocol["classes"],dtype=str)
    expected_train=np.asarray(protocol["inner_folds"][0]["treino_indices"])
    expected_valid=np.asarray(protocol["inner_folds"][0]["valid_indices"])
    if len(expected_train)!=1439 or len(expected_valid)!=720:raise ValueError("fold0 não corresponde ao escopo congelado")
    with np.load(source/"anchor_features.npz",allow_pickle=False) as f:
        X=f["features"].astype(np.float32);train=f["indices"];anchor_labels=f["labels"];anchor_prompts=f["prompts"]
        np.testing.assert_array_equal(f["classes"],classes)
    with np.load(source/"valid_best.npz",allow_pickle=False) as f:
        V=f["features"].astype(np.float32);valid=f["indices"];plain=f["plain"].astype(float)
        np.testing.assert_array_equal(f["classes"],classes)
    np.testing.assert_array_equal(train,expected_train);np.testing.assert_array_equal(valid,expected_valid)
    np.testing.assert_array_equal(anchor_prompts,df.prompt_id.iloc[train].to_numpy())
    if set(df.prompt_id.iloc[train])&set(df.prompt_id.iloc[valid]):raise ValueError("prompt leakage no fold0")
    lora_protocol=json.loads((source/"protocolo.json").read_text())
    np.testing.assert_array_equal(lora_protocol["train_indices"],train)
    np.testing.assert_array_equal(lora_protocol["valid_indices"],valid)
    if X.ndim!=2 or V.ndim!=2 or X.shape[0]!=len(train) or V.shape!=(len(valid),X.shape[1]) or not np.isfinite(X).all() or not np.isfinite(V).all():
        raise ValueError("shape/features LoRA inválidas")
    ytrain=read_labels(train,df,classes);yvalid=read_labels(valid,df,classes)
    np.testing.assert_array_equal(anchor_labels,ytrain)
    with np.load(ROOT/"runs/experimentos_prompt_removal/inner_oof.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["classes"],classes)
        lookup={int(index):i for i,index in enumerate(f["indices"])}
        baseline=f["mean_d128"][[lookup[int(index)] for index in valid]]
    best=json.loads((source/"best.json").read_text())
    final=json.loads((source/"resultados.json").read_text())
    if best["epoch"]!=final["selected_epoch"]:raise ValueError("checkpoint best/anchors finais incompatíveis")
    spec=importlib.util.spec_from_file_location("own_gaussian_research",ROOT/"scripts/experimentos_prompt_removal.py")
    math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
    destination.mkdir(parents=True,exist_ok=True)
    save(destination/"protocolo.json",{"fold":0,"train_rows":len(train),"valid_rows":len(valid),"selected_epoch":best["epoch"],
                                         "dimension":128,"temperature":3.,"cluster_threshold":.60,"blend_weights":[.25,.5],
                                         "source_sha256":{name:sha(source/name) for name in required},
                                         "source_model":lora_protocol.get("model"),"holdout_read":False,"gpu":False,
                                         "caveat":"Single exploratory fold, LoRA epoch already selected on this same validation. No final candidate selection or complete CV claim."})
    sem=np.asarray(np.load(ROOT/"data/emb/BAAI__bge-small-en-v1.5.treino.npy",mmap_mode="r",allow_pickle=False)[:,12],dtype=np.float32)
    groups=math.infer_groups(df.text.iloc[valid].tolist(),sem[valid])
    started=time.monotonic()
    with threadpool_limits(limits=1):
        reducer,Z=math.reduction(X,anchor_prompts,max_dimension=128)
        head=math.fit_gaussian(Z,ytrain,anchor_prompts,len(classes))
        Zv=reducer["scaler"].transform(V)@reducer["V"]
        independent=math.predict_gaussian(Zv,head)
        inferred=math.predict_gaussian(Zv,head,groups)
        oracle=math.predict_gaussian(Zv,head,df.prompt_id.iloc[valid].to_numpy())
    predictions={"baseline_gaussian5":baseline,"lora_plain_selected_head":plain,"gaussian_independent":independent,
                 "gaussian_inferred":inferred,"gaussian_oracle_prompt_diagnostic":oracle,
                 "blend_new_025":.75*baseline+.25*inferred,"blend_new_050":.5*baseline+.5*inferred}
    result={"fold":0,"selected_lora_epoch":best["epoch"],"feature_shape_train":list(X.shape),"feature_shape_valid":list(V.shape),
            "scores":{name:math.expanded_metrics(yvalid,p,classes) for name,p in predictions.items()},
            "paired_vs_baseline":{name:math.paired_prompt(yvalid,p,baseline,df.prompt_id.iloc[valid].to_numpy(),classes)
                                  for name,p in predictions.items() if name in ["gaussian_inferred","blend_new_025","blend_new_050"]},
            "clusters":math.cluster_diagnostics(groups,df.prompt_id.iloc[valid].to_numpy()),"gaussian_fit_seconds":time.monotonic()-started,
            "hypothesis_selected":False,"holdout_read":False,"complete_cv":False,
            "caveat":"Epoch selected on same validation. Exploratory single fold; never reuse this supervised cache in another fold/holdout."}
    np.savez_compressed(destination/"predicoes.npz",**predictions,indices=valid,classes=classes)
    save(destination/"resultados.json",result)
    print(json.dumps({"source":source.name,"epoch":best["epoch"],"scores":{name:v["balanced_accuracy"] for name,v in result['scores'].items()},
                      "blend_CIs":{k:v['delta_ci95'] for k,v in result['paired_vs_baseline'].items()},"holdout":False}),flush=True)


if __name__=="__main__":main()
