"""Reextrai amostras CPU/offline e identifica sua linha entre todos os caches.

Auditoria da associação texto→features, sem treinar ou medir desempenho.
Não acessa MPS para permitir execução durante outro experimento GPU.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from name_that_llm import gaussiano as g
from name_that_llm import gaussiano_robusto as robust

VISTAS = g.VISTAS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/validacao_pesada/embeddings_reextraidos_cpu.json"


def main():
    torch.set_num_threads(1)
    df = pd.read_csv(ROOT / "data/train.csv", keep_default_na=False)
    lengths = df.text.str.len()
    rows = np.unique(np.r_[np.random.default_rng(20261103).choice(len(df), 12, replace=False), lengths.idxmin(), lengths.idxmax()])
    texts = df.text.iloc[rows].tolist()
    result = {"rows": rows.tolist(), "ids": df.id.iloc[rows].tolist(), "device": "cpu", "offline": True, "models": {}}
    feature_file = OUT.with_suffix(".npz")
    if feature_file.exists():
        with np.load(feature_file, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["rows"], rows)
            stored = {key: archive[key] for key in archive.files}
    else:
        stored = {"rows": rows, "ids": df.id.iloc[rows].to_numpy(dtype=str)}
    original_views, fresh_views = {}, {}
    for name, spec in VISTAS.items():
        started = time.monotonic()
        if name not in stored:
            path = ROOT / "data/modelos" / ("styledistance" if name == "StyleDistance_L12" else name)
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).to("cpu").eval()
            features = []
            with torch.inference_mode():
                for start in range(0, len(rows), 2):
                    batch = tokenizer(texts[start:start+2], padding=True, truncation=True, max_length=512, return_tensors="pt")
                    states = model(**batch, output_hidden_states=True).hidden_states
                    mask = batch["attention_mask"].unsqueeze(-1)
                    means = [(states[layer] * mask).sum(1) / mask.sum(1) for layer in spec["camadas"]]
                    features.append(torch.cat(means, dim=1).float().numpy().astype(np.float16))
                    del batch, states, means
            stored[name] = np.concatenate(features)
            np.savez_compressed(feature_file, **stored)
            del model, tokenizer, features
            gc.collect()
        fresh = stored[name].astype(np.float32)
        if name == "StyleDistance_L12":
            cached = np.asarray(np.load(ROOT / "runs/experimentos_styledistance/train_embeddings.npy", mmap_mode="r", allow_pickle=False)[:, 12], dtype=np.float32)
        else:
            cached = np.asarray(np.load(ROOT / "data/artefatos" / f"emb_{name}.npy", mmap_mode="r", allow_pickle=False), dtype=np.float32)
        assert cached.shape[1] == fresh.shape[1]
        a = fresh / np.maximum(np.linalg.norm(fresh, axis=1, keepdims=True), 1e-12)
        b = cached / np.maximum(np.linalg.norm(cached, axis=1, keepdims=True), 1e-12)
        similarities = a @ b.T
        nearest = similarities.argmax(axis=1)
        paired = similarities[np.arange(len(rows)), rows]
        metrics = {"shape": list(fresh.shape), "seconds": time.monotonic()-started,
                   "max_absolute_difference": float(np.max(np.abs(fresh-cached[rows]))),
                   "min_paired_cosine": float(paired.min()), "nearest_cache_rows": nearest.tolist(),
                   "all_nearest_rows_correct": bool(np.array_equal(nearest, rows))}
        result["models"][name] = metrics
        original_views[name], fresh_views[name] = cached[rows], fresh
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"model": name, **metrics}), flush=True)
        assert np.array_equal(nearest, rows) and np.all(paired > .999), metrics
    test = df.iloc[rows][["id", "text"]].reset_index(drop=True)
    semantic = np.asarray(np.load(ROOT / "data/emb/BAAI__bge-small-en-v1.5.treino.npy", mmap_mode="r", allow_pickle=False)[rows,12], dtype=np.float32)
    model = robust.carregar(ROOT / "runs/gaussiano_candidato/robusto")
    old = g.Matrizes(test.id.to_numpy(dtype=str), original_views, semantic)
    new = g.Matrizes(test.id.to_numpy(dtype=str), fresh_views, semantic)
    results = {}
    for name, predict in (("baseline", lambda x: robust.prever_baseline(model, test, x)),
                          ("gaussian", lambda x: g.prever(model.gaussiano, test, x).probabilidades),
                          ("robust", lambda x: robust.prever(model, test, x).probabilidades)):
        a, b = predict(old), predict(new)
        results[name] = {"max_probability_difference": float(np.max(np.abs(a-b))),
                         "predictions_changed": int(np.sum(a.argmax(axis=1) != b.argmax(axis=1)))}
    result["feature_precision_diagnostic"] = results
    result["semantic_grouping_fixed_for_precision_check"] = True
    result["complete"] = True
    result["accuracy_measured"] = False
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
