"""StyleDistance: extração autorizada em MPS e seleção CPU por protocolo comum.

Preparar (somente CPU): .venv/bin/python scripts/experimentos_styledistance.py --prepare-only
Extrair após liberação de MPS: PYTHONPATH=src .venv/bin/python scripts/experimentos_styledistance.py --extract --allow-mps
Com cache pronto, executar sem argumentos para selecionar em dev e avaliar holdout uma vez.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from experimentos_fusao import SEED, bootstrap_prompts, digest, metrics, save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "experimentos_styledistance"
MODEL = ROOT / "data" / "modelos" / "styledistance"
REVISION = "b7df5f0b0480773c097ba3121d83ca32b71015ca"
COMMON = ROOT / "runs" / "validacao_pesada" / "split.json"
BASE = ROOT / "runs" / "experimentos_fusao"
VIEWS = {"last12": [12], "upper8_10_12": [8, 10, 12], "multi2_4_6_8_10_12": [2, 4, 6, 8, 10, 12], "normalized_last12": [12]}
CONFIGS = [{"name": f"{view}_C{c:g}", "view": view, "C": c} for view in VIEWS for c in [0.001, 0.01, 0.1]]
ALPHAS = [0, 0.25, 0.5, 0.75, 1]


def file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    common = json.loads(COMMON.read_text())
    source = ROOT / "data" / "train.csv"
    assert file_digest(source) == common["arquivo_sha256"], "O CSV mudou em relação ao protocolo comum"
    df = pd.read_csv(source, keep_default_na=False)
    classes = np.array(common["classes"])
    np.testing.assert_array_equal(classes, np.sort(df["model"].unique()))
    development, holdout = np.array(common["dev_indices"]), np.array(common["holdout_indices"])
    assert len(set(df["prompt_id"].iloc[development]) & set(df["prompt_id"].iloc[holdout])) == 0
    model_config = json.loads((MODEL / "config.json").read_text())
    pooling = json.loads((MODEL / "1_Pooling" / "config.json").read_text())
    assert model_config["num_hidden_layers"] == 12 and model_config["hidden_size"] == 768
    assert pooling["pooling_mode_mean_tokens"] and not pooling["pooling_mode_cls_token"]
    revision = (MODEL / ".cache" / "huggingface" / "download" / "config.json.metadata").read_text().splitlines()[0]
    assert revision == REVISION
    manifest = {
        "model": "StyleDistance/styledistance", "revision": REVISION,
        "weights_sha256": file_digest(MODEL / "model.safetensors"),
        "tokenizer_sha256": file_digest(MODEL / "tokenizer.json"),
        "pooling": pooling, "model_config": model_config, "max_len": 512, "batch": 16,
        "source_csv_sha256": common["arquivo_sha256"], "source_ids_sha256": digest(df["id"]),
        "class_order": classes.tolist(), "row_order": "original train.csv; no rows filtered or reordered",
    }
    embedding_cache = OUT / "train_embeddings.npy"
    if embedding_cache.exists():
        previous = json.loads((OUT / "model_manifest.json").read_text())
        assert previous["source_csv_sha256"] == manifest["source_csv_sha256"]
        assert previous["weights_sha256"] == manifest["weights_sha256"]
        cached = np.load(embedding_cache, mmap_mode="r", allow_pickle=False)
        assert cached.shape == (len(df), 13, 768)
        embedding_sha = file_digest(embedding_cache)
        assert previous["embedding_sha256"] == embedding_sha
        manifest.update(embedding_shape=list(cached.shape), embedding_dtype=str(cached.dtype), embedding_sha256=embedding_sha)
    save_json(OUT / "model_manifest.json", manifest)
    protocol = {
        "common_split_path": str(COMMON), "common_split_sha256": file_digest(COMMON), "seed": SEED,
        "development_indices": development.tolist(), "holdout_indices": holdout.tolist(),
        "classes": classes.tolist(), "inner_folds": common["inner_folds"],
        "configs_predeclared": CONFIGS,
        "feature_processing": "masked token mean per layer (including specials), optional row L2, training-only StandardScaler, LR multinomial",
        "fusion": {"baseline": "mean5 from honest previous campaign", "alphas_style": ALPHAS,
                   "rule": "select best of 12 style configs by inner OOF, then alpha by same dev OOF; never holdout tuning"},
        "holdout_predeclared": ["selected fusion", "baseline mean5", "best standalone StyleDistance"],
        "limitations": [common["limite"], "Same diagnostic holdout has prior exposure in this session; this is not a pristine external test.",
                       "Layer/C/alpha selection on the same inner-CV has selection optimism; holdout evaluated only after selection is frozen."],
    }
    save_json(OUT / "protocolo.json", protocol)
    np.savez_compressed(OUT / "split.npz", development=development, holdout=holdout,
                        classes=classes.astype(str), ids=df["id"].to_numpy().astype(str))
    return df, classes, development, holdout, common, manifest, protocol


def feature_matrix(embeddings, view):
    x = np.concatenate([embeddings[:, layer] for layer in VIEWS[view]], axis=1).astype(np.float32)
    return normalize(x) if view.startswith("normalized_") else x


def model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--allow-mps", action="store_true")
    args = parser.parse_args()
    if (OUT / "resultados.json").exists():
        raise SystemExit("Holdout já avaliado; não repetir para ajustar configurações.")
    started = time.monotonic()
    df, classes, development, holdout, common, manifest, protocol = prepare()
    if args.prepare_only:
        print(f"Preparado somente em CPU: {len(CONFIGS)} configurações, {len(df)} textos; extração ainda não chamada.", flush=True)
        return
    cache = OUT / "train_embeddings.npy"
    if args.extract and not cache.exists():
        if not args.allow_mps:
            raise SystemExit("A extração pode usar MPS; execute --allow-mps somente após liberação do coordenador.")
        sys.path.insert(0, str(ROOT / "src"))
        from name_that_llm.pipeline import extrair
        print("Extraindo StyleDistance local, mean pooling, max_len512, lote16.", flush=True)
        embeddings = extrair(MODEL, df["text"].tolist(), lote=16, max_len=512)
        assert embeddings.shape == (len(df), 13, 768) and np.isfinite(embeddings).all()
        np.save(cache, embeddings)
        manifest["embedding_shape"] = list(embeddings.shape)
        manifest["embedding_sha256"] = file_digest(cache)
        manifest["embedding_dtype"] = str(embeddings.dtype)
        save_json(OUT / "model_manifest.json", manifest)
        print(f"Embeddings prontos em {time.monotonic() - started:.1f}s.", flush=True)
    if not cache.exists():
        raise SystemExit("Manifestos preparados. Sem cache; aguardando liberação de MPS para --extract --allow-mps.")
    if args.extract_only:
        print("Extração concluída; modelo liberado. CV ainda não iniciado.", flush=True)
        return
    embeddings = np.load(cache, mmap_mode="r", allow_pickle=False)
    assert embeddings.shape == (len(df), 13, 768)
    matrices = {view: feature_matrix(embeddings, view) for view in VIEWS}
    oof = {config["name"]: np.zeros((len(development), len(classes))) for config in CONFIGS}
    folds = {name: [] for name in oof}
    dev_position = {int(index): i for i, index in enumerate(development)}
    for number, fold in enumerate(common["inner_folds"]):
        train, valid = np.array(fold["treino_indices"]), np.array(fold["valid_indices"])
        local = np.array([dev_position[int(index)] for index in valid])
        assert not set(train) & set(valid) and not set(train) & set(holdout)
        predictions = {}
        for config in CONFIGS:
            x = matrices[config["view"]]
            p = model(config["C"]).fit(x[train], df["model"].to_numpy()[train]).predict_proba(x[valid])
            oof[config["name"]][local] = p
            predictions[config["name"]] = p
            score = float(balanced_accuracy_score(df["model"].iloc[valid], classes[p.argmax(axis=1)]))
            folds[config["name"]].append(score)
            print(f"fold{number + 1} {config['name']} {score:.4f}", flush=True)
        np.savez_compressed(OUT / f"inner_fold{number + 1}.npz", **predictions, indices=valid, classes=classes.astype(str))
    rankings = [{"config": config, "inner_cv": metrics(df["model"].to_numpy()[development], oof[config["name"]], classes),
                 "fold_scores": folds[config["name"]]} for config in CONFIGS]
    rankings.sort(key=lambda row: row["inner_cv"]["balanced_accuracy"], reverse=True)
    best = rankings[0]["config"]
    with np.load(BASE / "inner_oof.npz", allow_pickle=False) as base:
        np.testing.assert_array_equal(base["indices"], development)
        np.testing.assert_array_equal(base["classes"], classes)
        baseline_oof = base["mean5"]
    fusion_rankings, fusion_oof = [], {}
    for alpha in ALPHAS:
        name = f"style_alpha{alpha:g}"
        p = (1 - alpha) * baseline_oof + alpha * oof[best["name"]]
        fusion_oof[name] = p
        fusion_rankings.append({"alpha_style": alpha,
            "inner_cv": metrics(df["model"].to_numpy()[development], p, classes)})
    fusion_rankings.sort(key=lambda row: row["inner_cv"]["balanced_accuracy"], reverse=True)
    alpha = fusion_rankings[0]["alpha_style"]
    save_json(OUT / "selecao_sem_holdout.json", {"best_style": best, "alpha_style": alpha,
              "style_ranking": rankings, "fusion_ranking": fusion_rankings})
    np.savez_compressed(OUT / "inner_oof.npz", **oof, **fusion_oof, baseline_mean5=baseline_oof,
                        indices=development, classes=classes.astype(str))
    print(f"Seleção congelada: {best}, alpha_style={alpha}. Avaliar holdout uma vez.", flush=True)
    # The holdout probability cache is intentionally opened only after selection is persisted.
    with np.load(BASE / "holdout_predictions.npz", allow_pickle=False) as base:
        np.testing.assert_array_equal(base["indices"], holdout)
        np.testing.assert_array_equal(base["classes"], classes)
        baseline = base["baseline_mean5"]
    x = matrices[best["view"]]
    style = model(best["C"]).fit(x[development], df["model"].to_numpy()[development]).predict_proba(x[holdout])
    final = {"selected": (1 - alpha) * baseline + alpha * style,
             "baseline_mean5": baseline, "standalone_style": style}
    y, groups = df["model"].to_numpy()[holdout], df["prompt_id"].to_numpy()[holdout]
    result = {
        "protocol": protocol, "model_manifest": manifest, "selected_style": best, "selected_alpha_style": alpha,
        "ranking_style_inner_cv": rankings, "ranking_fusion_inner_cv": fusion_rankings,
        "holdout": {name: metrics(y, p, classes) for name, p in final.items()},
        "bootstrap": bootstrap_prompts(y, final, groups, classes), "elapsed_seconds": time.monotonic() - started,
    }
    np.savez_compressed(OUT / "holdout_predictions.npz", **final, indices=holdout, classes=classes.astype(str))
    save_json(OUT / "resultados.json", result)
    print(json.dumps({"selected_style": best, "alpha_style": alpha,
                     "holdout": {k: v["balanced_accuracy"] for k, v in result["holdout"].items()},
                     "bootstrap": result["bootstrap"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
