"""Campanha CPU de fusão com seleção por prompt e avaliação externa única.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experimentos_fusao.py

Os embeddings são fixos. O holdout é novo para esta campanha, mas NÃO é um
teste intocado historicamente: escolha de encoders/camadas e outros experimentos
anteriores usaram os 150 prompts. OOF históricos não são usados neste script.
"""

import hashlib
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "experimentos_fusao"
SEED = 20261103
LAYERS = {
    "bge": ("BAAI__bge-small-en-v1.5", [2, 4, 6]),
    "roberta": ("FacebookAI__roberta-base", [2, 4, 6, 8, 10, 12]),
    "deberta": ("microsoft__deberta-v3-base", [4, 8, 10, 11, 12]),
    "qwen": ("Qwen__Qwen3-0.6B", [0, 4, 6, 18, 20]),
}
BASES = ["tfidf", *LAYERS]
CONFIGS = (
    [{"name": n, "kind": "base"} for n in BASES]
    + [{"name": f"concat_lr_C{c:g}", "kind": "concat_lr", "C": c} for c in [0.0001, 0.0005, 0.002, 0.01, 0.05]]
    + [{"name": f"unit_lr_C{c:g}", "kind": "unit_lr", "C": c} for c in [0.1, 1, 10]]
    + [{"name": f"ridge_a{a:g}", "kind": "ridge", "alpha": a} for a in [10, 100, 1000]]
    + [{"name": f"pca{n}_lda", "kind": "lda", "n": n} for n in [64, 128, 256]]
    + [{"name": f"cosine_knn_k{k}", "kind": "knn", "k": k} for k in [5, 15, 45]]
    + [{"name": n, "kind": "mean", "bases": b} for n, b in [
        ("mean5", BASES), ("mean4_no_bge", ["tfidf", "roberta", "deberta", "qwen"]),
        ("mean3_tfidf_deberta_qwen", ["tfidf", "deberta", "qwen"]),
    ]]
    + [{"name": f"stack_log_C{c:g}", "kind": "stack", "C": c} for c in [0.01, 0.1, 1]]
    + [{"name": "family_stack_C0.1", "kind": "family_stack", "C": 0.1}]
    + [{"name": "geomean5", "kind": "geomean"}]
)
assert len(CONFIGS) == 30


def digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def family(label):
    return {"claude": "Anthropic", "gemini": "Google", "gpt": "OpenAI",
            "glm": "ZAI", "kimi": "Moonshot", "qwen": "Qwen"}[label.split("-")[0].split("3")[0]]


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def lr(c):
    return LogisticRegression(C=c, max_iter=600, tol=1e-4)


def tfidf():
    return make_pipeline(FeatureUnion([
        ("word", TfidfVectorizer(lowercase=False, min_df=2, sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2))),
        ("char", TfidfVectorizer(lowercase=False, min_df=3, sublinear_tf=True,
            analyzer="char", ngram_range=(1, 5))),
    ]), lr(20))


def base_predictions(df, features, train, valid):
    result = {}
    y = df["model"].to_numpy()
    start = time.monotonic()
    result["tfidf"] = tfidf().fit(df["text"].iloc[train], y[train]).predict_proba(df["text"].iloc[valid])
    for name, x in features.items():
        result[name] = make_pipeline(StandardScaler(), lr(0.05)).fit(x[train], y[train]).predict_proba(x[valid])
    log(f"bases train={len(train)} valid={len(valid)}: {time.monotonic() - start:.1f}s")
    return result


def meta_features(predictions):
    return np.concatenate([np.log(np.clip(predictions[b], 1e-8, 1)) for b in BASES], axis=1)


def stack_training_predictions(df, features, train, n_classes):
    p = {b: np.zeros((len(train), n_classes)) for b in BASES}
    for subtrain, subvalid in GroupKFold(n_splits=2, shuffle=True, random_state=SEED + 2).split(
        train, groups=df["prompt_id"].iloc[train]
    ):
        part = base_predictions(df, features, train[subtrain], train[subvalid])
        for b in BASES:
            p[b][subvalid] = part[b]
    return p


def hierarchical(trainx, y, validx, classes, c):
    fam = np.array([family(label) for label in y])
    model = make_pipeline(StandardScaler(), lr(c)).fit(trainx, fam)
    fp = model.predict_proba(validx)
    p = np.zeros((len(validx), len(classes)))
    for f in np.unique(fam):
        ix = np.flatnonzero(np.array([family(label) for label in classes]) == f)
        fi = list(model.classes_).index(f)
        if len(ix) == 1:
            p[:, ix[0]] = fp[:, fi]
        else:
            sub = make_pipeline(StandardScaler(), lr(c)).fit(trainx[fam == f], y[fam == f])
            p[:, ix] = sub.predict_proba(validx) * fp[:, fi, None]
    return p


def run_candidates(df, features, concat, unit, train, valid, classes, configs):
    result = {}
    kinds = {c["kind"] for c in configs}
    y = df["model"].to_numpy()
    need_base = kinds & {"base", "mean", "geomean", "stack", "family_stack"}
    bases = base_predictions(df, features, train, valid) if need_base else {}
    stdx = stdv = None
    if kinds & {"concat_lr", "ridge", "lda"}:
        scaler = StandardScaler()
        stdx = scaler.fit_transform(concat[train])
        stdv = scaler.transform(concat[valid])
    if "lda" in kinds:
        pca = PCA(n_components=256, svd_solver="randomized", random_state=SEED, iterated_power=3)
        pcax = pca.fit_transform(stdx)
        pcav = pca.transform(stdv)
    if "knn" in kinds:
        nn = NearestNeighbors(n_neighbors=45, metric="cosine", n_jobs=1).fit(unit[train])
        distances, neighbors = nn.kneighbors(unit[valid])
    if kinds & {"stack", "family_stack"}:
        crossfit = stack_training_predictions(df, features, train, len(classes))
        metax, metav = meta_features(crossfit), meta_features(bases)
    for config in configs:
        name, kind = config["name"], config["kind"]
        t0 = time.monotonic()
        if kind == "base":
            p = bases[name]
        elif kind == "concat_lr":
            p = lr(config["C"]).fit(stdx, y[train]).predict_proba(stdv)
        elif kind == "unit_lr":
            p = lr(config["C"]).fit(unit[train], y[train]).predict_proba(unit[valid])
        elif kind == "ridge":
            model = RidgeClassifier(alpha=config["alpha"], solver="lsqr", tol=1e-4).fit(stdx, y[train])
            p = softmax(model.decision_function(stdv), axis=1)
        elif kind == "lda":
            n = config["n"]
            p = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(pcax[:, :n], y[train]).predict_proba(pcav[:, :n])
        elif kind == "knn":
            k = config["k"]
            codes = np.searchsorted(classes, y[train][neighbors[:, :k]])
            weights = 1 / np.maximum(distances[:, :k], 1e-6)
            p = np.zeros((len(valid), len(classes)))
            np.add.at(p, (np.arange(len(valid))[:, None], codes), weights)
            p /= p.sum(axis=1, keepdims=True)
        elif kind == "mean":
            p = np.mean([bases[b] for b in config["bases"]], axis=0)
        elif kind == "geomean":
            p = softmax(np.mean([np.log(np.clip(bases[b], 1e-8, 1)) for b in BASES], axis=0), axis=1)
        elif kind == "stack":
            p = make_pipeline(StandardScaler(), lr(config["C"])).fit(metax, y[train]).predict_proba(metav)
        elif kind == "family_stack":
            p = hierarchical(metax, y[train], metav, classes, config["C"])
        else:
            raise ValueError(kind)
        assert p.shape == (len(valid), len(classes)) and np.isfinite(p).all()
        result[name] = p
        score = balanced_accuracy_score(y[valid], classes[p.argmax(axis=1)])
        log(f"{name}: {score:.4f} ({time.monotonic() - t0:.1f}s)")
    return result


def metrics(y, p, classes):
    pred = classes[p.argmax(axis=1)]
    fy, fp = [family(label) for label in y], [family(label) for label in pred]
    by_family = {}
    for fam in sorted(set(fy)):
        take = np.array(fy) == fam
        by_family[fam] = {
            "n": int(take.sum()), "exact_model_accuracy": float(np.mean(pred[take] == y[take])),
            "family_accuracy": float(np.mean(np.array(fp)[take] == fam)),
        }
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "family_balanced_accuracy": float(balanced_accuracy_score(fy, fp)),
        "per_class_recall": dict(zip(classes, recall_score(y, pred, labels=classes, average=None).tolist())),
        "per_family": by_family,
    }


def bootstrap_prompts(y, predictions, groups, classes, seed=SEED, n=2000):
    unique = np.unique(groups)
    totals = np.zeros((len(unique), len(classes)))
    correct = {name: np.zeros_like(totals) for name in predictions}
    for i, group in enumerate(unique):
        for j, cl in enumerate(classes):
            ix = (groups == group) & (y == cl)
            totals[i, j] = ix.sum()
            for name, p in predictions.items():
                correct[name][i, j] = np.sum(classes[p[ix].argmax(axis=1)] == cl)
    take = np.random.default_rng(seed).integers(len(unique), size=(n, len(unique)))
    denom = totals[take].sum(axis=1)
    draws = {name: np.mean(counts[take].sum(axis=1) / denom, axis=1) for name, counts in correct.items()}
    result = {name: {"ci95": np.quantile(values, [0.025, 0.975]).tolist()} for name, values in draws.items()}
    if "selected" in draws and "baseline_mean5" in draws:
        diff = draws["selected"] - draws["baseline_mean5"]
        result["paired_delta"] = {"ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
                                  "probability_positive_bootstrap": float(np.mean(diff > 0))}
    return {"unit": "prompt_id", "replications": n, "seed": seed, "results": result}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "resultados.json").exists():
        raise SystemExit("Avaliação externa já gravada; não a repetir para ajustar configurações.")
    start = time.monotonic()
    df = pd.read_csv(ROOT / "data" / "train.csv", keep_default_na=False)
    classes = np.sort(df["model"].unique())
    prompts = np.sort(df["prompt_id"].unique())
    development, holdout = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
        df, groups=df["prompt_id"]))
    holdout_prompts = np.sort(df["prompt_id"].iloc[holdout].unique())
    folds = list(GroupKFold(n_splits=3).split(
        development, groups=df["prompt_id"].iloc[development]))
    protocol = {
        "seed": SEED, "sklearn": sklearn.__version__, "n_rows": len(df), "n_classes": len(classes),
        "development_rows": len(development), "development_prompts": len(prompts) - len(holdout_prompts),
        "holdout_rows": len(holdout), "holdout_prompts": holdout_prompts.tolist(),
        "holdout_prompt_sha256": digest(holdout_prompts), "holdout_id_sha256": digest(df["id"].iloc[holdout]),
        "development_id_sha256": digest(df["id"].iloc[development]),
        "inner_cv": "GroupKFold3 no shuffle, stacking nested GroupKFold2 shuffled seed20261105",
        "holdout_split": "GroupShuffleSplit1 test_size0.2 random_state20261103, original row order",
        "class_order": classes.tolist(), "source_id_sha256": digest(df["id"]),
        "selection_metric": "pooled inner-OOF balanced accuracy; tie by declared configuration order",
        "configs_predeclared": CONFIGS, "layers_previously_selected": LAYERS,
        "holdout_evaluation_predeclared": ["selected configuration", "mean5 baseline"],
        "limitations": [
            "Holdout novo para esta campanha, mas camadas/encoders e experimentos históricos usaram todos os prompts.",
            "Não reutiliza OOF históricos; TF-IDF, scaler, PCA e classificadores são ajustados só nos treinos de cada dobra.",
            "30 configurações e seleção no mesmo inner-CV podem sobreajustar CV; holdout é diagnóstico, não leaderboard.",
            "Reutiliza embeddings congelados de encoders públicos; não treina nem usa GPU nesta campanha.",
        ],
    }
    save_json(OUT / "protocolo.json", protocol)
    np.savez_compressed(OUT / "split.npz", development=development, holdout=holdout,
                        classes=classes.astype(str), ids=df["id"].to_numpy().astype(str))
    features = {}
    for name, (file, layers) in LAYERS.items():
        raw = np.load(ROOT / "data" / "emb" / f"{file}.treino.npy", mmap_mode="r")
        assert raw.shape[0] == len(df)
        features[name] = np.concatenate([raw[:, layer] for layer in layers], axis=1).astype(np.float32)
    concat = np.concatenate(list(features.values()), axis=1)
    unit = np.concatenate([normalize(x) for x in features.values()], axis=1) / np.sqrt(len(features))
    log(f"desenvolvimento={len(development)} holdout={len(holdout)} atributos={concat.shape[1]}; 30 candidatos")
    oof = {c["name"]: np.zeros((len(development), len(classes))) for c in CONFIGS}
    fold_scores = {c["name"]: [] for c in CONFIGS}
    for i, (train, valid) in enumerate(folds):
        cache = OUT / f"inner_fold{i + 1}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as loaded:
                predictions = {name: loaded[name] for name in oof}
            log(f"dobra {i + 1}/3 recuperada do cache do mesmo protocolo")
        else:
            log(f"início dobra {i + 1}/3")
            predictions = run_candidates(df, features, concat, unit, development[train], development[valid], classes, CONFIGS)
            np.savez_compressed(cache, **predictions, indices=development[valid], classes=classes.astype(str))
        for name, p in predictions.items():
            oof[name][valid] = p
            fold_scores[name].append(float(balanced_accuracy_score(df["model"].iloc[development[valid]], classes[p.argmax(axis=1)])))
        save_json(OUT / "progresso.json", {"completed_folds": i + 1, "fold_scores": fold_scores,
                  "elapsed_seconds": time.monotonic() - start})
    rankings = []
    for config in CONFIGS:
        name = config["name"]
        rankings.append({"config": config, "inner_cv": metrics(df["model"].to_numpy()[development], oof[name], classes),
                         "fold_scores": fold_scores[name]})
    rankings.sort(key=lambda row: row["inner_cv"]["balanced_accuracy"], reverse=True)
    winner = rankings[0]["config"]
    save_json(OUT / "selecao_sem_holdout.json", {"ranking": rankings, "selected": winner,
              "elapsed_seconds": time.monotonic() - start})
    np.savez_compressed(OUT / "inner_oof.npz", **oof, indices=development, classes=classes.astype(str))
    log(f"seleção congelada: {winner['name']}; CV={rankings[0]['inner_cv']['balanced_accuracy']:.4f}. Avaliar holdout uma vez.")
    final_configs = [winner]
    baseline = next(c for c in CONFIGS if c["name"] == "mean5")
    if winner["name"] != "mean5":
        final_configs.append(baseline)
    final = run_candidates(df, features, concat, unit, development, holdout, classes, final_configs)
    final_named = {"selected": final[winner["name"]], "baseline_mean5": final["mean5"]}
    y = df["model"].to_numpy()[holdout]
    result = {
        "protocol": protocol, "ranking_inner_cv": rankings, "selected": winner,
        "holdout": {name: metrics(y, p, classes) for name, p in final_named.items()},
        "bootstrap": bootstrap_prompts(y, final_named, df["prompt_id"].to_numpy()[holdout], classes),
        "elapsed_seconds": time.monotonic() - start,
    }
    np.savez_compressed(OUT / "holdout_predictions.npz", **final_named, indices=holdout, classes=classes.astype(str))
    save_json(OUT / "resultados.json", result)
    log(json.dumps({"selected": winner, "holdout": result["holdout"], "bootstrap": result["bootstrap"],
                    "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("once")
    main()
