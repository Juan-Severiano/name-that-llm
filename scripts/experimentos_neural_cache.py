"""Cabeças neurais CPU sobre embeddings congelados, sem usar holdout na seleção.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/experimentos_neural_cache.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from experimentos_fusao import LAYERS, SEED, bootstrap_prompts, digest, lr, metrics, save_json

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "experimentos_neural_cache"
CONFIGS = [
    {"name": "concat_lr_C0.0005", "kind": "linear", "C": 0.0005},
    {"name": "direct_mlp128", "kind": "mlp", "hidden": 128, "input": "direct", "dropout": 0.4, "lr": 0.0003},
    {"name": "direct_mlp256", "kind": "mlp", "hidden": 256, "input": "direct", "dropout": 0.5, "lr": 0.0003},
    {"name": "pca256_mlp128", "kind": "mlp", "hidden": 128, "input": "pca", "dropout": 0.3, "lr": 0.001},
    {"name": "pca256_mlp128_gradpenalty", "kind": "mlp", "hidden": 128, "input": "pca", "dropout": 0.3,
     "lr": 0.001, "gradient_penalty": 0.01},
    {"name": "pca256_mlp128_supcon", "kind": "mlp", "hidden": 128, "input": "pca", "dropout": 0.3,
     "lr": 0.001, "supervised_contrastive": 0.1, "temperature": 0.1},
]
EPOCHS, BATCH, WEIGHT_DECAY = 60, 64, 0.01


def log(message):
    print(message, flush=True)


class Head(nn.Module):
    def __init__(self, n_features, n_classes, hidden, dropout):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.feature(x)
        return self.classifier(self.dropout(h)), h


def contrastive(hidden, labels, prompts, temperature):
    hidden = F.normalize(hidden, dim=1)
    logits = hidden @ hidden.T / temperature
    diagonal = torch.eye(len(hidden), dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & (prompts[:, None] != prompts[None, :]) & ~diagonal
    log_probability = logits - torch.logsumexp(logits.masked_fill(diagonal, -1e9), dim=1, keepdim=True)
    count = positives.sum(dim=1)
    eligible = count > 0
    if not eligible.any():
        return hidden.sum() * 0
    return (-(log_probability * positives).sum(dim=1) / count.clamp(min=1))[eligible].mean()


def predict(model, x):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            scores, _ = model(x[start:start + BATCH])
            result.append(scores.softmax(dim=1).numpy())
    return np.concatenate(result)


def fit_neural(x, y, prompts, valid, classes, config, seed):
    torch.manual_seed(seed)
    x, valid = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(valid, dtype=np.float32))
    labels, groups = torch.from_numpy(np.searchsorted(classes, y)), torch.from_numpy(pd.factorize(prompts, sort=True)[0])
    model = Head(x.shape[1], len(classes), config["hidden"], config["dropout"]).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY, foreach=False)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    generator = torch.Generator().manual_seed(seed)
    trace, started = [], time.monotonic()
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(len(x), generator=generator)
        total = 0.0
        for start in range(0, len(x), BATCH):
            ix = permutation[start:start + BATCH]
            xb, yb, gb = x[ix], labels[ix], groups[ix]
            if config.get("gradient_penalty"):
                xb.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            scores, hidden = model(xb)
            loss = F.cross_entropy(scores, yb)
            if config.get("gradient_penalty"):
                gradient = torch.autograd.grad(scores.gather(1, yb[:, None]).sum(), xb, create_graph=True)[0]
                loss = loss + config["gradient_penalty"] * gradient.square().sum(dim=1).mean()
            if config.get("supervised_contrastive"):
                loss = loss + config["supervised_contrastive"] * contrastive(hidden, yb, gb, config["temperature"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(ix)
        schedule.step()
        if epoch in [0, 19, 39, 59]:
            row = {"epoch": epoch + 1, "training_objective": total / len(x), "elapsed_seconds": time.monotonic() - started}
            trace.append(row)
            log(f"  {config['name']} epoch={epoch + 1} objective={row['training_objective']:.4f} sec={row['elapsed_seconds']:.1f}")
    prediction = predict(model, valid)
    train_acc = float(np.mean(classes[predict(model, x).argmax(axis=1)] == y))
    return prediction, {"trace": trace, "train_accuracy": train_acc, "elapsed_seconds": time.monotonic() - started}


def evaluate_configs(df, concat, train, valid, classes, configs, seed):
    scaler = StandardScaler()
    x, v = scaler.fit_transform(concat[train]), scaler.transform(concat[valid])
    pca_data = None
    if any(c.get("input") == "pca" for c in configs):
        pca = PCA(n_components=256, svd_solver="randomized", random_state=SEED, iterated_power=3)
        px = pca.fit_transform(x)
        pv = pca.transform(v)
        pscale = StandardScaler()
        pca_data = pscale.fit_transform(px), pscale.transform(pv)
    y = df["model"].to_numpy()
    result, traces = {}, {}
    for number, config in enumerate(configs):
        name = config["name"]
        started = time.monotonic()
        if config["kind"] == "linear":
            p = lr(config["C"]).fit(x, y[train]).predict_proba(v)
            traces[name] = {"elapsed_seconds": time.monotonic() - started}
        else:
            a, b = pca_data if config["input"] == "pca" else (x, v)
            p, traces[name] = fit_neural(a, y[train], df["prompt_id"].to_numpy()[train], b, classes, config, seed + number)
        assert p.shape == (len(valid), len(classes)) and np.isfinite(p).all()
        result[name] = p
        score = balanced_accuracy_score(y[valid], classes[p.argmax(axis=1)])
        log(f"{name}: validation={score:.4f} elapsed={time.monotonic() - started:.1f}s")
    return result, traces


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "resultados.json").exists():
        raise SystemExit("Holdout já avaliado; não repetir para escolher configurações.")
    started = time.monotonic()
    df = pd.read_csv(ROOT / "data" / "train.csv", keep_default_na=False)
    classes = np.sort(df["model"].unique())
    development, holdout = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df["prompt_id"]))
    features = []
    for name, (file, layers) in LAYERS.items():
        raw = np.load(ROOT / "data" / "emb" / f"{file}.treino.npy", mmap_mode="r")
        assert raw.shape[0] == len(df)
        features.append(np.concatenate([raw[:, layer] for layer in layers], axis=1).astype(np.float32))
    concat = np.concatenate(features, axis=1)
    del features
    protocol = {
        "seed": SEED, "configs_predeclared": CONFIGS, "epochs_fixed": EPOCHS, "batch_size": BATCH,
        "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0, "optimizer": "AdamW with cosine LR decay",
        "threads": 1, "device": "cpu", "input_dimensions": concat.shape[1],
        "development_rows": len(development), "holdout_rows": len(holdout), "classes": classes.tolist(),
        "holdout_prompt_sha256": digest(np.sort(df["prompt_id"].iloc[holdout].unique())),
        "holdout_id_sha256": digest(df["id"].iloc[holdout]), "development_id_sha256": digest(df["id"].iloc[development]),
        "inner_cv": "GroupKFold3 shuffle=False; no early stopping or calibration on validation folds",
        "selection": "highest pooled inner CV balanced accuracy; declared order for ties",
        "holdout_predeclared": ["selected inner-CV config", "concat_lr_C0.0005"],
        "limitations": [
            "Layers/encoders previously selected using all 150 prompts; holdout is historically exposed.",
            "This same holdout was evaluated by an earlier independent frozen-embedding campaign; no settings here use its labels or errors.",
            "Six new configs are fixed before this run. Scaler and PCA are fitted only on each training fold.",
            "Supervised contrastive positives share author label and must have different prompt_id.",
        ],
    }
    save_json(OUT / "protocolo.json", protocol)
    np.savez_compressed(OUT / "split.npz", development=development, holdout=holdout, classes=classes.astype(str))
    oof = {c["name"]: np.zeros((len(development), len(classes))) for c in CONFIGS}
    traces, fold_scores = {}, {c["name"]: [] for c in CONFIGS}
    folds = GroupKFold(n_splits=3).split(development, groups=df["prompt_id"].iloc[development])
    for fold, (train, valid) in enumerate(folds):
        log(f"Fold {fold + 1}/3 train={len(train)} validation={len(valid)}")
        cache = OUT / f"inner_fold{fold + 1}.npz"
        trace_path = OUT / f"training_fold{fold + 1}.json"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as loaded:
                predictions = {name: loaded[name] for name in oof}
            trace = __import__("json").loads(trace_path.read_text())
        else:
            predictions, trace = evaluate_configs(df, concat, development[train], development[valid], classes, CONFIGS, SEED + fold * 17)
            np.savez_compressed(cache, **predictions, indices=development[valid], classes=classes.astype(str))
            save_json(trace_path, trace)
        traces[str(fold + 1)] = trace
        for name, p in predictions.items():
            oof[name][valid] = p
            fold_scores[name].append(float(balanced_accuracy_score(df["model"].iloc[development[valid]], classes[p.argmax(axis=1)])))
        save_json(OUT / "progresso.json", {"completed_folds": fold + 1, "fold_scores": fold_scores,
                                         "elapsed_seconds": time.monotonic() - started})
    rankings = [{"config": c, "inner_cv": metrics(df["model"].to_numpy()[development], oof[c["name"]], classes),
                 "fold_scores": fold_scores[c["name"]]} for c in CONFIGS]
    rankings.sort(key=lambda row: row["inner_cv"]["balanced_accuracy"], reverse=True)
    selected = rankings[0]["config"]
    save_json(OUT / "selecao_sem_holdout.json", {"ranking": rankings, "selected": selected})
    np.savez_compressed(OUT / "inner_oof.npz", **oof, indices=development, classes=classes.astype(str))
    log(f"Selection frozen: {selected['name']}; CV={rankings[0]['inner_cv']['balanced_accuracy']:.4f}")
    configs = [selected] + ([] if selected["kind"] == "linear" else [CONFIGS[0]])
    predictions, final_trace = evaluate_configs(df, concat, development, holdout, classes, configs, SEED + 99)
    final = {"selected": predictions[selected["name"]], "baseline_concat_lr": predictions[CONFIGS[0]["name"]]}
    bootstrap = bootstrap_prompts(df["model"].to_numpy()[holdout],
        {"selected": final["selected"], "baseline_mean5": final["baseline_concat_lr"]},
        df["prompt_id"].to_numpy()[holdout], classes)
    bootstrap["results"]["baseline_concat_lr"] = bootstrap["results"].pop("baseline_mean5")
    result = {
        "protocol": protocol, "selected": selected, "ranking_inner_cv": rankings,
        "holdout_comparator": "concat_lr_C0.0005",
        "holdout": {name: metrics(df["model"].to_numpy()[holdout], p, classes) for name, p in final.items()},
        "bootstrap": bootstrap,
        "training_traces": traces, "final_training_trace": final_trace,
        "elapsed_seconds": time.monotonic() - started,
    }
    np.savez_compressed(OUT / "holdout_predictions.npz", **final, indices=holdout, classes=classes.astype(str))
    save_json(OUT / "resultados.json", result)
    log(f"DONE {selected['name']} holdout={result['holdout']['selected']['balanced_accuracy']:.4f}; sec={result['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
