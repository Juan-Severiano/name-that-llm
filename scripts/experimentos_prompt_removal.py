"""Prompt removal / two-way Gaussian, local, sem executar código de terceiros.

Ideia matemática lida nos notebooks públicos kirilltym e prvsiyan: decompor
embedding em média da classe + efeito compartilhado do prompt + ruído.
Adaptação independente: nenhum at-most-once, repel, Sinkhorn ou quota por classe.
Grupos de inferência recebem somente textos e embeddings semânticos fixos.

  .venv/bin/python scripts/experimentos_prompt_removal.py selftest
  .venv/bin/python scripts/experimentos_prompt_removal.py pilot
  .venv/bin/python scripts/experimentos_prompt_removal.py run
"""

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import ast
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/experimentos_prompt_removal"
SEED = 20261103
TEMPERATURE = 3.
DIMENSIONS = (64, 128)


def log(message):
    print(f"{datetime.now(ZoneInfo('America/Fortaleza')).isoformat(timespec='seconds')} {message}", flush=True)


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_literal():
    """AST/literal_eval apenas: nunca exec/import de qualquer célula remota."""
    source = next((ROOT / "data/public_notebooks/teach_remove_prompt").glob("*.ipynb"))
    nb = json.loads(source.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_GAUSSIAN_SOURCE" for t in node.targets):
                text = ast.literal_eval(node.value)
                if not isinstance(text, str):
                    raise TypeError("source não é literal string")
                OUT.mkdir(parents=True, exist_ok=True)
                (OUT / "gaussian_source_literal.txt").write_text(text)
                return {"notebook_sha256": sha(source), "literal_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "execution": False, "attribution": "Ideia de shared-prompt Gaussian: prvsiyan; notebook consultado kirilltym/0-600-teach-the-model-remove-the-prompt",
                        "author_lb_claims_unverified": ["V19 0.594", "V4 0.595"],
                        "differences": "Implementação independente, nenhuma injetividade/repulsão/quota de classe/Sinkhorn; apenas textos para inferir grupos."}
    raise ValueError("literal _GAUSSIAN_SOURCE ausente")


def centered(X, groups):
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    means = np.zeros((len(counts), X.shape[1]), dtype=X.dtype)
    np.add.at(means, inv, X)
    means /= counts[:, None]
    return X - means[inv]


def infer_groups(texts, semantic_embeddings, content_only=False):
    """A interface não aceita prompt_id/model: agrupamento totalmente textual."""
    if len(texts) < 2:
        return np.arange(len(texts))
    try:
        vector = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=100_000, dtype=np.float32)
        content = vector.fit_transform(texts)
        distance = np.clip(1 - (content @ content.T).toarray(), 0, 1)
    except ValueError:  # pool de strings vazias ou só stopwords
        distance = np.ones((len(texts), len(texts)))
    if content_only:
        median = max(float(np.median(distance[np.triu_indices(len(texts), 1)])), 1e-6)
        distance = distance / median
        threshold = .90
    else:
        E = np.asarray(semantic_embeddings, dtype=np.float64)
        E = normalize(E - E.mean(axis=0))
        distance = .5 * (distance + np.clip(1 - E @ E.T, 0, 2))
        threshold = .60
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    groups = fcluster(tree, threshold, criterion="distance")
    return np.unique(groups, return_inverse=True)[1]


def cluster_diagnostics(groups, truth):
    """Somente relatório: nunca chamado dentro de infer_groups/predict."""
    frame = pd.DataFrame({"g": groups, "t": truth})
    pairs = lambda n: np.sum(n * (n-1) // 2)
    tp = int(pairs(frame.value_counts().to_numpy()))
    predicted_pairs = int(pairs(frame.groupby("g").size().to_numpy()))
    true_pairs = int(pairs(frame.groupby("t").size().to_numpy()))
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return {"pair_precision": tp / predicted_pairs if predicted_pairs else None, "pair_recall": tp / true_pairs if true_pairs else None,
            "true_positive_pairs": tp, "predicted_pairs": predicted_pairs, "true_pairs": true_pairs,
            "adjusted_rand": float(adjusted_rand_score(truth, groups)), "clusters": len(counts),
            "grouped_fraction": float(np.mean(counts[inv] > 1)), "singleton_rows": int(np.sum(counts[inv] == 1)),
            "group_size_min_median_max": [int(counts.min()), float(np.median(counts)), int(counts.max())]}


def reduction(X, groups, max_dimension=128):
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X).astype(np.float32)
    residual = centered(Z, groups)
    dimension = min(max_dimension, len(X) - len(np.unique(groups)) - 1, X.shape[1])
    _, _, vt = randomized_svd(residual, n_components=dimension, n_iter=4, random_state=SEED)
    return {"scaler": scaler, "V": vt.T}, Z @ vt.T


def two_way(Z, y, groups, classes, iterations=8):
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    effect = np.zeros((len(counts), Z.shape[1]))
    means = np.zeros((classes, Z.shape[1]))
    for _ in range(iterations):
        residual = Z - effect[inv]
        for c in range(classes):
            means[c] = residual[y == c].mean(axis=0)
        effect[:] = 0
        np.add.at(effect, inv, Z - means[y])
        effect /= counts[:, None]
        offset = effect.mean(axis=0)
        effect -= offset
        means += offset
    return means, effect, inv, counts


def shrink(S, alpha):
    return (1-alpha) * S + alpha * np.trace(S) / len(S) * np.eye(len(S))


def fit_gaussian(Z, y, groups, classes):
    Z = np.asarray(Z, dtype=np.float64)
    means, effect, inv, counts = two_way(Z, y, groups, classes)
    E = Z - means[y] - effect[inv]
    degrees = max(len(Z)-len(counts)-classes+1, 1)
    raw = E.T @ E / degrees
    alpha_e = float(ledoit_wolf_shrinkage(E, assume_centered=True))
    Se = shrink(raw, alpha_e)
    Se += np.eye(Z.shape[1]) * max(float(np.trace(Se) / len(Se) * 1e-7), 1e-9)
    P = effect - effect.mean(axis=0)
    raw_prompt = P.T @ P / max(len(P)-1, 1) - np.mean(1 / counts) * raw
    alpha_p = float(ledoit_wolf_shrinkage(P, assume_centered=True))
    trace = max(float(np.trace(raw_prompt)), 1e-8 * float(np.trace(Se)))
    Sp = (1-alpha_p) * raw_prompt + alpha_p * trace / len(Se) * np.eye(len(Se))
    L = np.linalg.cholesky(Se)
    inverse = np.linalg.inv(L)
    joint = inverse @ Sp @ inverse.T
    values, Q = np.linalg.eigh((joint + joint.T) / 2)
    lam = np.maximum(values, 1e-4)
    A = Q.T @ inverse
    return {"A": A, "means": means @ A.T, "lam": lam,
            "info": {"dimension": Z.shape[1], "shrink_residual": alpha_e, "shrink_prompt": alpha_p,
                     "lambda_percentiles": np.quantile(lam, [.05,.5,.95,1]).tolist()}}


def singleton(U, model):
    residual = U[:, None] - model["means"][None]
    return -.5 * np.sum(residual ** 2 / (1 + model["lam"]), axis=2)


def predict_gaussian(Z, model, groups=None):
    U = np.asarray(Z, dtype=np.float64) @ model["A"].T
    plain = softmax(singleton(U, model) / TEMPERATURE, axis=1)
    if groups is None:
        return plain
    result = plain.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < 2:
            continue
        q = result[idx].copy()
        v = model["lam"] / (1 + (len(idx)-1) * model["lam"])
        for _ in range(6):
            # Cavity estimate excludes this row; repeated classes remain fully legal.
            errors = U[idx] - q @ model["means"]
            prompt = (errors.sum(axis=0)[None] - errors) * v
            residual = U[idx, None] - prompt[:, None] - model["means"][None]
            logits = -.5 * np.sum(residual ** 2 / (1+v), axis=2)
            q = softmax(logits / TEMPERATURE, axis=1)
        result[idx] = q
    assert np.isfinite(result).all()
    return result


def metrics(y, p, classes):
    predicted = p.argmax(axis=1)
    recalls = [float(np.mean(predicted[y == c] == c)) if np.any(y == c) else None for c in range(len(classes))]
    return {"balanced_accuracy": float(np.mean([r for r in recalls if r is not None])),
            "accuracy": float(np.mean(predicted == y)), "log_loss": float(log_loss(y, p, labels=np.arange(len(classes)))),
            "recall_by_class": dict(zip(classes, recalls)), "class_counts": dict(zip(classes, np.bincount(y, minlength=len(classes)).tolist()))}


def load(seed=SEED):
    df = pd.read_csv(ROOT / "data/train.csv", keep_default_na=False)
    protocol = json.loads((ROOT / "runs/validacao_pesada/split.json").read_text())
    classes = np.asarray(protocol["classes"])
    y = np.array([list(classes).index(c) for c in df.model])
    dev, holdout = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(df, groups=df.prompt_id))
    if seed == SEED:
        np.testing.assert_array_equal(dev, protocol["dev_indices"])
        np.testing.assert_array_equal(holdout, protocol["holdout_indices"])
    views = {name: np.load(ROOT / "data/artefatos" / f"emb_{name}.npy", allow_pickle=False).astype(np.float32)
             for name in ("bge-small-en-v1.5", "roberta-base", "deberta-v3-base", "Qwen3-0.6B")}
    style = ROOT / "runs/experimentos_styledistance/train_embeddings.npy"
    if style.exists():
        S = np.load(style, mmap_mode="r", allow_pickle=False)
        views["StyleDistance_L12"] = np.asarray(S[:,12], dtype=np.float32)
    sem = np.asarray(np.load(ROOT / "data/emb/BAAI__bge-small-en-v1.5.treino.npy", mmap_mode="r", allow_pickle=False)[:,12], dtype=np.float32)
    inner = [(dev[t], dev[v]) for t, v in GroupKFold(3).split(df.iloc[dev], y[dev], df.prompt_id.iloc[dev])]
    return df, y, classes, dev, holdout, views, sem, inner


def pilot():
    source = extract_literal()
    df, y, classes, dev, holdout, views, sem, inner = load()
    train, valid = inner[0]
    X = views["bge-small-en-v1.5"]
    groups_train = df.prompt_id.iloc[train].to_numpy()
    groups_valid = df.prompt_id.iloc[valid].to_numpy()
    inferred = infer_groups(df.text.iloc[valid].tolist(), sem[valid])
    content = infer_groups(df.text.iloc[valid].tolist(), sem[valid], content_only=True)
    with threadpool_limits(limits=1):
        plain = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=1000)).fit(X[train], y[train])
        centered_head = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=1000)).fit(centered(X[train], groups_train), y[train])
        lr = {"plain": metrics(y[valid], plain.predict_proba(X[valid]), classes),
              "oracle_center": metrics(y[valid], centered_head.predict_proba(centered(X[valid], groups_valid)), classes),
              "inferred_center": metrics(y[valid], centered_head.predict_proba(centered(X[valid], inferred)), classes)}
        reducer, Z = reduction(X[train], groups_train)
        Zvalid = reducer["scaler"].transform(X[valid]) @ reducer["V"]
        head = fit_gaussian(Z, y[train], groups_train, len(classes))
        gauss = {"plain": metrics(y[valid], predict_gaussian(Zvalid, head), classes),
                 "oracle_groups": metrics(y[valid], predict_gaussian(Zvalid, head, groups_valid), classes),
                 "inferred_groups": metrics(y[valid], predict_gaussian(Zvalid, head, inferred), classes)}
    result = {"source": source, "split": "first inner fold, development only; holdout untouched", "rows": len(valid),
              "clusters_mixed": cluster_diagnostics(inferred, groups_valid), "clusters_content_only": cluster_diagnostics(content, groups_valid),
              "lr": lr, "gaussian": gauss, "note": "oracle usa prompt_id verdadeiro somente como diagnóstico indisponível em produção"}
    save("pilot.json", result)
    log(json.dumps({"lr": {k:v["balanced_accuracy"] for k,v in lr.items()}, "gaussian": {k:v["balanced_accuracy"] for k,v in gauss.items()}, "clusters": result["clusters_mixed"]}))


def fit_views_predict(views, train, valid, y, train_groups, groups, classes, dimensions=DIMENSIONS):
    predictions, plain, oracle, fitted = {}, {}, {}, {}
    for name, X in views.items():
        t0 = time.monotonic()
        reducer, Z = reduction(X[train], train_groups)
        Zvalid = reducer["scaler"].transform(X[valid]) @ reducer["V"]
        for dimension in dimensions:
            d = min(dimension, Z.shape[1])
            head = fit_gaussian(Z[:,:d], y[train], train_groups, len(classes))
            key = f"{name}_d{dimension}"
            predictions[key] = predict_gaussian(Zvalid[:,:d], head, groups)
            plain[key] = predict_gaussian(Zvalid[:,:d], head)
            fitted[key] = (reducer, head, d)
        log(f"Gaussian {name} dimensões{dimensions}: {time.monotonic()-t0:.1f}s")
    for dimension in dimensions:
        keys = [k for k in predictions if k.endswith(f"_d{dimension}")]
        predictions[f"mean_d{dimension}"] = np.mean([predictions[k] for k in keys], axis=0)
        plain[f"mean_d{dimension}"] = np.mean([plain[k] for k in keys], axis=0)
    return predictions, plain, fitted


def predict_selected(key, fitted, views, indices, groups):
    keys = [k for k in fitted if k.endswith(key.removeprefix("mean"))] if key.startswith("mean_") else [key]
    ps = []
    for k in keys:
        name = k.rsplit("_d",1)[0]
        reducer, head, d = fitted[k]
        Z = reducer["scaler"].transform(views[name][indices]) @ reducer["V"][:,:d]
        ps.append(predict_gaussian(Z, head, groups))
    return np.mean(ps, axis=0)


def run():
    if (OUT / "holdout.json").exists():
        log("holdout já congelado; não reabrir seleção")
        return
    source = extract_literal()
    df, y, classes, dev, holdout, views, sem, inner = load()
    group_truth = df.prompt_id.to_numpy()
    config_names = [f"{v}_d{d}" for v in views for d in DIMENSIONS] + [f"mean_d{d}" for d in DIMENSIONS]
    save("protocol.json", {"seed": SEED, "train_sha256": sha(ROOT / "data/train.csv"), "source": source,
                           "candidates": config_names, "candidates_count": len(config_names), "temperature": TEMPERATURE,
                           "cluster_threshold_fixed": .60, "cluster_inputs": ["text", "BGE final-layer semantic embedding"],
                           "forbidden_constraints": ["one class per group", "class quotas", "repel", "Sinkhorn"],
                           "selection": "inner GKF3 dev only, fixed GSS20% seed20261103 holdout diagnostic",
                           "historical_exposure": "all prompts already used historically; campaign holdout is not pristine"})
    oof = {k: np.empty((len(dev),len(classes))) for k in config_names}
    diagnostics = []
    mapping = {int(g): i for i,g in enumerate(dev)}
    with threadpool_limits(limits=1):
        for fold, (train, valid) in enumerate(inner):
            inferred = infer_groups(df.text.iloc[valid].tolist(), sem[valid])
            # prompt truth is used here only to REPORT clustering, never to build groups/predictions.
            diagnostics.append(cluster_diagnostics(inferred, group_truth[valid]))
            predictions, _, _ = fit_views_predict(views, train, valid, y, group_truth[train], inferred, classes)
            rows = [mapping[int(i)] for i in valid]
            for k,p in predictions.items():
                oof[k][rows] = p
            log(f"fold{fold+1}: cluster precision={diagnostics[-1]['pair_precision']:.4f}, recall={diagnostics[-1]['pair_recall']:.4f}")
    ranking = sorted(({"name": k, **metrics(y[dev],p,classes)} for k,p in oof.items()), key=lambda a:-a["balanced_accuracy"])
    save("development.json", {"ranking": ranking, "cluster_diagnostics": diagnostics})
    np.savez_compressed(OUT / "inner_oof.npz", **oof, indices=dev, classes=classes.astype(str))
    winner = ranking[0]["name"]
    save("selection_frozen.json", {"winner": winner, "inner_balanced_accuracy": ranking[0]["balanced_accuracy"], "holdout_metrics_read": False})
    log(f"SELEÇÃO CONGELADA: {winner} inner={ranking[0]['balanced_accuracy']:.4f}")
    # Fit final: only the selected dimension/views, and one holdout evaluation.
    dimension = int(winner.rsplit("_d",1)[1])
    selected_views = views if winner.startswith("mean_") else {winner.rsplit("_d",1)[0]: views[winner.rsplit("_d",1)[0]]}
    inferred = infer_groups(df.text.iloc[holdout].tolist(), sem[holdout])
    with threadpool_limits(limits=1):
        predictions, independent, fitted = fit_views_predict(selected_views, dev, holdout, y, group_truth[dev], inferred, classes, dimensions=(dimension,))
        p = predictions[winner]
        p_plain = independent[winner]
        p_oracle = predict_selected(winner, fitted, selected_views, holdout, group_truth[holdout])
    result = {"winner": winner, "normal": metrics(y[holdout],p,classes), "without_grouping": metrics(y[holdout],p_plain,classes),
              "oracle_prompt_diagnostic": metrics(y[holdout],p_oracle,classes), "clusters": cluster_diagnostics(inferred, group_truth[holdout]),
              "rows":len(holdout), "prompts":len(np.unique(group_truth[holdout]))}
    save("holdout.json", result)
    np.savez_compressed(OUT / "holdout_predictions.npz", indices=holdout, classes=classes.astype(str), probas=p, plain=p_plain, oracle=p_oracle, inferred_groups=inferred)
    # Stress protocols were fixed before inspecting their results. Labels construct an imbalanced
    # diagnostic subset but are never passed into inference or used to adapt class priors.
    rng = np.random.default_rng(SEED)
    sparse = []
    for g in np.unique(group_truth[holdout]):
        members = holdout[group_truth[holdout] == g]
        n = int(rng.choice([1,1,2,3,5,9]))
        sparse.extend(rng.choice(members, min(n,len(members)), replace=False).tolist())
    keep = .15 + .85 * y[holdout] / max(len(classes)-1,1)
    imbalanced = holdout[rng.random(len(holdout)) < keep]
    stress = {}
    with threadpool_limits(limits=1):
        for name, indices in (("rare_small_groups", np.sort(sparse)), ("unequal_class_prevalence", imbalanced)):
            gs = infer_groups(df.text.iloc[indices].tolist(), sem[indices])
            pp = predict_selected(winner, fitted, selected_views, indices, gs)
            pn = predict_selected(winner, fitted, selected_views, indices, None)
            stress[name] = {"rows":len(indices), "metrics":metrics(y[indices],pp,classes), "without_grouping":metrics(y[indices],pn,classes),
                            "clusters":cluster_diagnostics(gs,group_truth[indices]), "no_prior_adaptation_or_label_allocation":True}
    save("stress.json", stress)
    log(json.dumps({"winner":winner,"normal":result['normal']['balanced_accuracy'],"independent":result['without_grouping']['balanced_accuracy'],
                    "oracle":result['oracle_prompt_diagnostic']['balanced_accuracy'],"stress":{k:v['metrics']['balanced_accuracy'] for k,v in stress.items()}}))


def selftest():
    model = {"A":np.eye(2), "means":np.array([[0.,0.],[8.,0.]]),"lam":np.array([.1,.1])}
    Z = np.array([[.1,0.],[.2,.1],[.3,-.1],[.2,.2]])
    p = predict_gaussian(Z,model,np.zeros(4,dtype=int))
    assert np.all(p.argmax(axis=1)==0)  # n=4 > C=2, repeated classes explicitly permitted.
    np.testing.assert_allclose(p.sum(axis=1),1)
    single = predict_gaussian(Z,model,np.arange(4))
    np.testing.assert_allclose(single,predict_gaussian(Z,model))
    permutation=np.array([3,1,0,2])
    perm=predict_gaussian(Z[permutation],model,np.zeros(4,dtype=int))
    np.testing.assert_allclose(perm,p[permutation],atol=1e-12)
    inferred=infer_groups(["", ""],np.zeros((2,3)))
    assert len(inferred)==2
    save("selftest.json", {"repeated_classes_allowed":True,"group_larger_than_class_count":True,
                           "singletons_equal_independent":True,"row_permutation_invariant":True,"empty_texts":True})
    log("selftest CPU passou")


def expanded_metrics(y, p, classes):
    p=np.asarray(p,dtype=np.float64)
    roundoff=float(np.max(np.abs(p.sum(axis=1)-1)))
    p=p/p.sum(axis=1,keepdims=True)  # remove only FP32 row-sum roundoff, not a calibration fit
    result = metrics(y, p, classes)
    result["max_probability_row_sum_roundoff_before_normalization"]=roundoff
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = predicted == y
    ece = 0.
    calibration = []
    for low, high in zip(np.linspace(0,1,11)[:-1], np.linspace(0,1,11)[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            accuracy = float(correct[mask].mean())
            certainty = float(confidence[mask].mean())
            ece += mask.mean() * abs(accuracy-certainty)
            calibration.append({"range":[float(low),float(high)],"count":int(mask.sum()),"accuracy":accuracy,"confidence":certainty})
    onehot = np.eye(len(classes))[y]
    result.update({"ece10":float(ece),"brier_multiclass":float(np.mean(np.sum((p-onehot)**2,axis=1))),
                   "confidence_mean":float(confidence.mean()),"calibration_bins":calibration,
                   "class_details":{c:{"recall":result["recall_by_class"][c],
                                        "precision":float(np.mean(y[predicted==j]==j)) if np.any(predicted==j) else None,
                                        "true_fraction":float(np.mean(y==j)),"predicted_fraction":float(np.mean(predicted==j)),
                                        "mean_true_probability":float(p[y==j,j].mean()) if np.any(y==j) else None,
                                        "brier_ovr":float(np.mean((p[:,j]-(y==j))**2))} for j,c in enumerate(classes)}})
    family = lambda c: "qwen" if c.startswith("qwen") else c.split("-")[0]
    fs = np.array([family(c) for c in classes])
    families = np.unique(fs)
    fp = np.column_stack([p[:,fs==f].sum(axis=1) for f in families])
    fy = np.array([list(families).index(fs[c]) for c in y])
    fmetrics = metrics(fy,fp,families)
    result["family"] = fmetrics
    family_correct=fs[predicted]==fs[y]
    family_recalls={f:float(np.mean(family_correct[fs[y]==f])) if np.any(fs[y]==f) else None for f in families}
    result["family_of_class_top1"]={"accuracy":float(family_correct.mean()),
                                   "balanced_accuracy":float(np.mean([v for v in family_recalls.values() if v is not None])),
                                   "recall_by_family":family_recalls}
    result["family_probability_mass_note"]="family above predicts argmax of summed class probabilities; large families receive more mass, especially with underconfident class probabilities. family_of_class_top1 maps the actual 18-class prediction instead."
    result["same_family_error_fraction"] = float(np.mean(fs[predicted[~correct]] == fs[y[~correct]])) if np.any(~correct) else 0.
    result["classes_absent_from_truth"] = [c for j,c in enumerate(classes) if not np.any(y==j)]
    result["balanced_accuracy_scope"] = "mean recall over truth classes present; missing classes reported explicitly"
    return result


def paired_prompt(y, candidate, baseline, prompts, classes):
    unique, inv = np.unique(prompts,return_inverse=True)
    total = np.zeros((len(unique),len(classes)))
    a = np.zeros_like(total)
    b = np.zeros_like(total)
    np.add.at(total,(inv,y),1)
    np.add.at(a,(inv,y),candidate.argmax(axis=1)==y)
    np.add.at(b,(inv,y),baseline.argmax(axis=1)==y)
    weights = np.random.default_rng(SEED).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=5000)
    denominator = weights @ total
    with np.errstate(divide="ignore",invalid="ignore"):
        score_a = np.nanmean((weights@a)/denominator,axis=1)
        score_b = np.nanmean((weights@b)/denominator,axis=1)
    delta = score_a-score_b
    prompt_delta = (a.sum(axis=1)-b.sum(axis=1))/total.sum(axis=1)
    ma,mb = metrics(y,candidate,classes),metrics(y,baseline,classes)
    return {"candidate_balanced_accuracy":ma["balanced_accuracy"],"baseline_balanced_accuracy":mb["balanced_accuracy"],
            "delta":ma["balanced_accuracy"]-mb["balanced_accuracy"],
            "candidate_ci95":np.quantile(score_a,[.025,.975]).tolist(),"baseline_ci95":np.quantile(score_b,[.025,.975]).tolist(),
            "delta_ci95":np.quantile(delta,[.025,.975]).tolist(),"bootstrap_fraction_delta_nonpositive":float(np.mean(delta<=0)),
            "prompts_won_tied_lost":[int(np.sum(prompt_delta>0)),int(np.sum(prompt_delta==0)),int(np.sum(prompt_delta<0))],
            "bootstrap":"5000 paired prompt-cluster resamples, conditional on fitted models/selection; not a new external test",
            "per_class_recall_delta":{c:float(ma['recall_by_class'][c]-mb['recall_by_class'][c]) if ma['recall_by_class'][c] is not None else None for c in classes}}


def choose_alpha(y, base, gaussian, classes, name):
    grid = []
    for alpha in (0.,.25,.5,.75,1.):
        p=(1-alpha)*base+alpha*gaussian
        grid.append({"alpha_gaussian":alpha,"balanced_accuracy":metrics(y,p,classes)["balanced_accuracy"]})
    winner=max(grid,key=lambda r:(r["balanced_accuracy"],-r["alpha_gaussian"]))
    save(name,{"grid":grid,"selected_alpha":winner["alpha_gaussian"],"selection_inputs":"development OOF only; freeze before outer predictions","retune_on_second_split":False})
    return winner["alpha_gaussian"]


def baseline_predict(df,views,train,valid,y):
    # Reuse the local owning implementation; importing it does not load any model/GPU.
    sys.path.insert(0,str(ROOT/"src"))
    from name_that_llm.pipeline import montar_tfidf, lr_embeddings
    parts=[]
    pipe=montar_tfidf(LogisticRegression(C=20,max_iter=2000)).fit(df.iloc[train],y[train])
    np.testing.assert_array_equal(pipe.classes_,np.arange(18))
    parts.append(pipe.predict_proba(df.iloc[valid]))
    for name in ("bge-small-en-v1.5","roberta-base","deberta-v3-base","Qwen3-0.6B"):
        model=lr_embeddings().fit(views[name][train],y[train])
        np.testing.assert_array_equal(model.classes_,np.arange(18))
        parts.append(model.predict_proba(views[name][valid]))
    return np.mean(parts,axis=0)


def stress_selected(df,y,classes,dev,holdout,views,sem,alpha,base_holdout):
    truth=df.prompt_id.to_numpy()
    inferred=infer_groups(df.text.iloc[holdout].tolist(),sem[holdout])
    _,_,fitted=fit_views_predict(views,dev,holdout,y,truth[dev],inferred,classes,dimensions=(128,))
    rng=np.random.default_rng(SEED+77)
    subsets={}
    for k in (1,2,3):
        subset=[]
        for g in np.unique(truth[holdout]):
            members=holdout[truth[holdout]==g]
            subset.extend(rng.choice(members,min(k,len(members)),replace=False).tolist())
        subsets[f"exactly_k{k}_per_prompt"]=np.sort(subset)
    mixture=[]
    for g in np.unique(truth[holdout]):
        members=holdout[truth[holdout]==g]
        k=int(rng.choice([1,2,3,5,9,18],p=[.30,.25,.20,.15,.07,.03]))
        mixture.extend(rng.choice(members,min(k,len(members)),replace=False).tolist())
    subsets["hypothetical_mixture"]=np.sort(mixture)
    probability=.15+.85*y[holdout]/max(len(classes)-1,1)
    subsets["imbalanced_classes"]=holdout[rng.random(len(holdout))<probability]
    mapping={int(v):i for i,v in enumerate(holdout)}
    results={}
    for name,indices in subsets.items():
        groups=infer_groups(df.text.iloc[indices].tolist(),sem[indices])
        pg=predict_selected("mean_d128",fitted,views,indices,groups)
        pb=base_holdout[[mapping[int(i)] for i in indices]]
        blend=(1-alpha)*pb+alpha*pg
        results[name]={"rows":len(indices),"cluster":cluster_diagnostics(groups,truth[indices]),
                       "baseline":expanded_metrics(y[indices],pb,classes),"gaussian":expanded_metrics(y[indices],pg,classes),
                       "frozen_blend":expanded_metrics(y[indices],blend,classes),
                       "baseline_accuracy":float(np.mean(pb.argmax(axis=1)==y[indices])),
                       "gaussian_accuracy":float(np.mean(pg.argmax(axis=1)==y[indices])),
                       "blend_accuracy":float(np.mean(blend.argmax(axis=1)==y[indices])),
                       "hypothesis_only":"post-selection stress, not optimization; mask uses true prompts only to construct scenario, never for inference"}
    return {"alpha_fixed":alpha,"mixture_assumption":{"counts":[1,2,3,5,9,18],"probabilities":[.30,.25,.20,.15,.07,.03],
                                                     "actual_hidden_pool_distribution_known":False},"results":results}


def robustness():
    if (OUT/"robustness_complete.json").exists():
        log("Robustez já registrada; não repetir seleção/holdouts")
        return
    df,y,classes,dev,holdout,views,sem,inner=load()
    with np.load(ROOT/"runs/experimentos_fusao/inner_oof.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["indices"],dev)
        np.testing.assert_array_equal(f["classes"],classes)
        baseline_dev=f["mean5"]
    with np.load(OUT/"inner_oof.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["indices"],dev)
        gaussian_dev=f["mean_d128"]
    alpha=choose_alpha(y[dev],baseline_dev,gaussian_dev,classes,"blend_frozen.json")
    # Original holdout has already been inspected in the first experiment; this is post-selection evidence.
    with np.load(ROOT/"runs/experimentos_fusao/holdout_predictions.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["indices"],holdout)
        baseline=f["baseline_mean5"]
    with np.load(OUT/"holdout_predictions.npz",allow_pickle=False) as f:
        np.testing.assert_array_equal(f["indices"],holdout)
        gaussian=f["probas"]
    blend=(1-alpha)*baseline+alpha*gaussian
    original={"seed":SEED,"alpha_fixed":alpha,"post_selection":True,
              "baseline":expanded_metrics(y[holdout],baseline,classes),"gaussian":expanded_metrics(y[holdout],gaussian,classes),
              "blend":expanded_metrics(y[holdout],blend,classes),
              "gaussian_vs_baseline":paired_prompt(y[holdout],gaussian,baseline,df.prompt_id.iloc[holdout],classes),
              "blend_vs_baseline":paired_prompt(y[holdout],blend,baseline,df.prompt_id.iloc[holdout],classes)}
    save("paired_original.json",original)
    np.savez_compressed(OUT/"blend_holdout_predictions.npz",indices=holdout,classes=classes.astype(str),baseline=baseline,gaussian=gaussian,probas=blend,alpha=alpha)
    with threadpool_limits(limits=1):
        save("stress_hard_original.json",stress_selected(df,y,classes,dev,holdout,views,sem,alpha,baseline))
    log(f"Original paired delta Gaussian={original['gaussian_vs_baseline']['delta']:.4f}, CI={original['gaussian_vs_baseline']['delta_ci95']}; alpha={alpha}")
    # Second split: all architecture/temperature/clustering/PCA dimension and alpha are frozen.
    second_seed=20261104
    df,y,classes,dev2,holdout2,views,sem,inner2=load(second_seed)
    save("second_split_protocol.json",{"seed":second_seed,"architecture":"five-view Gaussian d128 mean", "temperature":TEMPERATURE,
                                       "alpha_from_first_dev":alpha,"tuning":False,"inner_folds":3,
                                       "holdout_rows":len(holdout2),"holdout_prompts":df.prompt_id.iloc[holdout2].unique().tolist(),
                                       "overlap_original_holdout_rows":len(set(holdout)&set(holdout2)),
                                       "limitation":"Second random partition of the SAME historically exposed data; not independent new data or leaderboard confirmation."})
    oof_b=np.empty((len(dev2),18));oof_g=np.empty_like(oof_b)
    lookup={int(v):i for i,v in enumerate(dev2)}
    with threadpool_limits(limits=1):
        for fold,(train,valid) in enumerate(inner2):
            inferred=infer_groups(df.text.iloc[valid].tolist(),sem[valid])
            ps,_,_=fit_views_predict(views,train,valid,y,df.prompt_id.iloc[train].to_numpy(),inferred,classes,dimensions=(128,))
            pb=baseline_predict(df,views,train,valid,y)
            rows=[lookup[int(i)] for i in valid]
            oof_g[rows]=ps["mean_d128"];oof_b[rows]=pb
            log(f"Second split inner{fold+1}: gauss={metrics(y[valid],ps['mean_d128'],classes)['balanced_accuracy']:.4f}, base={metrics(y[valid],pb,classes)['balanced_accuracy']:.4f}")
        np.savez_compressed(OUT/"second_split_inner_oof.npz",indices=dev2,classes=classes.astype(str),baseline=oof_b,gaussian=oof_g)
        save("second_split_development.json",{"alpha_already_fixed":alpha,"no_retuning":True,
                                              "baseline":metrics(y[dev2],oof_b,classes),"gaussian":metrics(y[dev2],oof_g,classes),
                                              "blend":metrics(y[dev2],(1-alpha)*oof_b+alpha*oof_g,classes)})
        inferred=infer_groups(df.text.iloc[holdout2].tolist(),sem[holdout2])
        ps,_,_=fit_views_predict(views,dev2,holdout2,y,df.prompt_id.iloc[dev2].to_numpy(),inferred,classes,dimensions=(128,))
        pb=baseline_predict(df,views,dev2,holdout2,y)
        pg=ps["mean_d128"];pm=(1-alpha)*pb+alpha*pg
    second={"seed":second_seed,"alpha_fixed":alpha,"no_retuning":True,
            "baseline":expanded_metrics(y[holdout2],pb,classes),"gaussian":expanded_metrics(y[holdout2],pg,classes),"blend":expanded_metrics(y[holdout2],pm,classes),
            "gaussian_vs_baseline":paired_prompt(y[holdout2],pg,pb,df.prompt_id.iloc[holdout2],classes),
            "blend_vs_baseline":paired_prompt(y[holdout2],pm,pb,df.prompt_id.iloc[holdout2],classes),
            "clusters":cluster_diagnostics(inferred,df.prompt_id.iloc[holdout2].to_numpy())}
    save("paired_second_split.json",second)
    np.savez_compressed(OUT/"second_split_holdout_predictions.npz",indices=holdout2,classes=classes.astype(str),baseline=pb,gaussian=pg,probas=pm,alpha=alpha)
    save("robustness_complete.json",{"original_gaussian_delta":original['gaussian_vs_baseline']['delta'],
                                      "original_blend_delta":original['blend_vs_baseline']['delta'],
                                      "second_gaussian_delta":second['gaussian_vs_baseline']['delta'],
                                      "second_blend_delta":second['blend_vs_baseline']['delta'],"frozen_alpha":alpha})
    log(json.dumps({"original_delta":original['gaussian_vs_baseline']['delta'],"original_CI":original['gaussian_vs_baseline']['delta_ci95'],
                    "second_base":second['baseline']['balanced_accuracy'],"second_gauss":second['gaussian']['balanced_accuracy'],
                    "second_blend":second['blend']['balanced_accuracy'],"second_deltaCI":second['blend_vs_baseline']['delta_ci95'],"alpha":alpha}))


def route_small_groups(baseline,gaussian,groups):
    """Política experimental separada: nenhuma alteração no modelo Gaussian puro."""
    if baseline.shape!=gaussian.shape or len(groups)!=len(baseline) or not np.isfinite(baseline).all() or not np.isfinite(gaussian).all():
        raise ValueError("probabilidades/grupos inválidos; não gerar fallback silencioso")
    _,inv,counts=np.unique(groups,return_inverse=True,return_counts=True)
    use_gaussian=counts[inv]>=4
    result=baseline.copy()
    result[use_gaussian]=gaussian[use_gaussian]
    return result,use_gaussian


def small_group_policy():
    """Hipótese pós-diagnóstico, congelada antes do split 20261105; zero retuning."""
    prefix="policy_seed20261105"
    if (OUT/f"{prefix}_complete.json").exists():
        log("política já avaliada; não repetir seleção/holdout")
        return
    seed=20261105
    save(f"{prefix}_protocol.json",{"seed":seed,"rule":"inferred group size <4 => baseline mean5; >=4 => frozen meanGaussian five views d128",
                                    "threshold_tuned":False,"source_of_hypothesis":"parent proposal after original k1/k3 stress regressions",
                                    "parameters_frozen":{"dimension":128,"temperature":TEMPERATURE,"group_cut":.60,"views":5},
                                    "historical_exposure":"new partition of same historically exposed data, post-diagnostic hypothesis; not immaculate independent evidence",
                                    "pure_candidate_artifacts_modified":False})
    df,y,classes,dev,holdout,views,sem,_=load(seed)
    truth=df.prompt_id.to_numpy()
    inferred=infer_groups(df.text.iloc[holdout].tolist(),sem[holdout])
    with threadpool_limits(limits=1):
        ps,_,fitted=fit_views_predict(views,dev,holdout,y,truth[dev],inferred,classes,dimensions=(128,))
        baseline=baseline_predict(df,views,dev,holdout,y)
    gaussian=ps['mean_d128']
    policy,mask=route_small_groups(baseline,gaussian,inferred)
    results={"seed":seed,"rows":len(holdout),"gaussian_routed_rows":int(mask.sum()),"gaussian_routed_fraction":float(mask.mean()),
             "clusters":cluster_diagnostics(inferred,truth[holdout]),
             "baseline":expanded_metrics(y[holdout],baseline,classes),"pure_gaussian":expanded_metrics(y[holdout],gaussian,classes),
             "policy":expanded_metrics(y[holdout],policy,classes),
             "policy_vs_baseline":paired_prompt(y[holdout],policy,baseline,truth[holdout],classes),
             "pure_gaussian_vs_baseline":paired_prompt(y[holdout],gaussian,baseline,truth[holdout],classes),
             "policy_vs_pure_gaussian":paired_prompt(y[holdout],policy,gaussian,truth[holdout],classes)}
    save(f"{prefix}_holdout.json",results)
    np.savez_compressed(OUT/f"{prefix}_predictions.npz",indices=holdout,classes=classes.astype(str),baseline=baseline,
                        gaussian=gaussian,policy=policy,inferred_groups=inferred,gaussian_routed=mask)
    rng=np.random.default_rng(seed)
    subsets={}
    for k in (1,2,3):
        indices=[]
        for group in np.unique(truth[holdout]):
            members=holdout[truth[holdout]==group]
            indices.extend(rng.choice(members,min(k,len(members)),replace=False).tolist())
        subsets[f"k{k}"]=np.sort(indices)
    mixed=[]
    for group in np.unique(truth[holdout]):
        members=holdout[truth[holdout]==group]
        k=int(rng.choice([1,2,3,5,9,18],p=[.30,.25,.20,.15,.07,.03]))
        mixed.extend(rng.choice(members,min(k,len(members)),replace=False).tolist())
    subsets['hypothetical_mix']=np.sort(mixed)
    lookup={int(v):i for i,v in enumerate(holdout)}
    stress={}
    with threadpool_limits(limits=1):
        for name,indices in subsets.items():
            groups=infer_groups(df.text.iloc[indices].tolist(),sem[indices])
            pg=predict_selected('mean_d128',fitted,views,indices,groups)
            pb=baseline[[lookup[int(i)] for i in indices]]
            pp,route=route_small_groups(pb,pg,groups)
            if not route.any():
                np.testing.assert_array_equal(pp,pb)
            stress[name]={"rows":len(indices),"gaussian_routes":int(route.sum()),"policy_exactly_baseline_probabilities":bool(np.array_equal(pp,pb)),
                          "policy_same_baseline_top1_fraction":float(np.mean(pp.argmax(axis=1)==pb.argmax(axis=1))),
                          "clusters":cluster_diagnostics(groups,truth[indices]),
                          "baseline":expanded_metrics(y[indices],pb,classes),"pure_gaussian":expanded_metrics(y[indices],pg,classes),
                          "policy":expanded_metrics(y[indices],pp,classes),
                          "post_selection_stress_not_tuning":True}
    save(f"{prefix}_stress.json",stress)
    save(f"{prefix}_complete.json",{"baseline_ba":results['baseline']['balanced_accuracy'],"gaussian_ba":results['pure_gaussian']['balanced_accuracy'],
                                   "policy_ba":results['policy']['balanced_accuracy'],"policy_delta_ci95":results['policy_vs_baseline']['delta_ci95'],
                                   "parameters_or_threshold_retuned":False,"pure_artifacts_modified":False})
    log(json.dumps({"seed":seed,"baseline":results['baseline']['balanced_accuracy'],"pure_gaussian":results['pure_gaussian']['balanced_accuracy'],
                    "policy":results['policy']['balanced_accuracy'],"policy_deltaCI":results['policy_vs_baseline']['delta_ci95'],
                    "stress":{k:{"routes":v['gaussian_routes'],"same_baseline":v['policy_exactly_baseline_probabilities'],"policy_ba":v['policy']['balanced_accuracy']} for k,v in stress.items()}}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("selftest","pilot","run","robustness","small-group-policy"))
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        {"selftest":selftest,"pilot":pilot,"run":run,"robustness":robustness,"small-group-policy":small_group_policy}[args.command]()


if __name__=="__main__":
    main()
