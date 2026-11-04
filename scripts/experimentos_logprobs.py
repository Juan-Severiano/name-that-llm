"""Sinais locais de previsibilidade do Qwen, sem salvar logits e sem treino do LLM.

  .venv/bin/python scripts/experimentos_logprobs.py selftest
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/experimentos_logprobs.py extract --device mps
  .venv/bin/python scripts/experimentos_logprobs.py evaluate

Extração lê somente id/text, guarda hashes/ordem e retoma por linha. O backbone é
executado uma vez por microbatch; o head calcula blocos pequenos de posições.
Os pesos ficam BF16 e normalização/entropia/rank em FP32. Nenhum tensor de logits
de uma sequência inteira é materializado ou salvo. Avaliação usa o protocolo
GSS 20%, semente 20261103, e GKF3 somente nos prompts de desenvolvimento.
"""

import os

for variavel in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variavel] = "1"

import argparse
import hashlib
import inspect
import json
import time
import unicodedata
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/experimentos_logprobs"
SEED = 20261103
METRICS = ("surprisal", "entropy", "logrank", "top1_surprisal", "gap", "top5_mass", "varentropy")
VERSION = 1


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_strings(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(ZoneInfo("America/Fortaleza")).isoformat(timespec="seconds")


def log(message):
    print(f"{now()} {message}", flush=True)


@torch.inference_mode()
def token_statistics(logits, targets):
    """[posições,vocabulário] -> apenas oito números por token, FP32.

    Rank usa 1 + número de logits estritamente maiores (empates recebem mesmo
    rank). Entropia e varentropia são em nats. Esta função recebe só um bloco.
    """
    z = logits.float()
    true = z.gather(1, targets[:, None]).squeeze(1)
    lz = torch.logsumexp(z, dim=1)
    top = torch.topk(z, k=min(5, z.shape[1]), dim=1).values
    prob = torch.exp(z - lz[:, None])
    ez = (prob * z).sum(dim=1)
    entropy = lz - ez
    variance = ((prob * z.square()).sum(dim=1) - ez.square()).clamp_min(0)
    rank = (z > true[:, None]).sum(dim=1).float() + 1
    result = torch.stack((lz - true, entropy, torch.log(rank), lz - top[:, 0], top[:, 0] - true,
                          torch.exp(top - lz[:, None]).sum(dim=1), variance, rank), dim=1)
    return result.cpu().numpy().astype(np.float64)


@torch.inference_mode()
def score_hidden(model, hidden, targets, chunk_positions):
    """Não materializa [sequência,vocabulário]; descarrega cada bloco no CPU."""
    parts = []
    for start in range(0, len(targets), chunk_positions):
        end = min(start + chunk_positions, len(targets))
        logits = model.get_output_embeddings()(hidden[start:end])
        parts.append(token_statistics(logits, targets[start:end]))
        del logits
    return np.concatenate(parts) if parts else np.empty((0, 8), dtype=np.float64)


def token_properties(ids, tokenizer, cache):
    values = []
    for token in ids:
        token = int(token)
        if token not in cache:
            s = tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            stripped = s.strip()
            cache[token] = (
                bool(stripped) and all(unicodedata.category(c)[0] in "PS" for c in stripped),
                bool(stripped) and stripped.isalpha(), bool(stripped) and stripped.isdigit(),
                any(c.isspace() for c in s), any(ord(c) > 127 for c in s), "\ufffd" in s, len(s),
            )
        values.append(cache[token])
    return np.asarray(values, dtype=np.float64).reshape(-1, 7)


def summarize(stats, ids, properties, text, original_tokens, retained_tokens, max_length, vocab_size):
    """109 atributos fixos por texto; ausência de tokens é representada por zeros."""
    n = len(stats)
    effective = stats if n else np.zeros((1, 8), dtype=np.float64)
    out = {}
    for j, metric in enumerate(METRICS):
        a = effective[:, j]
        out[f"{metric}_mean"], out[f"{metric}_std"] = float(a.mean()), float(a.std())
        for q in (.1, .25, .5, .75, .9):
            out[f"{metric}_q{int(100*q):02}"] = float(np.quantile(a, q))
    for j, metric in enumerate(METRICS[:3]):
        a = effective[:, j]
        for quartile, idx in enumerate(np.array_split(np.arange(len(a)), 4)):
            out[f"{metric}_position_q{quartile+1}"] = float(a[idx].mean()) if len(idx) else 0.
        x = np.linspace(0, 1, len(a))
        out[f"{metric}_position_slope"] = float(np.polyfit(x, a, 1)[0]) if len(a) > 1 else 0.
        out[f"{metric}_adjacent_delta"] = float(np.abs(np.diff(a)).mean()) if len(a) > 1 else 0.
        out[f"{metric}_lag1"] = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if len(a) > 2 and np.std(a[:-1]) > 1e-10 and np.std(a[1:]) > 1e-10 else 0.
    rank = stats[:, 7] if n else np.array([np.inf])
    for k in (1, 5, 10, 100, 1000):
        out[f"rank_le_{k}"] = float(np.mean(rank <= k))
    out["rank_gt_1000"] = float(np.mean(rank > 1000)) if n else 0.
    centered = effective[:, 0] - effective[:, 1]
    out["surprisal_minus_entropy_mean"] = float(centered.mean())
    out["surprisal_minus_entropy_std"] = float(centered.std())
    out["surprisal_gt_entropy_fraction"] = float(np.mean(centered > 0)) if n else 0.
    out["surprisal_entropy_correlation"] = float(np.corrcoef(effective[:, 0], effective[:, 1])[0, 1]) if len(effective) > 1 and np.std(effective[:, 0]) > 1e-10 and np.std(effective[:, 1]) > 1e-10 else 0.
    props = properties if n else np.zeros((1, 7))
    for j, kind in enumerate(("punctuation", "alphabetic", "digit", "whitespace", "unicode", "replacement")):
        out[f"token_{kind}_fraction"] = float(props[:, j].mean())
    out["token_characters_mean"] = float(props[:, 6].mean())
    out["token_characters_std"] = float(props[:, 6].std())
    relative_ids = np.asarray(ids, dtype=float) / max(vocab_size, 1)
    out["token_id_relative_mean"] = float(relative_ids.mean()) if n else 0.
    out["token_id_relative_std"] = float(relative_ids.std()) if n else 0.
    for j, kind in enumerate(("punctuation", "alphabetic", "digit")):
        mask = properties[:, j].astype(bool) if n else np.array([], dtype=bool)
        for k, metric in enumerate(METRICS[:2]):
            out[f"{metric}_on_{kind}"] = float(stats[mask, k].mean()) if mask.any() else 0.
    out["scored_tokens"] = n
    out["original_tokens"] = original_tokens
    out["retained_tokens"] = retained_tokens
    out["retained_fraction"] = retained_tokens / max(original_tokens, 1)
    out["truncated_fraction"] = max(0, original_tokens - retained_tokens) / max(original_tokens, 1)
    out["at_length_limit"] = float(retained_tokens >= max_length)
    out["unscored_first_fraction"] = min(retained_tokens, 1) / max(retained_tokens, 1)
    raw = text.encode()
    out["zlib_byte_ratio"] = len(zlib.compress(raw)) / max(len(raw), 1)
    out["distinct_token_fraction"] = len(set(ids)) / max(n, 1)
    out["distinct_bigram_fraction"] = len(set(zip(ids[:-1], ids[1:]))) / max(n - 1, 1)
    hits = rank == 1
    best = current = 0
    for hit in hits:
        current = current + 1 if hit else 0
        best = max(best, current)
    out["longest_top1_run_fraction"] = best / max(n, 1)
    out["adjacent_repeated_token_fraction"] = float(np.mean(np.asarray(ids[1:]) == np.asarray(ids[:-1]))) if n > 1 else 0.
    out["top1_hit_change_fraction"] = float(np.mean(hits[1:] != hits[:-1])) if n > 1 else 0.
    values = np.asarray(list(out.values()), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("atributo não finito; não marcar linha como concluída")
    return out


def feature_names():
    return list(summarize(np.empty((0, 8)), [], np.empty((0, 7)), "", 0, 0, 384, 31))


def checkpoint_identity(args, df, tokenizer):
    model_dir = Path(args.model).resolve()
    files = [p for p in model_dir.iterdir() if p.suffix in (".json", ".safetensors", ".model")]
    checkpoint = {p.name: {"bytes": p.stat().st_size, "sha256": digest_file(p)} for p in sorted(files)}
    protocol_path = ROOT / "runs/validacao_pesada/split.json"
    # Somente a lista canônica de classes do protocolo, nunca labels por linha.
    classes = json.loads(protocol_path.read_text()).get("classes", []) if protocol_path.exists() else []
    source = "\n".join(inspect.getsource(f) for f in (token_statistics, score_hidden, token_properties, summarize))
    return {
        "version": VERSION, "csv_sha256": digest_file(args.csv), "id_text_sha256": digest_strings(df[["id", "text"]].itertuples(index=False, name=None)),
        "row_ids_sha256": digest_strings(df.id), "rows": len(df), "class_order": classes,
        "checkpoint": str(model_dir), "checkpoint_files": checkpoint,
        "feature_code_sha256": hashlib.sha256(source.encode()).hexdigest(), "feature_names": feature_names(),
        "max_length": args.max_length, "add_special_tokens": False, "first_token_is_unscored": True,
        "dtype_weights": args.dtype, "dtype_statistics": "float32", "device": args.device,
        "microbatch": args.batch_size, "chunk_positions": args.chunk_positions,
        "tokenizer_class": tokenizer.__class__.__name__, "tokenizer_size": len(tokenizer),
        "read_columns": ["id", "text"], "holdout_labels_read": False,
        "note": "Token-ID relativo é sinal lexical/BPE, não frequência linguística garantida; rank é a medida de raridade condicional.",
    }


def extract(args):
    torch.set_num_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS indisponível")
    target = OUT / args.name
    target.mkdir(parents=True, exist_ok=True)
    # Nenhum label é deserializado na extração, inclusive os do holdout.
    df = pd.read_csv(args.csv, usecols=["id", "text"], keep_default_na=False)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    identity = checkpoint_identity(args, df, tok)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise RuntimeError("cache incompatível (dados/checkpoint/config/código); use outro --name, não sobrescrever")
        features = np.lib.format.open_memmap(target / "features.npy", mode="r+")
        done = np.lib.format.open_memmap(target / "done.npy", mode="r+")
    else:
        features = np.lib.format.open_memmap(target / "features.npy", mode="w+", dtype=np.float32, shape=(len(df), len(feature_names())))
        features[:] = np.nan
        features.flush()
        done = np.lib.format.open_memmap(target / "done.npy", mode="w+", dtype=bool, shape=(len(df),))
        done[:] = False
        done.flush()
        np.save(target / "row_ids.npy", df.id.to_numpy(dtype=str), allow_pickle=False)
        np.save(target / "text_sha256.npy", np.array([hashlib.sha256(t.encode()).hexdigest() for t in df.text]), allow_pickle=False)
        save_json(manifest_path, {"created_at": now(), "identity": identity})
    todo = np.flatnonzero(~done)
    if args.limit:
        todo = todo[:args.limit]
    if not len(todo):
        log(f"cache pronto: {int(done.sum())}/{len(df)} linhas; nenhum LLM carregado")
        return
    # Comprimentos sem truncar são medidos com o mesmo tokenizador, sem labels.
    all_tokens = tok(df.text.tolist(), add_special_tokens=False, truncation=False)["input_ids"]
    lengths = np.array([len(t) for t in all_tokens])
    log(f"tokenização: max={lengths.max()}, p99={np.quantile(lengths,.99):.1f}; <= {args.max_length}: {np.mean(lengths<=args.max_length):.2%}; alvos={len(todo)}")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=dtype, attn_implementation="sdpa").eval().to(args.device)
    model.config.use_cache = False
    order = todo[np.argsort(lengths[todo], kind="stable")]
    cache = {}
    started = time.monotonic()
    completed = 0
    for start in range(0, len(order), args.batch_size):
        indices = order[start:start + args.batch_size]
        sequences = [all_tokens[i][:args.max_length] for i in indices]
        # Texto vazio continua válido: sem scores, somente atributos de tamanho/compressão.
        nonempty = [j for j, seq in enumerate(sequences) if len(seq) > 0]
        stats_rows = [np.empty((0, 8)) for _ in indices]
        if nonempty:
            batch = tok.pad({"input_ids": [sequences[j] for j in nonempty]}, padding=True, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                hidden = model.model(**batch, use_cache=False, return_dict=True).last_hidden_state
                for b, j in enumerate(nonempty):
                    n = len(sequences[j])
                    stats_rows[j] = score_hidden(model, hidden[b, :n-1], batch["input_ids"][b, 1:n], args.chunk_positions)
            del hidden, batch
        for j, i in enumerate(indices):
            ids = sequences[j][1:]
            props = token_properties(ids, tok, cache)
            values = summarize(stats_rows[j], ids, props, df.text.iloc[i], int(lengths[i]), len(sequences[j]), args.max_length, model.config.vocab_size)
            assert list(values) == identity["feature_names"]
            features[i] = list(values.values())
        features.flush()  # Gravar atributos antes do marcador; interrupção repete somente linhas incompletas.
        done[indices] = True
        done.flush()
        completed += len(indices)
        elapsed = time.monotonic() - started
        if completed % 20 < args.batch_size or completed == len(todo):
            progress = {"updated_at": now(), "done": int(done.sum()), "rows": len(df), "session_completed": completed,
                        "session_seconds": elapsed, "estimated_remaining_seconds": (len(todo)-completed) * elapsed / max(completed,1),
                        "features": features.shape[1], "token_lengths": {"max": int(lengths.max()), "p99": float(np.quantile(lengths,.99)),
                                                                  "fraction_complete": float(np.mean(lengths <= args.max_length))},
                        "mps_allocated_bytes": torch.mps.current_allocated_memory() if args.device == "mps" else None}
            save_json(target / "progress.json", progress)
            log(f"{int(done.sum())}/{len(df)}; {elapsed:.1f}s; ETA sessão {progress['estimated_remaining_seconds']:.0f}s")
    del model
    if args.device == "mps":
        torch.mps.empty_cache()
    if done.all():
        assert np.isfinite(features).all()
        save_json(target / "complete.json", {"completed_at": now(), "rows": len(df), "features": features.shape[1],
                                             "features_sha256": digest_file(target / "features.npy"), "labels_used": False})
    log(f"extração encerrada; {int(done.sum())}/{len(df)}; forma={features.shape}")


def candidates():
    return [
        {"name": "logprobs_lr_c001", "kind": "lr", "C": .01, "qwen": False},
        {"name": "logprobs_lr_c01", "kind": "lr", "C": .1, "qwen": False},
        {"name": "logprobs_lr_c1", "kind": "lr", "C": 1., "qwen": False},
        {"name": "logprobs_rbf_c5", "kind": "rbf", "C": 5., "qwen": False},
        # Cada bloco foi normalizado para norma 1; C maior compensa essa escala.
        {"name": "logprobs_qwen_lr_c10", "kind": "lr", "C": 10., "qwen": True},
    ]


def fit_predict(config, X, Q, y, train, valid):
    style = StandardScaler().fit(X[train])
    a, b = style.transform(X[train]), style.transform(X[valid])
    if config["qwen"]:
        qscale = StandardScaler().fit(Q[train])
        a = np.concatenate((normalize(a), normalize(qscale.transform(Q[train]))), axis=1)
        b = np.concatenate((normalize(b), normalize(qscale.transform(Q[valid]))), axis=1)
    if config["kind"] == "lr":
        model = LogisticRegression(C=config["C"], max_iter=2000)
    else:
        model = SVC(C=config["C"], kernel="rbf", gamma="scale", probability=True, random_state=SEED)
    model.fit(a, y[train])
    return model.predict_proba(b), model.classes_


def evaluate(args):
    destination = OUT / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "holdout.json").exists():
        log("holdout já avaliado e congelado; não reabrir seleção")
        return
    source = OUT / args.name
    manifest = json.loads((source / "manifest.json").read_text())["identity"]
    if not np.load(source / "done.npy", allow_pickle=False).all():
        raise RuntimeError("extração incompleta")
    meta = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "prompt_id"], keep_default_na=False)
    if digest_file(ROOT / "data/train.csv") != manifest["csv_sha256"]:
        raise RuntimeError("dados mudaram depois da extração")
    np.testing.assert_array_equal(meta.id.to_numpy(dtype=str), np.load(source / "row_ids.npy", allow_pickle=False))
    dev, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(meta, groups=meta.prompt_id))
    protocol = json.loads((ROOT / "runs/validacao_pesada/split.json").read_text())
    np.testing.assert_array_equal(dev, protocol["dev_indices"])
    np.testing.assert_array_equal(holdout, protocol["holdout_indices"])
    assert set(meta.prompt_id.iloc[dev]).isdisjoint(meta.prompt_id.iloc[holdout])
    classes = np.array(protocol["classes"], dtype=str)
    # Treino e seleção usam somente os labels de desenvolvimento.
    labels_dev = pd.read_csv(ROOT / "data/train.csv", usecols=["model"], keep_default_na=False).iloc[dev].model.to_numpy()
    y = np.full(len(meta), "", dtype=object)
    y[dev] = labels_dev
    X = np.load(source / "features.npy", allow_pickle=False)
    assert np.isfinite(X).all()
    Q = np.load(ROOT / "data/artefatos/emb_Qwen3-0.6B.npy", allow_pickle=False).astype(np.float32)
    inner = [(dev[t], dev[v]) for t, v in GroupKFold(n_splits=3).split(meta.iloc[dev], labels_dev, meta.prompt_id.iloc[dev])]
    setup = {"created_at": now(), "seed": SEED, "dev_rows": len(dev), "holdout_rows": len(holdout),
             "dev_prompts": int(meta.prompt_id.iloc[dev].nunique()), "holdout_prompts": int(meta.prompt_id.iloc[holdout].nunique()),
             "selection": "GroupKFold3 shuffle=False only development; predeclared five configurations",
             "holdout_limit": "Prompts usados na validação histórica; diagnóstico desta campanha, não teste totalmente virgem.",
             "candidates": candidates(), "feature_manifest": manifest, "class_order": classes.tolist()}
    save_json(destination / "protocol.json", setup)
    np.savez_compressed(destination / "split.npz", dev=dev, holdout=holdout, classes=classes)
    ranking = []
    with threadpool_limits(limits=1):
        for config in candidates():
            t0 = time.monotonic()
            p = np.empty((len(dev), len(classes)))
            lookup = {int(v): i for i, v in enumerate(dev)}
            scores = []
            for k, (t, v) in enumerate(inner):
                pv, order = fit_predict(config, X, Q, y, t, v)
                np.testing.assert_array_equal(order, classes)
                p[[lookup[int(vv)] for vv in v]] = pv
                scores.append(float(balanced_accuracy_score(y[v], classes[pv.argmax(axis=1)])))
                log(f"{config['name']} fold{k+1}: {scores[-1]:.4f}")
            score = float(balanced_accuracy_score(labels_dev, classes[p.argmax(axis=1)]))
            ranking.append({"config": config, "balanced_accuracy": score, "folds": scores, "seconds": time.monotonic()-t0})
            np.save(destination / f"oof_{config['name']}.npy", p, allow_pickle=False)
            save_json(destination / "development.json", sorted(ranking, key=lambda r: -r["balanced_accuracy"]))
        winner = max(ranking, key=lambda r: r["balanced_accuracy"])
        save_json(destination / "selection_frozen.json", {"frozen_at": now(), "winner": winner, "holdout_read": False})
        # Somente a configuração selecionada vê o holdout, uma vez.
        y[holdout] = pd.read_csv(ROOT / "data/train.csv", usecols=["model"], keep_default_na=False).iloc[holdout].model.to_numpy()
        p, order = fit_predict(winner["config"], X, Q, y, dev, holdout)
    np.testing.assert_array_equal(order, classes)
    predicted = classes[p.argmax(axis=1)]
    rng = np.random.default_rng(SEED)
    groups = meta.prompt_id.iloc[holdout].to_numpy()
    unique = np.unique(groups)
    boot = []
    for _ in range(1500):
        idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique), replace=True)])
        boot.append(balanced_accuracy_score(y[holdout][idx], predicted[idx]))
    result = {"completed_at": now(), "winner": winner["config"], "development_balanced_accuracy": winner["balanced_accuracy"],
              "holdout_balanced_accuracy": float(balanced_accuracy_score(y[holdout], predicted)),
              "holdout_log_loss": float(log_loss(y[holdout], p, labels=classes)), "prompt_bootstrap95": np.quantile(boot, [.025,.975]).tolist(),
              "holdout_rows": len(holdout), "holdout_prompts": len(unique), "features": X.shape[1]}
    np.savez_compressed(destination / "holdout_predictions.npz", indices=holdout, classes=classes, row_ids=meta.id.iloc[holdout].to_numpy(dtype=str), probas=p)
    save_json(destination / "holdout.json", result)
    log(json.dumps(result, ensure_ascii=False))


def selftest():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    logits = torch.tensor([[1., 2., -1., 0., .5, -2.], [-.5, .3, .1, -.3, .9, -.9]])
    targets = torch.tensor([2, 4])
    actual = token_statistics(logits, targets)
    z = logits.numpy().astype(float)
    probabilities = np.exp(z - np.max(z, axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    surprisal = -np.log(probabilities[np.arange(2), targets.numpy()])
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    variance = np.sum(probabilities * np.log(probabilities) ** 2, axis=1) - entropy ** 2
    np.testing.assert_allclose(actual[:, 0], surprisal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[:, 1], entropy, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[:, 6], variance, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual[:, 7], [5., 1.])
    from transformers import Qwen3Config
    config = Qwen3Config(vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         max_position_embeddings=32, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.tensor([[1, 3, 4, 2, 0], [1, 7, 8, 9, 10]])
    mask = ids != 0
    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        for i, length in enumerate([4,5]):
            reference = token_statistics(full[i,:length-1], ids[i,1:length])
            for chunk in (1,2,8):
                obtained = score_hidden(model, hidden[i,:length-1], ids[i,1:length], chunk)
                np.testing.assert_allclose(obtained, reference, rtol=1e-5, atol=1e-6)
    names = feature_names()
    empty = summarize(np.empty((0,8)), [], np.empty((0,7)), "", 0, 0, 384, 31)
    short = summarize(actual, [2,4], np.ones((2,7)), "a!", 3, 3, 384, 6)
    assert list(empty) == list(short) == names
    assert len(names) == 109
    assert np.isfinite(list(empty.values())).all() and np.isfinite(list(short.values())).all()
    result = {"completed_at": now(), "device": "cpu", "tiny_model": True, "features": len(names),
              "manual_probability_entropy_variance": True, "causal_shift": True, "padding": True,
              "chunk_equals_full_head": [1,2,8], "empty_inputs": True}
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "selftest.json", result)
    log(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("extract")
    p.add_argument("--csv", type=Path, default=ROOT / "data/train.csv")
    p.add_argument("--model", type=Path, default=ROOT / "data/modelos/Qwen3-0.6B")
    p.add_argument("--name", default="train")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--chunk-positions", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="somente teste de tempo; retoma as linhas restantes depois")
    p = sub.add_parser("evaluate")
    p.add_argument("--name", default="train")
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "extract":
        if args.max_length < 2 or args.batch_size < 1 or args.chunk_positions < 1:
            parser.error("max-length>=2; batch-size>=1; chunk-positions>=1")
        extract(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
