"""Adaptação local LoRA20 Qwen3-0.6B, somente validação interna por prompt.

Receita adaptada do notebook público kirilltym/0-600-teach-the-model-remove-the-prompt.
Nenhum checkpoint de terceiros treinado nos rótulos oficiais é carregado.
O holdout não é avaliado por este script.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, log_loss
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261103
LAYERS = [2, 4, 6, 8, 10, 12, 16, 20]
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}


def log(value):
    print(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value, flush=True)


def json_save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def public_subset(out, classes, target=2000):
    """Whole public prompt groups, at most one text per model in a group."""
    source = ROOT / "data/open_pool/pool_conservador.parquet"
    manifest_path, selected_path = out / "pool_selection.json", out / "selected_pool.parquet"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if selected_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_sha256"] == source_sha
        frame = pd.read_parquet(selected_path)
        assert hashlib.sha256("\n".join(frame["id"].astype(str)).encode()).hexdigest() == manifest["selected_ids_sha256"]
        return frame, manifest
    pool = pd.read_parquet(source)
    pool = pool.loc[pool["model"].isin(classes)].copy()
    rng = np.random.default_rng(SEED)
    pool = pool.iloc[rng.permutation(len(pool))].drop_duplicates(["prompt_canon", "model"], keep="first")
    eligible = pool.groupby("prompt_canon")["model"].nunique()
    groups = np.sort(eligible[eligible >= 3].index.to_numpy())
    groups = rng.permutation(groups)
    grouped = {prompt: group for prompt, group in pool.groupby("prompt_canon", sort=False)}
    selected, seen, size = [], set(), 0
    for prompt in groups:
        group = grouped[prompt]
        if size >= target and len(seen) == len(classes):
            break
        if size >= target and not (set(group["model"]) - seen):
            continue
        if size + len(group) > 3072:
            continue
        selected.append(group)
        seen.update(group["model"])
        size += len(group)
    frame = pd.concat(selected).reset_index(drop=True)
    frame["prompt_id"] = frame["prompt_canon"].astype(str)
    assert 1800 <= len(frame) <= 3072 and len(seen) == len(classes)
    assert frame.groupby("prompt_id")["model"].nunique().min() >= 3
    assert not frame.duplicated(["prompt_id", "model"]).any()
    manifest = {"source": str(source), "source_sha256": source_sha, "seed": SEED,
        "selection": "shuffle public rows; deduplicate author per prompt; shuffle eligible whole groups; stop at target and all18classes",
        "target_rows": target, "selected_rows": len(frame), "selected_groups": frame["prompt_id"].nunique(),
        "classes": classes.tolist(), "class_counts": {str(k): int(v) for k, v in frame.model.value_counts().items()},
        "selected_ids": frame["id"].astype(str).tolist(),
        "selected_ids_sha256": hashlib.sha256("\n".join(frame["id"].astype(str)).encode()).hexdigest(),
        "source_dataset_counts": {str(k): int(v) for k, v in frame.source_dataset.value_counts().items()},
        "source_license_counts": {str(k): int(v) for k, v in frame.source_license.value_counts().items()},
        "prompt_evidence_counts": {str(k): int(v) for k, v in frame.prompt_evidence.value_counts().items()},
        "official_train_or_validation_used_for_stage1": False}
    frame.to_parquet(selected_path, index=False)
    json_save(manifest_path, manifest)
    return frame, manifest


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        delta = F.linear(F.linear(self.dropout(x.float()), self.lora_A), self.lora_B)
        return result + (delta * self.scale).to(result.dtype)


class LoRAClassifier(nn.Module):
    def __init__(self, model_path, n_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(str(model_path), local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa")
        self.backbone.layers = nn.ModuleList(list(self.backbone.layers[:20]))
        self.backbone.config.num_hidden_layers = 20
        if hasattr(self.backbone.config, "layer_types"):
            self.backbone.config.layer_types = self.backbone.config.layer_types[:20]
        self.backbone.config.use_cache = False
        self.backbone.norm = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        count = 0
        for module in list(self.backbone.modules()):
            for name, child in list(module.named_children()):
                if name in TARGETS and isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child))
                    count += 1
        assert count == 20 * len(TARGETS), count
        hidden = self.backbone.config.hidden_size
        self.norms = nn.ModuleList([nn.LayerNorm(hidden, dtype=torch.float32) for _ in LAYERS])
        self.plain = nn.Linear(hidden * len(LAYERS), n_classes, dtype=torch.float32)
        self.centered = nn.Linear(hidden * len(LAYERS), n_classes, dtype=torch.float32)
        self.injected_linear_count = count

    def features(self, batch):
        device = batch["input_ids"].device.type
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "mps"):
            output = self.backbone(**batch, use_cache=False, output_hidden_states=True)
        mask = batch["attention_mask"].clone().float()
        mask[:, 0] = 0  # attention sink excluded, as in source recipe
        mask = mask.unsqueeze(-1)
        with torch.autocast(device, enabled=False):
            features = [norm((output.hidden_states[layer].float() * mask).sum(1) / mask.sum(1).clamp_min(1))
                        for layer, norm in zip(LAYERS, self.norms)]
            return torch.cat(features, dim=1)

    def training_logits(self, batch):
        features = self.features(batch)
        assert len(features) % 3 == 0
        triplets = features.reshape(-1, 3, features.shape[-1])
        centered = (triplets - triplets.mean(1, keepdim=True)).reshape_as(features)
        return self.plain(features), self.centered(centered)


def save_trainable(net, directory):
    directory.mkdir(parents=True, exist_ok=True)
    adapters, heads = {}, {}
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            target = adapters if "lora_" in name else heads
            target[name] = parameter.detach().cpu().contiguous()
    save_file(adapters, str(directory / "adapter.safetensors"))
    save_file(heads, str(directory / "heads.safetensors"))


def load_trainable(net, directory):
    state = load_file(str(directory / "adapter.safetensors")) | load_file(str(directory / "heads.safetensors"))
    parameters = dict(net.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def triplet_batches(frame, train, lengths, triplets_per_batch, rng):
    buckets = {}
    for _, group in frame.iloc[train].groupby("prompt_id", sort=True):
        indices = rng.permutation(group.index.to_numpy())
        for start in range(0, len(indices) - 2, 3):
            triple = indices[start:start + 3]
            assert frame["model"].iloc[triple].nunique() == 3
            size = math.ceil(max(lengths[int(i)] for i in triple) / 64) * 64
            buckets.setdefault(size, []).append(triple)
    batches = []
    for triples in buckets.values():
        rng.shuffle(triples)
        for start in range(0, len(triples), triplets_per_batch):
            batches.append(np.concatenate(triples[start:start + triplets_per_batch]))
    rng.shuffle(batches)
    return batches


def inferred_groups(texts):
    x = TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float32).fit_transform(texts)
    distance = np.clip(1 - (x @ x.T).toarray(), 0, 1)
    distance /= max(float(np.median(distance[np.triu_indices(len(texts), 1)])), 1e-6)
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance, checks=False), method="average"), 0.85, criterion="distance")


def centered_probas(net, features, groups, plain):
    centered = features.copy()
    sizes = np.zeros(len(features), dtype=np.int64)
    for group in np.unique(groups):
        take = groups == group
        sizes[take] = take.sum()
        centered[take] -= centered[take].mean(axis=0)
    weight = net.centered.weight.detach().float().cpu().numpy()
    bias = net.centered.bias.detach().float().cpu().numpy()
    p = softmax(centered @ weight.T + bias, axis=1)
    blend = plain.copy()
    eligible = (sizes >= 2) & (sizes <= 18)
    blend[eligible] = softmax(np.log(np.clip(plain[eligible], 1e-9, 1)) + 0.15 * np.log(np.clip(p[eligible], 1e-9, 1)), axis=1)
    return p, blend, {"eligible_rows": int(eligible.sum()), "groups": int(len(np.unique(groups)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="official_fold0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--triplets", type=int, default=2)
    parser.add_argument("--accum", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.65)
    parser.add_argument("--stage", choices=["official", "pool"], default="official")
    parser.add_argument("--pool-target-rows", type=int, default=2000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.prepare_only and args.stage != "pool":
        parser.error("--prepare-only requires --stage pool; no GPU work was started")
    assert args.triplets in [1, 2, 3]
    torch.set_num_threads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not args.prepare_only:
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    out = ROOT / "runs" / "experimentos_lora" / args.name
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "resultados.json").exists(), "Run already completed; use a new explicit name"
    common = json.loads((ROOT / "runs" / "validacao_pesada" / "split.json").read_text())
    classes = np.array(common["classes"])
    pool_manifest = None
    if args.stage == "pool":
        assert args.resume is None, "Stage1 must start from the original fresh pretrained model"
        frame, pool_manifest = public_subset(out, classes, args.pool_target_rows)
        train, valid = np.arange(len(frame)), np.array([], dtype=np.int64)
        if args.prepare_only:
            log({"prepared_pool": {k: pool_manifest[k] for k in ["selected_rows", "selected_groups", "class_counts", "selected_ids_sha256"]}})
            return
    else:
        source = ROOT / "data" / "train.csv"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == common["arquivo_sha256"]
        frame = pd.read_csv(source, keep_default_na=False)
        fold = common["inner_folds"][args.fold]
        train, valid = np.array(fold["treino_indices"]), np.array(fold["valid_indices"])
        assert not set(frame["prompt_id"].iloc[train]) & set(frame["prompt_id"].iloc[valid])
        assert not set(train) & set(common["holdout_indices"])
        assert not set(valid) & set(common["holdout_indices"])
    labels = np.searchsorted(classes, frame["model"].to_numpy())
    model_path = ROOT / "data" / "modelos" / "Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Only inner train/validation texts are tokenized; holdout rows are never forwarded.
    relevant = np.concatenate([train, valid])
    encoded = tokenizer(frame["text"].iloc[relevant].tolist(), add_special_tokens=False,
                        truncation=True, max_length=args.max_len - 1)["input_ids"]
    ids = {int(i): row + [tokenizer.eos_token_id] for i, row in zip(relevant, encoded)}
    lengths = {i: len(row) for i, row in ids.items()}
    net = LoRAClassifier(model_path, len(classes)).to(device)
    if args.resume:
        resume_protocol = json.loads((args.resume.parent / "protocolo.json").read_text())
        np.testing.assert_array_equal(resume_protocol["classes"], classes)
        load_trainable(net, args.resume)
    lora = [p for n, p in net.named_parameters() if p.requires_grad and "lora_" in n]
    heads = [p for n, p in net.named_parameters() if p.requires_grad and "lora_" not in n]
    assert all(p.dtype == torch.float32 for p in lora + heads)
    assert all(p.dtype == torch.bfloat16 for p in net.backbone.parameters() if not p.requires_grad)
    optimizer = torch.optim.AdamW([{"params": lora, "lr": args.lr}, {"params": heads, "lr": args.head_lr}],
                                  weight_decay=0.01, foreach=False)
    rng = np.random.default_rng(SEED)
    first_batches = triplet_batches(frame, train, lengths, args.triplets, np.random.default_rng(SEED))
    estimated_updates = math.ceil(len(first_batches) / args.accum) * args.epochs
    warmup = max(1, int(0.1 * estimated_updates))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, estimated_updates - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    valid_groups = inferred_groups(frame["text"].iloc[valid].tolist()) if len(valid) else None
    class_weight = None
    if args.stage == "pool":
        counts = np.bincount(labels[train], minlength=len(classes))
        assert np.all(counts > 0)
        class_weight = torch.tensor(len(train) / (len(classes) * counts), dtype=torch.float32, device=device)
    serial_args = vars(args) | {"resume": str(args.resume) if args.resume else None}
    protocol = {"args": serial_args, "seed": SEED, "model": "local original Qwen/Qwen3-0.6B", "keep_layers": 20,
        "pool_layers": LAYERS, "rank": 8, "alpha": 16, "lora_dropout": 0.05, "targets": sorted(TARGETS),
        "base_dtype": "bfloat16 frozen", "trainable_dtype": "float32", "autocast": "bfloat16 on MPS",
        "layernorm_per_layer": True, "final_norm": "Identity", "mean_pool_excludes_first_token": True,
        "eos_appended": True, "effective_batch_target": args.triplets * 3 * args.accum,
        "train_indices": train.tolist(), "valid_indices": valid.tolist(), "classes": classes.tolist(),
        "selection": "maximum inner-fold plain balanced accuracy; oracle centered scores are auxiliary",
        "auxiliary_metrics": ["fixed .15 inferred centered twin", "standalone inferred centered with plain fallback", "true-prompt oracle centered"],
        "inferred_twin": "content-word TFIDF median-normalized distance, average-linkage threshold .85; groups2..18; logweight .15",
        "source_recipe": "data/public_notebooks/teach_remove_prompt/source.txt, public author's claims are not independently verified",
        "source_recipe_sha256": hashlib.sha256((ROOT / "data/public_notebooks/teach_remove_prompt/source.txt").read_bytes()).hexdigest(),
        "data_stage": args.stage, "class_weight": class_weight.detach().cpu().tolist() if class_weight is not None else None,
        "pool_manifest": str(out / "pool_selection.json") if pool_manifest else None,
        "adaptations": ["0.6B original pretrained model instead of source 8B/14B", "20 layers / rank8 alpha16 / smaller physical batches",
                        "Public subset rather than full35k pool when stage=pool", "No holdout scoring"],
        "injected_linears": net.injected_linear_count, "trainable_parameters": sum(p.numel() for p in lora + heads),
        "token_lengths": {"min": min(lengths.values()), "max": max(lengths.values()), "median": float(np.median(list(lengths.values())))}}
    json_save(out / "protocolo.json", protocol)
    log({"start": protocol["token_lengths"], "device": device, "train": len(train), "valid": len(valid),
         "trainable": protocol["trainable_parameters"], "batches_per_epoch": len(first_batches), "updates": estimated_updates})

    def batch(indices):
        value = tokenizer.pad({"input_ids": [ids[int(i)] for i in indices]}, pad_to_multiple_of=64, return_tensors="pt")
        return {k: v.to(device) for k, v in value.items()}

    @torch.no_grad()
    def extract(indices):
        net.eval()
        order = np.argsort([lengths[int(i)] for i in indices])
        features = np.empty((len(indices), 1024 * len(LAYERS)), dtype=np.float32)
        p = np.empty((len(indices), len(classes)), dtype=np.float32)
        for start in range(0, len(indices), args.triplets * 3 * 2):
            positions = order[start:start + args.triplets * 3 * 2]
            f = net.features(batch(indices[positions]))
            features[positions] = f.detach().float().cpu().numpy()
            p[positions] = net.plain(f).softmax(-1).cpu().numpy()
        assert np.isfinite(features).all() and np.isfinite(p).all()
        return features, p

    started, last_log = time.monotonic(), time.monotonic()
    history, best_score, best_epoch, microsteps, updates = [], -1, 0, 0, 0
    save_trainable(net, out / "initial")
    stop = {"signal": None}
    def request_stop(signum, _frame):
        stop["signal"] = signum
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for epoch in range(1, args.epochs + 1):
        net.train()
        batches = triplet_batches(frame, train, lengths, args.triplets, rng)
        if args.smoke_steps:
            by_shape = {}
            for indices in batches:
                shape = math.ceil(max(lengths[int(i)] for i in indices) / 64) * 64
                by_shape.setdefault(shape, indices)
            prefix = [by_shape[key] for key in sorted(by_shape)]
            batches = prefix + batches
            log({"smoke_padding_shapes": sorted(by_shape), "smoke_first_shapes": [math.ceil(max(lengths[int(i)] for i in x) / 64) * 64 for x in batches[:args.smoke_steps]]})
        optimizer.zero_grad(set_to_none=True)
        losses, epoch_start = [], time.monotonic()
        for position, indices in enumerate(batches):
            plain, centered = net.training_logits(batch(indices))
            target = torch.tensor(labels[indices], device=device)
            plain_loss, center_loss = F.cross_entropy(plain, target, weight=class_weight), F.cross_entropy(centered, target, weight=class_weight)
            loss = plain_loss + center_loss
            assert torch.isfinite(loss).item(), "non-finite loss"
            (loss / args.accum).backward()
            losses.append(float(loss.detach().cpu()))
            microsteps += 1
            if (position + 1) % args.accum == 0 or position + 1 == len(batches) or (args.smoke_steps and microsteps >= args.smoke_steps):
                grad_norm = nn.utils.clip_grad_norm_(lora + heads, 1.0)
                assert torch.isfinite(grad_norm).item(), "non-finite gradients"
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if args.checkpoint_every and updates % args.checkpoint_every == 0:
                    save_trainable(net, out / "last")
                    json_save(out / "last.json", {"epoch": epoch, "microbatch": position + 1, "updates": updates,
                              "mean_training_loss": float(np.mean(losses)), "elapsed_seconds": time.monotonic() - started})
            del plain, centered, plain_loss, center_loss, loss, target
            if device == "mps" and args.empty_cache_every and microsteps % args.empty_cache_every == 0:
                torch.mps.empty_cache()
            if stop["signal"] is not None:
                save_trainable(net, out / "last")
                info = {"status": "STOPPED_WITH_CHECKPOINT", "signal": stop["signal"], "epoch": epoch,
                        "microbatch": position + 1, "updates": updates, "elapsed_seconds": time.monotonic() - started,
                        "holdout_evaluated": False}
                json_save(out / "interrompido.json", info)
                log(info)
                return
            now = time.monotonic()
            if position == 0 or now - last_log >= 30:
                log({"epoch": epoch, "microbatch": position + 1, "total": len(batches), "loss": float(np.mean(losses)),
                     "elapsed_seconds": now - started, "epoch_seconds": now - epoch_start,
                     "mps_allocated_gb": torch.mps.current_allocated_memory() / 1e9 if device == "mps" else 0,
                     "mps_driver_gb": torch.mps.driver_allocated_memory() / 1e9 if device == "mps" else 0})
                last_log = now
            if args.smoke_steps and microsteps >= args.smoke_steps:
                assert all(p.grad is None for p in net.backbone.parameters() if not p.requires_grad)
                assert any(torch.count_nonzero(p.detach()).item() for n, p in net.named_parameters() if "lora_B" in n)
                smoke_rows = valid[:6] if len(valid) else train[:6]
                f, p = extract(smoke_rows)
                assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
                save_trainable(net, out / "smoke_checkpoint")
                load_trainable(net, out / "smoke_checkpoint")
                _, restored = extract(smoke_rows)
                np.testing.assert_allclose(restored, p, rtol=1e-5, atol=1e-6)
                result = {"status": "SMOKE_OK", "microsteps": microsteps, "proba_shape": list(p.shape),
                          "features_shape": list(f.shape), "seconds": time.monotonic() - started}
                json_save(out / "smoke.json", result)
                log(result)
                return
        if args.stage == "pool":
            record = {"epoch": epoch, "training_loss": float(np.mean(losses)), "elapsed_seconds": time.monotonic() - started,
                      "stage": "public_only", "official_validation_evaluated": False}
            history.append(record)
            save_trainable(net, out / "final")
            json_save(out / "historico.json", {"history": history})
            log(record)
            continue
        features, p = extract(valid)
        center_p, inferred_p, group_info = centered_probas(net, features, valid_groups, p)
        sizes = np.array([np.sum(valid_groups == g) for g in valid_groups])
        eligible = (sizes >= 2) & (sizes <= 18)
        inferred_center_only = p.copy()
        inferred_center_only[eligible] = center_p[eligible]
        true_groups = frame["prompt_id"].iloc[valid].to_numpy()
        oracle_p, oracle_blend, _ = centered_probas(net, features, true_groups, p)
        record = {"epoch": epoch, "training_loss": float(np.mean(losses)),
            "plain_balanced_accuracy": float(balanced_accuracy_score(labels[valid], p.argmax(axis=1))),
            "plain_log_loss": float(log_loss(labels[valid], p)),
            "inferred_twin_blend_accuracy": float(balanced_accuracy_score(labels[valid], inferred_p.argmax(axis=1))),
            "inferred_centered_standalone_accuracy": float(balanced_accuracy_score(labels[valid], inferred_center_only.argmax(axis=1))),
            "oracle_centered_accuracy": float(balanced_accuracy_score(labels[valid], oracle_p.argmax(axis=1))),
            "oracle_twin_blend_accuracy": float(balanced_accuracy_score(labels[valid], oracle_blend.argmax(axis=1))),
            "inferred_groups": group_info, "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        np.savez_compressed(out / f"valid_epoch{epoch}.npz", plain=p, inferred_twin=inferred_p, inferred_centered=inferred_center_only,
                            oracle_centered=oracle_p, oracle_twin=oracle_blend, indices=valid, classes=classes)
        if record["plain_balanced_accuracy"] > best_score:
            best_score, best_epoch = record["plain_balanced_accuracy"], epoch
            save_trainable(net, out / "best")
            np.savez_compressed(out / "valid_best.npz", plain=p, inferred_twin=inferred_p, inferred_centered=inferred_center_only,
                                indices=valid, classes=classes,
                                features=features.astype(np.float16))
            json_save(out / "best.json", record)
        json_save(out / "historico.json", {"history": history, "best_epoch": best_epoch, "best_plain_accuracy": best_score})
        log(record)
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if epoch == 2:
            previous = history[:-1]
            worse_loss_without_accuracy_gain = (record["plain_log_loss"] > min(x["plain_log_loss"] for x in previous)
                and record["plain_balanced_accuracy"] <= max(x["plain_balanced_accuracy"] for x in previous))
            if record["plain_balanced_accuracy"] < 0.35 or worse_loss_without_accuracy_gain:
                log({"predeclared_stop_epoch2": True, "accuracy_floor": 0.35,
                     "worse_loss_without_accuracy_gain": worse_loss_without_accuracy_gain})
                break
        if (out / "STOP_AFTER_EPOCH").exists():
            log({"stop_after_epoch": epoch, "best_plain_accuracy": best_score})
            break
    if args.stage == "pool":
        result = {"protocol": protocol, "history": history, "elapsed_seconds": time.monotonic() - started,
                  "selected_pool_manifest": pool_manifest, "official_validation_evaluated": False, "holdout_evaluated": False}
        json_save(out / "resultados.json", result)
        log({"stage1_complete": True, "elapsed_seconds": result["elapsed_seconds"], "checkpoint": str(out / "final")})
        return
    load_trainable(net, out / "best")
    anchor_features, anchor_p = extract(train)
    centered_anchors = anchor_features.copy()
    train_prompts = frame["prompt_id"].iloc[train].to_numpy()
    for prompt in np.unique(train_prompts):
        take = train_prompts == prompt
        centered_anchors[take] -= centered_anchors[take].mean(axis=0)
    centroids = np.stack([anchor_features[labels[train] == c].mean(axis=0) for c in range(len(classes))])
    centered_centroids = np.stack([centered_anchors[labels[train] == c].mean(axis=0) for c in range(len(classes))])
    np.savez_compressed(out / "anchor_features.npz", features=anchor_features.astype(np.float16), labels=labels[train],
                        indices=train, classes=classes, prompts=train_prompts.astype(str),
                        class_centroids=centroids, centered_class_centroids=centered_centroids)
    result = {"protocol": protocol, "history": history, "selected_epoch": best_epoch,
              "best_inner_plain_accuracy": best_score, "anchor_train_accuracy": float(np.mean(anchor_p.argmax(axis=1) == labels[train])),
              "elapsed_seconds": time.monotonic() - started, "holdout_evaluated": False,
              "warning": "Single inner fold selected over epochs; oracle prompt-group scores are auxiliary only."}
    json_save(out / "resultados.json", result)
    log(result)


if __name__ == "__main__":
    main()
