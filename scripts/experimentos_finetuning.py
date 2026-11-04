"""Fine-tuning local com seleção interna e avaliação de prompts reservados separadas.

Lê runs/validacao_pesada/split.json. `inner` nunca avalia o holdout; `final`
treina nos prompts de desenvolvimento e avalia o holdout apenas ao terminar.
Os dados já foram usados na CV histórica, portanto não são um teste externo novo.
"""

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import balanced_accuracy_score, log_loss
from torch import nn
from transformers import AutoModel, AutoTokenizer

RAIZ = Path(__file__).resolve().parents[1]


class Classificador(nn.Module):
    def __init__(self, modelo, n_classes, congelar=0, pooling="mean_cls"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            str(modelo), local_files_only=True, attn_implementation="sdpa",
        )
        self.pooling = pooling
        dimensao = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(dimensao * (2 if pooling == "mean_cls" else 1), n_classes)
        if congelar:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for camada in self.encoder.encoder.layer[:congelar]:
                for p in camada.parameters():
                    p.requires_grad = False

    def forward(self, **batch):
        h = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        media = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        x = torch.cat([media, h[:, 0]], dim=1) if self.pooling == "mean_cls" else media
        return self.head(self.dropout(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nome", required=True)
    parser.add_argument("--modelo", default="roberta-base")
    parser.add_argument("--stage", choices=["inner", "final"], default="inner")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--congelar", type=int, default=0)
    parser.add_argument("--pooling", choices=["mean", "mean_cls"], default="mean_cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    saida = RAIZ / "runs/experimentos_finetuning" / args.nome
    saida.mkdir(parents=True, exist_ok=True)
    protocolo = json.loads((RAIZ / "runs/validacao_pesada/split.json").read_text())
    dados = RAIZ / "data/train.csv"
    assert hashlib.sha256(dados.read_bytes()).hexdigest() == protocolo["arquivo_sha256"]
    df = pd.read_csv(dados, keep_default_na=False)
    classes = np.asarray(protocolo["classes"])
    y = np.searchsorted(classes, df["model"].to_numpy())
    if args.stage == "inner":
        fold = protocolo["inner_folds"][args.fold]
        treino = np.asarray(fold["treino_indices"])
        valid = np.asarray(fold["valid_indices"])
    else:
        treino = np.asarray(protocolo["dev_indices"])
        valid = np.asarray(protocolo["holdout_indices"])
    assert not set(df.iloc[treino].prompt_id) & set(df.iloc[valid].prompt_id)
    modelo = RAIZ / "data/modelos" / args.modelo
    tok = AutoTokenizer.from_pretrained(str(modelo), local_files_only=True)
    tokens = tok(df["text"].tolist(), truncation=True, max_length=args.max_len)
    net = Classificador(modelo, len(classes), args.congelar, args.pooling).to(dispositivo)
    encoder = [p for p in net.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": encoder, "lr": args.lr},
        {"params": net.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    passos = math.ceil(math.ceil(len(treino) / args.batch) / args.accum) * args.epochs
    warmup = max(1, int(passos * 0.1))

    def fator(passo):
        if passo < warmup:
            return (passo + 1) / warmup
        return max(0.0, (passos - passo) / max(1, passos - warmup))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, fator)
    rng = np.random.default_rng(args.seed)

    def lote(indices):
        exemplos = [{k: v[int(i)] for k, v in tokens.items()} for i in indices]
        # Poucas formas evitam milhares de grafos/cache de tamanhos diferentes no MPS.
        return {k: v.to(dispositivo) for k, v in tok.pad(exemplos, pad_to_multiple_of=64, return_tensors="pt").items()}

    @torch.no_grad()
    def avaliar(indices):
        net.eval()
        ordem = np.argsort([len(tokens["input_ids"][int(i)]) for i in indices])
        p = np.empty((len(indices), len(classes)), dtype=np.float32)
        for inicio in range(0, len(indices), args.batch * 2):
            lugar = ordem[inicio:inicio + args.batch * 2]
            with torch.autocast(dispositivo, dtype=torch.bfloat16, enabled=dispositivo == "mps"):
                logits = net(**lote(indices[lugar]))
            p[lugar] = logits.float().softmax(-1).cpu().numpy()
        return p

    inicio = time.monotonic()
    historico = []
    melhor, sem_ganho = -1, 0
    iteracoes = 0
    print(json.dumps({"args": vars(args), "device": dispositivo, "treino": len(treino), "valid": len(valid),
                      "parametros_treinaveis": sum(p.numel() for p in net.parameters() if p.requires_grad)}), flush=True)
    for epoca in range(1, args.epochs + 1):
        net.train()
        ordem = rng.permutation(treino)
        # Lotes de comprimentos parecidos reduzem padding; blocos e linhas são embaralhados.
        blocos = [np.asarray(sorted(ordem[i:i + 128], key=lambda j: len(tokens["input_ids"][int(j)])))
                  for i in range(0, len(ordem), 128)]
        lotes = [b[i:i + args.batch] for b in blocos for i in range(0, len(b), args.batch)]
        rng.shuffle(lotes)
        opt.zero_grad(set_to_none=True)
        perdas = []
        t_epoca = time.monotonic()
        for passo, indices in enumerate(lotes):
            with torch.autocast(dispositivo, dtype=torch.bfloat16, enabled=dispositivo == "mps"):
                logits = net(**lote(indices))
                perda = nn.functional.cross_entropy(logits.float(), torch.tensor(y[indices], device=dispositivo))
            assert torch.isfinite(perda), "perda não finita"
            (perda / args.accum).backward()
            perdas.append(float(perda.detach().cpu()))
            if (passo + 1) % args.accum == 0 or passo + 1 == len(lotes):
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
            iteracoes += 1
            if iteracoes == 1 or (passo + 1) % 100 == 0:
                print(json.dumps({"epoca": epoca, "batch": passo + 1, "total": len(lotes),
                                  "loss": float(np.mean(perdas)), "segundos": time.monotonic() - t_epoca,
                                  "mps_alocado_gb": torch.mps.current_allocated_memory() / 1e9 if dispositivo == "mps" else 0,
                                  "mps_driver_gb": torch.mps.driver_allocated_memory() / 1e9 if dispositivo == "mps" else 0}), flush=True)
            if args.smoke_steps and iteracoes >= args.smoke_steps:
                p = avaliar(valid[:8])
                assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)
                print("SMOKE_OK", json.dumps({"segundos": time.monotonic() - inicio, "shape": list(p.shape)}), flush=True)
                return
        registro = {"epoca": epoca, "loss_treino": float(np.mean(perdas)), "segundos": time.monotonic() - inicio}
        if args.stage == "inner":
            p = avaliar(valid)
            registro |= {"balanced_accuracy": balanced_accuracy_score(y[valid], p.argmax(1)), "log_loss": log_loss(y[valid], p)}
            if registro["balanced_accuracy"] > melhor:
                melhor, sem_ganho = registro["balanced_accuracy"], 0
                save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(saida / "best.safetensors"))
                np.savez_compressed(saida / "valid.npz", indices=valid, classes=classes, probas=p)
                (saida / "best.json").write_text(json.dumps({"epoca": epoca, **registro, "args": vars(args)}, indent=2) + "\n")
            else:
                sem_ganho += 1
        historico.append(registro)
        (saida / "historico.json").write_text(json.dumps({"args": vars(args), "historico": historico}, indent=2) + "\n")
        print(json.dumps(registro), flush=True)
        if args.stage == "inner" and sem_ganho >= args.patience:
            break
    if args.stage == "final":
        p = avaliar(valid)
        res = {"args": vars(args), "balanced_accuracy": balanced_accuracy_score(y[valid], p.argmax(1)),
               "log_loss": log_loss(y[valid], p), "segundos": time.monotonic() - inicio,
               "limite": protocolo["limite"]}
        np.savez_compressed(saida / "holdout.npz", indices=valid, classes=classes, probas=p)
        save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(saida / "best.safetensors"))
        (saida / "resultado.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res), flush=True)
    del net, opt
    gc.collect()
    if dispositivo == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
