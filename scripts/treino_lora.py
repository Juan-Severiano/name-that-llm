"""Treino LoRA de um Qwen3 como classificador dos 18 modelos, em duas etapas: pool público e textos oficiais.

Roda no Kaggle (GPU, internet ligada só neste notebook de treino). Etapa 1 treina no pool conservador e mede nos
2.699 textos oficiais, que ela nunca vê. Etapa 2 (opcional) ajusta nos oficiais, menos 30 prompts reservados
(um a cada cinco, ordenados), e mede nesses 540. Salva o adaptador, as probabilidades e o log.
Este arquivo também roda local com um modelo minúsculo (scripts/ensaio_treino_lora.py).
"""

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CFG = {
    "modelo": "Qwen/Qwen3-1.7B",
    "max_textos_pool": 4000,  # etapa 1: quantos textos do pool (sorteados)
    "epocas_pool": 1,
    "epocas_oficial": 0,  # etapa 2; 0 desliga
    "max_len": 384,
    "micro_lote": 8,
    "acumular": 4,
    "lr_pool": 2e-4,
    "lr_oficial": 5e-5,
    "r": 16,
    "checkpointing": True,  # a T4 tem 15 GB: sem isso o lote de 8 estoura a memória
    "tempo_maximo_s": 2400,  # para de treinar e segue para avaliação e gravação
    "salvar_cada": 200,  # passos
    "avaliar_max_textos": 0,  # 0 = todos os oficiais
    "saida": "/kaggle/working",
    "entrada": "/kaggle/input",
}
CFG_JSON = "{}"  # o gerador do notebook troca esta linha
CFG.update(json.loads(CFG_JSON))

INICIO = time.time()


def log(*args):
    print(f"[{time.time() - INICIO:7.0f}s]", *args, flush=True)


def preparar_ambiente():
    """O peft do Kaggle exige torchao >= 0.16 e a imagem traz o 0.10: sem o torchao o peft simplesmente não o usa."""
    import importlib.metadata as md

    try:
        versao = md.version("torchao")
    except md.PackageNotFoundError:
        return
    if tuple(int(x) for x in versao.split(".")[:2]) < (0, 16):
        log(f"torchao {versao} incompatível com o peft: removendo")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)


def achar(sufixo: str) -> Path:
    return next(p for p in Path(CFG["entrada"]).rglob("*") if p.as_posix().endswith(sufixo))


def carregar_dados():
    oficial = pd.read_csv(achar("train.csv"), keep_default_na=False)
    pool = pd.read_parquet(achar("pool_conservador.parquet"), columns=["text", "model"])
    pool = pool[pool.text.str.split().str.len().between(30, 300)].sample(frac=1, random_state=0)
    if CFG["max_textos_pool"]:
        pool = pool.iloc[: CFG["max_textos_pool"]]
    classes = sorted(oficial.model.unique())
    assert len(classes) == 18 and set(pool.model) <= set(classes), "classes do pool não batem com as oficiais"
    return oficial, pool.reset_index(drop=True), classes


def dispositivo_e_tipo():
    if torch.cuda.is_available():
        nativo = torch.cuda.get_device_capability()[0] >= 8  # T4 e P100 só emulam bf16, devagar
        log("GPU", torch.cuda.get_device_name(), "capacidade", torch.cuda.get_device_capability(), "memória GB",
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
        return "cuda", torch.bfloat16 if nativo else torch.float16
    return ("mps" if torch.backends.mps.is_available() else "cpu"), torch.float32


def montar_modelo(tok, dispositivo, tipo):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(CFG["modelo"], num_labels=18, dtype=tipo)
    base.config.pad_token_id = tok.pad_token_id
    if CFG["checkpointing"]:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    cfg = LoraConfig(task_type="SEQ_CLS", r=CFG["r"], lora_alpha=2 * CFG["r"], lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    modelo = get_peft_model(base, cfg)
    for p in modelo.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # só o adaptador e a cabeça treinam, em fp32
    modelo.to(dispositivo)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    log(f"modelo {CFG['modelo']} em {dispositivo}/{tipo}, parâmetros treináveis {treinaveis / 1e6:.1f} M")
    return modelo


def tokenizar(tok, textos):
    return tok([t.strip() for t in textos], truncation=True, max_length=CFG["max_len"], add_special_tokens=True)["input_ids"]


def lotes(ids, tamanho, rng, embaralhar):
    ordem = rng.permutation(len(ids)) if embaralhar else np.arange(len(ids))
    blocos = [ordem[i : i + tamanho * 50] for i in range(0, len(ids), tamanho * 50)]
    saida = []
    for bloco in blocos:
        bloco = sorted(bloco, key=lambda i: len(ids[i]))  # textos de tamanho parecido juntos
        saida += [bloco[i : i + tamanho] for i in range(0, len(bloco), tamanho)]
    if embaralhar:
        rng.shuffle(saida)
    return saida


def montar_lote(ids, indices, pad, dispositivo):
    maior = max(len(ids[i]) for i in indices)
    entrada = torch.full((len(indices), maior), pad, dtype=torch.long)
    mascara = torch.zeros((len(indices), maior), dtype=torch.long)
    for k, i in enumerate(indices):
        entrada[k, : len(ids[i])] = torch.tensor(ids[i])
        mascara[k, : len(ids[i])] = 1
    return entrada.to(dispositivo), mascara.to(dispositivo)


@torch.no_grad()
def prever(modelo, ids, pad, dispositivo, tipo, tamanho=32):
    modelo.eval()
    saida = np.zeros((len(ids), 18), dtype=np.float32)
    for indices in lotes(ids, tamanho, np.random.default_rng(0), False):
        x, m = montar_lote(ids, indices, pad, dispositivo)
        with torch.autocast(device_type=dispositivo, dtype=tipo, enabled=tipo != torch.float32):
            saida[indices] = modelo(input_ids=x, attention_mask=m).logits.float().cpu().numpy()
    return saida


def treinar(modelo, ids, rotulos, epocas, lr, nome, dispositivo, tipo, pad, tok):
    """Laço próprio: AdamW, aquecimento e queda cosseno, acumulação de gradiente, limite de tempo e gravação."""
    rng = np.random.default_rng(1)
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.AdamW(parametros, lr=lr, weight_decay=0.0)
    passos_por_epoca = math.ceil(len(ids) / (CFG["micro_lote"] * CFG["acumular"]))
    total = passos_por_epoca * epocas
    escala = torch.amp.GradScaler("cuda", enabled=(tipo == torch.float16 and dispositivo == "cuda"))
    y = torch.tensor(rotulos, dtype=torch.long)
    passo, tokens, perda_media = 0, 0, None
    modelo.train()
    t0 = time.time()
    for epoca in range(epocas):
        micro = lotes(ids, CFG["micro_lote"], rng, True)
        for g in range(0, len(micro), CFG["acumular"]):
            grupo = micro[g : g + CFG["acumular"]]
            fator = min(1.0, (passo + 1) / max(1, int(0.03 * total)))
            decai = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * passo / max(1, total)))
            for p in otimizador.param_groups:
                p["lr"] = lr * fator * decai
            for indices in grupo:
                x, m = montar_lote(ids, indices, pad, dispositivo)
                with torch.autocast(device_type=dispositivo, dtype=tipo, enabled=tipo != torch.float32):
                    logits = modelo(input_ids=x, attention_mask=m).logits
                perda = torch.nn.functional.cross_entropy(logits.float(), y[indices].to(dispositivo)) / len(grupo)
                escala.scale(perda).backward()
                tokens += int(m.sum())
                perda_media = perda.item() * len(grupo) if perda_media is None else 0.98 * perda_media + 0.02 * perda.item() * len(grupo)
            escala.unscale_(otimizador)
            norma = torch.nn.utils.clip_grad_norm_(parametros, 1.0)
            if torch.isfinite(norma):
                escala.step(otimizador)
            else:
                log(f"passo {passo}: gradiente não finito, passo ignorado")
            escala.update()
            otimizador.zero_grad(set_to_none=True)
            passo += 1
            if passo % 20 == 0:
                log(f"{nome} época {epoca + 1}/{epocas} passo {passo}/{total} perda {perda_media:.3f} "
                    f"{tokens / (time.time() - t0):.0f} tokens/s")
            if passo % CFG["salvar_cada"] == 0:
                modelo.save_pretrained(Path(CFG["saida"]) / f"adaptador_{nome}")
            if time.time() - t0 > CFG["tempo_maximo_s"]:
                log(f"{nome}: limite de tempo ({CFG['tempo_maximo_s']}s) no passo {passo}/{total}")
                modelo.save_pretrained(Path(CFG["saida"]) / f"adaptador_{nome}")
                return passo, tokens / (time.time() - t0)
    modelo.save_pretrained(Path(CFG["saida"]) / f"adaptador_{nome}")
    return passo, tokens / max(1e-9, time.time() - t0)


def nota(y, logits):
    y = np.asarray(y)
    previsto = logits.argmax(1)
    return float(np.mean([(previsto[y == c] == c).mean() for c in np.unique(y)]))


def main():
    from transformers import AutoTokenizer

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    saida = Path(CFG["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    preparar_ambiente()
    dispositivo, tipo = dispositivo_e_tipo()
    log("config", json.dumps(CFG))
    oficial, pool, classes = carregar_dados()
    indice = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(CFG["modelo"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids_pool, ids_of = tokenizar(tok, pool.text.tolist()), tokenizar(tok, oficial.text.tolist())
    y_pool, y_of = pool.model.map(indice).to_numpy(), oficial.model.map(indice).to_numpy()
    log(f"pool {len(pool)} textos, {np.mean([len(i) for i in ids_pool]):.0f} tokens em média; oficial {len(oficial)}")
    modelo = montar_modelo(tok, dispositivo, tipo)
    resultado = {"config": CFG, "classes": classes}

    passos, vel = treinar(modelo, ids_pool, y_pool, CFG["epocas_pool"], CFG["lr_pool"], "pool", dispositivo, tipo, pad, tok)
    resultado["etapa1"] = {"passos": passos, "tokens_por_s": vel}
    avaliar = np.arange(len(oficial))
    if CFG["avaliar_max_textos"]:
        avaliar = np.random.default_rng(2).permutation(len(oficial))[: CFG["avaliar_max_textos"]]
    logits = prever(modelo, [ids_of[i] for i in avaliar], pad, dispositivo, tipo)
    resultado["etapa1"]["balanced_accuracy_oficial"] = nota(y_of[avaliar], logits)
    resultado["etapa1"]["textos_avaliados"] = int(len(avaliar))
    np.save(saida / "logits_oficial_etapa1.npy", logits)
    np.save(saida / "indices_avaliados_etapa1.npy", avaliar)
    log("ETAPA 1: balanced accuracy nos oficiais", resultado["etapa1"]["balanced_accuracy_oficial"], f"({len(avaliar)} textos)")

    if CFG["epocas_oficial"]:
        prompts = sorted(oficial.prompt_id.unique())
        reservados = set(prompts[::5])  # 30 prompts
        tem = oficial.prompt_id.isin(reservados).to_numpy()
        treino, reserva = np.where(~tem)[0], np.where(tem)[0]
        antes = prever(modelo, [ids_of[i] for i in reserva], pad, dispositivo, tipo)
        passos, vel = treinar(modelo, [ids_of[i] for i in treino], y_of[treino], CFG["epocas_oficial"], CFG["lr_oficial"],
                              "oficial", dispositivo, tipo, pad, tok)
        depois = prever(modelo, [ids_of[i] for i in reserva], pad, dispositivo, tipo)
        resultado["etapa2"] = {"passos": passos, "tokens_por_s": vel, "reservados": int(len(reserva)),
                               "balanced_accuracy_antes": nota(y_of[reserva], antes), "balanced_accuracy_depois": nota(y_of[reserva], depois)}
        np.save(saida / "logits_reservados_etapa2.npy", depois)
        np.save(saida / "indices_reservados.npy", reserva)
        log("ETAPA 2: reservados antes/depois", resultado["etapa2"]["balanced_accuracy_antes"], resultado["etapa2"]["balanced_accuracy_depois"])

    (saida / "resultado.json").write_text(json.dumps(resultado, indent=2))
    log("pronto", json.dumps({k: v for k, v in resultado.items() if k.startswith("etapa")}))


if __name__ == "__main__":
    main()
