"""Etapa 1 do LoRA24: Qwen3 nas primeiras 24 camadas, lido em várias profundidades, treinado no pool público e medido
nos 2.699 textos oficiais, que nunca entram no treino (por isso a nota é uma medida limpa).

Arquitetura da receita pública: LoRA r16 em todas as lineares, média dos tokens em 8 profundidades (cada uma com
LayerNorm), cabeça linear, gradient checkpointing. Perda: entropia cruzada com amostra do pool balanceada por classe.
Treina em 1 ou mais GPUs com dados paralelos: cada GPU guarda uma cópia das camadas e os gradientes (só do LoRA e das
cabeças, ~30 M parâmetros) são somados por all_reduce, com escala de perda dinâmica própria (fp16 na T4), de modo que
todas as GPUs decidem juntas pular ou aplicar o passo.

Uso: python scripts/treino_lora24.py   (sem RANK no ambiente, ele mesmo lança um processo por GPU; "processos" na config
força outro número). Local (modelo minúsculo, CPU): TREINO_CFG='{"base": "...", "pool": "...", "oficial": "...",
"processos": 2}' python scripts/treino_lora24.py
No Kaggle: notebook gerado por scripts/montar_notebook_treino_lora24.py (internet desligada).
"""

import json
import math
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn

CFG = {
    "base": "",  # pasta do modelo; vazio = procurar em /kaggle/input pelo base_spec
    "base_spec": {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36},
    "tokenizer": "",  # vazio = o da base
    "keep": 24,
    "layers": [2, 4, 6, 8, 12, 16, 20, 24],
    "r": 16,
    "max_len": 384,
    "pool": "",  # vazio = procurar ntl_open_pool.parquet em /kaggle/input
    "oficial": "",  # vazio = procurar train.csv em /kaggle/input
    "n_pool_por_classe": 1944,  # 18 x 1944 = 34.992 textos
    "excluir_proxy_pos_regra": True,  # linhas geradas pelo proxy do Kaggle depois da regra do host (28/10)
    "micro_lote": 8,  # por GPU
    "acumular": 1,
    "epocas": 1,
    "lr_nucleo": 1e-4,
    "lr_cabeca": 1e-3,
    "aquecimento_frac": 0.10,
    "escala_inicial": 1024.0,  # escala da perda em fp16; em fp32 não se usa
    "max_passos": 0,  # 0 = até acabar as épocas
    "avaliar_cada": 500,  # passos; 0 desliga
    "avaliar_n": 400,  # textos oficiais nas avaliações intermediárias
    "avaliar_final_n": 0,  # 0 = todos os 2.699 oficiais
    "salvar_cada": 500,
    "tempo_maximo_s": 10800,
    "saida": "/kaggle/working",
    "backend": "",  # vazio = nccl com CUDA, gloo sem
    "timeout_min": 30,
    "processos": 0,  # 0 = um por GPU visível (ou 1 sem GPU)
    "perfil_passos": False,  # mede tempos por passo (forward/backward, espera entre GPUs, all_reduce); só no check
    # etapa 2: retoma o adaptador e as cabeças da etapa 1 e ajusta nos textos do host, reservando prompts inteiros
    "etapa": 1,
    "inicio": "",  # pasta com adaptador/ e cabecas.pt da etapa 1; vazio = procurar cabecas.pt em /kaggle/input
    "logits_etapa1": "",  # logits_oficial.npy da etapa 1 (confere a retomada); vazio = procurar em /kaggle/input
    "dobras": [0, 1],  # dobras por prompt reservadas, uma de cada vez: prompts[k::n_dobras]
    "n_dobras": 5,
    "final_todos": False,  # depois das dobras, ajusta em todos os oficiais e salva adaptador_etapa2/ (sem avaliação)
    # etapa 3: inferência no teste com o modelo final (não lê train.csv: a reexecução do Kaggle troca a competição)
    "teste": "",  # vazio = procurar test.csv em /kaggle/input
    "sufixo_inicio": "",  # sufixo dos arquivos a carregar: "" na etapa 1, "_etapa2" no modelo final
}
CLASSES_PADRAO = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
                  "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3",
                  "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3",
                  "qwen3.8-max"]  # ordem alfabética, a mesma do treino; só serve se as cabeças salvas não trouxerem as classes
CFG_JSON = "{}"  # o gerador do notebook troca esta linha
CFG.update(json.loads(os.environ.get("TREINO_CFG", CFG_JSON)))

RANK = int(os.environ.get("RANK", 0))
MUNDO = int(os.environ.get("WORLD_SIZE", 1))
LOCAL = int(os.environ.get("LOCAL_RANK", 0))
T0 = time.time()


def log(*args):
    if RANK == 0:
        print(f"[{time.time() - T0:6.0f}s]", *args, flush=True)


def preparar_ambiente():
    """O peft do Kaggle exige torchao >= 0.16 e a imagem traz o 0.10; sem o torchao o peft simplesmente não o usa."""
    import importlib.metadata as md

    try:
        versao = md.version("torchao")
    except md.PackageNotFoundError:
        return
    if tuple(int(x) for x in versao.split(".")[:2]) < (0, 16):
        log(f"torchao {versao} incompatível com o peft: removendo")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)


def achar(padrao: str) -> str:
    achados = sorted(Path("/kaggle/input").rglob(padrao))
    if not achados:
        raise FileNotFoundError(f"{padrao} não encontrado em /kaggle/input")
    return str(achados[0])


def achar_base() -> str:
    if CFG["base"]:
        return CFG["base"]
    spec = CFG["base_spec"]
    for cfg in sorted(Path("/kaggle/input").rglob("config.json")):
        try:
            c = json.loads(cfg.read_text())
        except Exception:
            continue
        if all(c.get(k) == v for k, v in spec.items()):
            return str(cfg.parent)
    raise FileNotFoundError(f"modelo {spec} não encontrado em /kaggle/input")


def normalizar(texto: str) -> str:
    return " ".join(texto.split()).lower()


def tokenizar(tok, textos):
    bruto = tok([str(t).strip() for t in textos], add_special_tokens=False)["input_ids"]
    return [x[: CFG["max_len"] - 1] + [tok.eos_token_id] for x in bruto]


def carregar_dados(tok, com_pool=True):
    oficial = pd.read_csv(CFG["oficial"] or achar("train.csv"), keep_default_na=False)
    classes = sorted(oficial.model.unique())
    assert len(classes) == 18, f"{len(classes)} classes nos oficiais"
    indice = {c: i for i, c in enumerate(classes)}
    ids_of, y_of = tokenizar(tok, oficial.text), oficial.model.map(indice).to_numpy()
    prompts = oficial.prompt_id.astype(str).to_numpy()
    if not com_pool:  # etapa 2 só usa os oficiais
        return [], np.zeros(0, dtype=int), ids_of, y_of, classes, prompts
    pool = pd.read_parquet(CFG["pool"] or achar("ntl_open_pool.parquet"),
                           columns=["text", "model", "n_words", "flag_kb_proxy_after_ruling"])
    n0 = len(pool)
    pool = pool[pool.n_words.between(30, 300)]
    if CFG["excluir_proxy_pos_regra"]:
        pool = pool[~pool.flag_kb_proxy_after_ruling.astype(bool)]
    oficiais = {normalizar(t) for t in oficial.text}  # nenhum texto oficial pode estar no treino
    pool = pool[~pool.text.map(normalizar).isin(oficiais)].drop_duplicates("text")
    assert set(pool.model) <= set(classes), f"classes do pool fora das oficiais: {set(pool.model) - set(classes)}"
    amostra = pd.concat([g.sample(min(len(g), CFG["n_pool_por_classe"]), random_state=0) for _, g in pool.groupby("model")])
    amostra = amostra.sample(frac=1, random_state=0).reset_index(drop=True)
    log(f"pool: {n0} linhas, {len(pool)} após filtros, {len(amostra)} na amostra balanceada "
        f"({amostra.model.value_counts().min()} a {amostra.model.value_counts().max()} por classe)")
    return tokenizar(tok, amostra.text), amostra.model.map(indice).to_numpy(), ids_of, y_of, classes, prompts


class Leitor(nn.Module):
    """Camadas do Qwen3 com LoRA, média dos tokens em várias profundidades (cada uma com LayerNorm) e cabeça linear."""

    def __init__(self, trunk, layers, hidden, n_classes):
        super().__init__()
        self.trunk, self.layers = trunk, layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in layers])
        self.head = nn.Linear(hidden * len(layers), n_classes)

    def forward(self, ids, mask):
        hs = self.trunk(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
        assert len(hs) == CFG["keep"] + 1, f"hidden_states com {len(hs)} entradas, esperado {CFG['keep'] + 1}"
        m = mask.float().clone()
        m[:, 0] = 0  # o primeiro token é um "attention sink": fora da média
        m = m.unsqueeze(-1)
        den = m.sum(1).clamp(min=1)
        z = torch.cat([nm((hs[l].float() * m).sum(1) / den) for l, nm in zip(self.layers, self.norms)], -1)
        return self.head(z)


def montar_modelo(base, dispositivo, tipo, n_classes):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModel

    mc = AutoConfig.from_pretrained(base)
    mc.num_hidden_layers, mc.use_cache = CFG["keep"], False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[: CFG["keep"]]
    if dispositivo.startswith("cuda"):
        trunk = AutoModel.from_pretrained(base, config=mc, dtype=tipo, device_map={"": LOCAL})
    else:
        trunk = AutoModel.from_pretrained(base, config=mc, dtype=tipo).to(dispositivo)
    trunk.norm = nn.Identity()  # a receita pública lê a camada 24 sem a normalização final
    trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trunk.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=CFG["r"], lora_alpha=2 * CFG["r"], lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    suf = CFG["sufixo_inicio"]
    if CFG["etapa"] in (2, 3):  # retoma o adaptador e as cabeças salvos (etapa 2: treina de novo; etapa 3: só infere)
        raiz = Path(CFG["inicio"]) if CFG["inicio"] else Path(achar(f"cabecas{suf}.pt")).parent
        trunk = PeftModel.from_pretrained(trunk, str(raiz / f"adaptador{suf}"), is_trainable=CFG["etapa"] == 2)
    else:
        trunk = get_peft_model(trunk, lora)
    for p in trunk.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # só o adaptador treina, em fp32
    leitor = Leitor(trunk, CFG["layers"], mc.hidden_size, n_classes)
    leitor.norms.to(dispositivo)
    leitor.head.to(dispositivo)
    if CFG["etapa"] in (2, 3):
        cabecas = torch.load(raiz / f"cabecas{suf}.pt", map_location=dispositivo)
        leitor.norms.load_state_dict(cabecas["norms"])
        leitor.head.load_state_dict(cabecas["head"])
    treinaveis = sum(p.numel() for p in leitor.parameters() if p.requires_grad)
    log(f"modelo com {CFG['keep']} camadas em {dispositivo}/{tipo}, treináveis {treinaveis / 1e6:.1f} M, {MUNDO} GPU(s)")
    return leitor


def parametros_treinaveis(leitor):
    nucleo = [p for n, p in leitor.named_parameters() if p.requires_grad and n.startswith("trunk")]
    cabeca = [p for n, p in leitor.named_parameters() if p.requires_grad and not n.startswith("trunk")]
    return nucleo, cabeca


def lotes(ids, tamanho, rng, mundo=1):
    """Lista global de lotes (iguais em todas as GPUs): tamanhos parecidos juntos, ordem embaralhada.
    Com várias GPUs, cada grupo de `mundo` lotes seguidos (um por GPU, no mesmo passo) tem comprimentos parecidos:
    sem isso cada passo espera o lote mais longo dos sorteados (medido: +26% de tempo com 2 GPUs)."""
    ordem = rng.permutation(len(ids))
    saida = []
    for i in range(0, len(ids), tamanho * 50):
        bloco = sorted(ordem[i : i + tamanho * 50], key=lambda j: len(ids[j]))
        saida += [bloco[k : k + tamanho] for k in range(0, len(bloco), tamanho)]
    saida = [b for b in saida if len(b) == tamanho]
    if mundo == 1:
        rng.shuffle(saida)
        return saida
    saida.sort(key=lambda b: max(len(ids[i]) for i in b))
    grupos = [saida[k : k + mundo] for k in range(0, len(saida) - mundo + 1, mundo)]
    for grupo in grupos:
        rng.shuffle(grupo)  # a mesma GPU não fica sempre com o lote mais curto do par
    rng.shuffle(grupos)
    return [b for grupo in grupos for b in grupo]


def sincronizar(dispositivo):
    if dispositivo.startswith("cuda"):
        torch.cuda.synchronize()
    elif dispositivo == "mps":
        torch.mps.synchronize()


def montar_lote(ids, indices, pad, dispositivo):
    maior = max(len(ids[i]) for i in indices)
    x = torch.full((len(indices), maior), pad, dtype=torch.long)
    m = torch.zeros((len(indices), maior), dtype=torch.long)
    for k, i in enumerate(indices):
        x[k, : len(ids[i])] = torch.tensor(ids[i])
        m[k, : len(ids[i])] = 1
    return x.to(dispositivo), m.to(dispositivo)


@torch.no_grad()
def prever(leitor, ids, pad, dispositivo, tipo, tamanho=16):
    """Logits de todos os textos; cada GPU faz uma fatia (i % mundo == rank) e as fatias são reunidas."""
    leitor.eval()
    if dispositivo.startswith("cuda"):
        torch.cuda.empty_cache()
    n = len(ids)
    meus = np.arange(n)[RANK::MUNDO]
    k = math.ceil(n / MUNDO)
    local = torch.zeros((k, 18), device=dispositivo)
    ordem = sorted(range(len(meus)), key=lambda j: len(ids[meus[j]]))
    tipo_disp = dispositivo.split(":")[0]
    for a in range(0, len(ordem), tamanho):
        pos = ordem[a : a + tamanho]
        x, m = montar_lote(ids, [meus[j] for j in pos], pad, dispositivo)
        with torch.autocast(device_type=tipo_disp, dtype=tipo, enabled=tipo != torch.float32):
            local[pos] = leitor(x, m).float()
    if MUNDO > 1:
        partes = [torch.zeros_like(local) for _ in range(MUNDO)]
        dist.all_gather(partes, local)
    else:
        partes = [local]
    saida = np.zeros((n, 18), dtype=np.float32)
    for r in range(MUNDO):
        quantos = len(np.arange(n)[r::MUNDO])
        saida[np.arange(n)[r::MUNDO]] = partes[r][:quantos].cpu().numpy()
    leitor.train()
    return saida


def nota(y, logits):
    y = np.asarray(y)
    previsto = logits.argmax(1)
    return float(np.mean([(previsto[y == c] == c).mean() for c in np.unique(y)]))


def passo_otimizador(todos, otimizador, escala):
    """Soma os gradientes entre as GPUs, tira a escala, corta a norma e aplica. Retorna (aplicou, norma).
    Todas as GPUs recebem o mesmo vetor somado, então decidem igual: se uma teve inf/nan, todas pulam."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in todos]
    plano = torch.cat([g.reshape(-1) for g in grads])
    if MUNDO > 1:
        dist.all_reduce(plano)
    plano /= escala * MUNDO
    norma = plano.norm()
    if not bool(torch.isfinite(norma)):
        return False, float("nan")
    plano *= min(1.0, 1.0 / (float(norma) + 1e-6))
    inicio = 0
    for p, g in zip(todos, grads):
        p.grad = plano[inicio : inicio + g.numel()].view_as(g).clone()
        inicio += g.numel()
    otimizador.step()
    return True, float(norma)


def treinar(leitor, ids, y, ids_of, y_of, pad, dispositivo, tipo, resultado):
    nucleo, cabeca = parametros_treinaveis(leitor)
    todos = nucleo + cabeca
    otimizador = torch.optim.AdamW([{"params": nucleo, "lr": CFG["lr_nucleo"]}, {"params": cabeca, "lr": CFG["lr_cabeca"]}],
                                   weight_decay=0.0)
    base_lr = [CFG["lr_nucleo"], CFG["lr_cabeca"]]
    fp16 = tipo == torch.float16
    escala = float(CFG["escala_inicial"]) if fp16 else 1.0
    por_passo = MUNDO * CFG["acumular"]
    passos_epoca = len(lotes(ids, CFG["micro_lote"], np.random.default_rng(1), MUNDO)) // por_passo
    total = passos_epoca * CFG["epocas"]
    if CFG["max_passos"]:
        total = min(total, CFG["max_passos"])
    aquec = max(1, int(CFG["aquecimento_frac"] * total))
    tipo_disp = dispositivo.split(":")[0]
    yt = torch.as_tensor(y, dtype=torch.long)
    avaliar_idx = np.random.default_rng(2).permutation(len(ids_of))[: CFG["avaliar_n"]]
    ids_av, y_av = [ids_of[i] for i in avaliar_idx], y_of[avaliar_idx]
    resultado.update(passos_total_previsto=total, passos_por_epoca=passos_epoca, avaliacoes=[], historico_perda=[])
    log(f"{len(ids)} textos, {passos_epoca} passos por época, {total} passos no total, micro-lote {CFG['micro_lote']} por GPU")

    leitor.train()
    passo, tokens, perda_media, pulados, bons_seguidos = 0, 0, None, 0, 0
    t_aval = 0.0  # tempo gasto avaliando: fora da velocidade de treino
    perfil, acum = CFG["perfil_passos"], {"passos": 0, "fwbw_s": 0.0, "espera_s": 0.0, "all_reduce_e_otimizador_s": 0.0}
    t_ini = time.time()
    parar = False
    for epoca in range(CFG["epocas"]):
        fila = lotes(ids, CFG["micro_lote"], np.random.default_rng(1 + epoca), MUNDO)
        for g in range(passos_epoca):
            if passo >= total or parar:
                break
            fator = min(1.0, (passo + 1) / aquec) * 0.5 * (1 + math.cos(math.pi * passo / max(1, total)))
            for grupo, lr in zip(otimizador.param_groups, base_lr):
                grupo["lr"] = lr * fator
            t_a = time.time()
            for a in range(CFG["acumular"]):
                indices = fila[(g * CFG["acumular"] + a) * MUNDO + RANK]
                x, m = montar_lote(ids, indices, pad, dispositivo)
                with torch.autocast(device_type=tipo_disp, dtype=tipo, enabled=tipo != torch.float32):
                    logits = leitor(x, m)
                perda = torch.nn.functional.cross_entropy(logits.float(), yt[indices].to(dispositivo))
                (perda * escala / CFG["acumular"]).backward()
                tokens += int(m.sum())
                v = float(perda)
                if math.isfinite(v):
                    perda_media = v if perda_media is None else 0.98 * perda_media + 0.02 * v
            if perfil:  # separa: forward/backward, espera pela outra GPU (desbalanceamento) e all_reduce + otimizador
                sincronizar(dispositivo)
                t_b = time.time()
                if MUNDO > 1:
                    dist.barrier()
                    sincronizar(dispositivo)
                t_c = time.time()
            aplicou, _ = passo_otimizador(todos, otimizador, escala)
            otimizador.zero_grad(set_to_none=True)
            if perfil:
                sincronizar(dispositivo)
                if passo >= 3:  # sem o aquecimento do CUDA
                    acum["passos"] += 1
                    acum["fwbw_s"] += t_b - t_a
                    acum["espera_s"] += t_c - t_b
                    acum["all_reduce_e_otimizador_s"] += time.time() - t_c
            if aplicou:
                bons_seguidos += 1
                if fp16 and bons_seguidos >= 200 and escala < 2 ** 16:
                    escala, bons_seguidos = escala * 2, 0
            else:
                pulados += 1
                bons_seguidos = 0
                if fp16:
                    escala = max(1.0, escala / 2)
                log(f"passo {passo}: gradiente não finito, passo pulado; escala {escala:g}")
            passo += 1
            if passo % 20 == 0:
                dt = time.time() - t_ini - t_aval
                resultado["historico_perda"].append([passo, perda_media])
                eta = dt / passo * (total - passo)
                log(f"passo {passo}/{total} perda {perda_media:.3f} {tokens * MUNDO / dt:.0f} tokens/s (todas as GPUs) "
                    f"escala {escala:g} pulados {pulados} ETA {eta / 60:.0f} min")
                if MUNDO > 1:  # as GPUs decidem juntas parar por tempo
                    flag = torch.tensor([1.0 if time.time() - T0 > CFG["tempo_maximo_s"] else 0.0], device=dispositivo)
                    dist.broadcast(flag, 0)
                    parar = bool(flag.item())
                else:
                    parar = time.time() - T0 > CFG["tempo_maximo_s"]
                if parar:
                    log(f"tempo máximo atingido no passo {passo}")
            if CFG["avaliar_cada"] and passo % CFG["avaliar_cada"] == 0:
                t_e = time.time()
                ba = nota(y_av, prever(leitor, ids_av, pad, dispositivo, tipo))
                t_aval += time.time() - t_e
                resultado["avaliacoes"].append({"passo": passo, "balanced_accuracy": ba, "textos": len(ids_av)})
                log(f"AVALIAÇÃO passo {passo}: balanced accuracy {ba:.4f} em {len(ids_av)} textos oficiais")
            if CFG["salvar_cada"] and passo % CFG["salvar_cada"] == 0:
                salvar(leitor)
        if passo >= total or parar:
            break
    dt = time.time() - t_ini - t_aval
    resultado.update(passos=passo, passos_pulados=pulados, escala_final=escala, tempo_treino_s=dt, tempo_avaliacao_s=t_aval,
                     tokens_por_s_todas_gpus=tokens * MUNDO / dt, perda_final=perda_media)
    if perfil and acum["passos"]:
        n = acum["passos"]
        resultado["perfil_passos"] = {"passos_medidos": n, "fwbw_s": acum["fwbw_s"] / n, "espera_barreira_s": acum["espera_s"] / n,
                                      "all_reduce_e_otimizador_s": acum["all_reduce_e_otimizador_s"] / n}
        log("PERFIL por passo:", json.dumps({k: round(v, 3) for k, v in resultado["perfil_passos"].items()}))


def salvar(leitor, sufixo=""):
    if RANK != 0:
        return
    saida = Path(CFG["saida"])
    leitor.trunk.save_pretrained(saida / f"adaptador{sufixo}")
    torch.save({"norms": leitor.norms.state_dict(), "head": leitor.head.state_dict(),
                "classes": getattr(leitor, "classes", None)}, saida / f"cabecas{sufixo}.pt")


def nota_pred(y, previsto):
    return float(np.mean([(previsto[y == c] == c).mean() for c in np.unique(y)]))


def intervalo_bootstrap(y, antes, depois, prompts, n=2000, seed=0):
    """IC95% do ganho em balanced accuracy (depois menos antes), reamostrando prompts inteiros."""
    rng = np.random.default_rng(seed)
    unicos = np.unique(prompts)
    por_prompt = {p: np.where(prompts == p)[0] for p in unicos}
    deltas = []
    for _ in range(n):
        idx = np.concatenate([por_prompt[p] for p in rng.choice(unicos, len(unicos))])
        deltas.append(nota_pred(y[idx], depois[idx]) - nota_pred(y[idx], antes[idx]))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def verificar_retomada(leitor, ids_of, pad, dispositivo, tipo, resultado):
    """A retomada tem de reproduzir os logits que a etapa 1 salvou; senão o adaptador ou as cabeças não carregaram."""
    salvos = np.load(CFG["logits_etapa1"] or achar("logits_oficial.npy"))
    n = min(128, len(ids_of))
    novos = prever(leitor, ids_of[:n], pad, dispositivo, tipo)
    concordancia = float((novos.argmax(1) == salvos[:n].argmax(1)).mean())
    resultado["retomada"] = {"textos": n, "concordancia_argmax": concordancia,
                             "maior_diferenca_logit": float(np.abs(novos - salvos[:n]).max())}
    log(f"RETOMADA: {concordancia:.3f} de concordância com os logits da etapa 1, maior diferença "
        f"{resultado['retomada']['maior_diferenca_logit']:.4f}")
    if concordancia < 0.9:
        raise RuntimeError(f"a retomada não reproduz a etapa 1 (concordância {concordancia:.3f}): abortando antes de treinar")


def executar_etapa2(leitor, ids_of, y_of, prompts, classes, pad, dispositivo, tipo, resultado):
    """Ajusta nos oficiais, uma dobra por prompt de cada vez, medindo as mesmas passagens antes e depois."""
    todos = sum(parametros_treinaveis(leitor), [])
    inicial = [p.detach().clone() for p in todos]
    verificar_retomada(leitor, ids_of, pad, dispositivo, tipo, resultado)

    def restaurar():
        with torch.no_grad():
            for p, p0 in zip(todos, inicial):
                p.copy_(p0)

    unicos = sorted(set(prompts))
    antes_pred, depois_pred = np.full(len(y_of), -1), np.full(len(y_of), -1)
    resultado["dobras"] = []
    for k in CFG["dobras"]:
        restaurar()
        reservados = set(unicos[k :: CFG["n_dobras"]])
        reservado = np.array([p in reservados for p in prompts])
        ev, tr = np.where(reservado)[0], np.where(~reservado)[0]
        ids_ev = [ids_of[i] for i in ev]
        antes = prever(leitor, ids_ev, pad, dispositivo, tipo).argmax(1)
        info = {}
        treinar(leitor, [ids_of[i] for i in tr], y_of[tr], ids_of, y_of, pad, dispositivo, tipo, info)
        depois = prever(leitor, ids_ev, pad, dispositivo, tipo).argmax(1)
        antes_pred[ev], depois_pred[ev] = antes, depois
        d = {"dobra": k, "prompts_reservados": len(reservados), "treino": int(len(tr)), "avaliacao": int(len(ev)),
             "passos": info["passos"], "passos_pulados": info["passos_pulados"], "perda_final": info["perda_final"],
             "tempo_treino_s": info["tempo_treino_s"], "ba_antes": nota_pred(y_of[ev], antes),
             "ba_depois": nota_pred(y_of[ev], depois)}
        d["delta"] = d["ba_depois"] - d["ba_antes"]
        resultado["dobras"].append(d)
        log(f"DOBRA {k}: {len(ev)} passagens reservadas, balanced accuracy {d['ba_antes']:.4f} -> {d['ba_depois']:.4f} "
            f"({d['delta']:+.4f})")

    if CFG["dobras"]:  # sem dobras (ajuste final direto) não há o que comparar
        medido = antes_pred >= 0
        y, a, b, pr = y_of[medido], antes_pred[medido], depois_pred[medido], prompts[medido]
        familia = np.array([c.split("-")[0] for c in classes])
        resultado["pareado"] = {
            "passagens": int(medido.sum()), "prompts": int(len(np.unique(pr))),
            "ba_antes": nota_pred(y, a), "ba_depois": nota_pred(y, b), "delta": nota_pred(y, b) - nota_pred(y, a),
            "ic95_delta_por_prompt": intervalo_bootstrap(y, a, b, pr),
            "acuracia_familia_antes": float((familia[a] == familia[y]).mean()),
            "acuracia_familia_depois": float((familia[b] == familia[y]).mean()),
            "recall_antes": {classes[c]: float((a[y == c] == c).mean()) for c in np.unique(y)},
            "recall_depois": {classes[c]: float((b[y == c] == c).mean()) for c in np.unique(y)},
        }
        p = resultado["pareado"]
        log(f"PAREADO ({p['passagens']} passagens, {p['prompts']} prompts): {p['ba_antes']:.4f} -> {p['ba_depois']:.4f} "
            f"({p['delta']:+.4f}, IC95% {p['ic95_delta_por_prompt'][0]:+.4f} a {p['ic95_delta_por_prompt'][1]:+.4f})")

    if CFG["final_todos"]:  # modelo para enviar: ajuste em todos os oficiais, sem avaliação possível
        restaurar()
        info = {}
        treinar(leitor, ids_of, y_of, ids_of, y_of, pad, dispositivo, tipo, info)
        salvar(leitor, "_etapa2")
        resultado["final_todos"] = {"treino": int(len(ids_of)), "passos": info["passos"], "passos_pulados": info["passos_pulados"],
                                    "perda_final": info["perda_final"]}


def inferir(tok, base, pad, dispositivo, tipo, saida):
    """Etapa 3: prevê o teste com o modelo final e grava submission.parquet (id, model). Não lê train.csv."""
    suf = CFG["sufixo_inicio"]
    raiz = Path(CFG["inicio"]) if CFG["inicio"] else Path(achar(f"cabecas{suf}.pt")).parent
    classes = torch.load(raiz / f"cabecas{suf}.pt", map_location="cpu").get("classes") or CLASSES_PADRAO
    leitor = montar_modelo(base, dispositivo, tipo, len(classes))
    teste = pd.read_csv(CFG["teste"] or achar("test.csv"), keep_default_na=False)
    log(f"teste: {len(teste)} textos; {len(classes)} classes")
    t0 = time.time()
    logits = prever(leitor, tokenizar(tok, teste.text), pad, dispositivo, tipo)
    previsto = np.array(classes)[logits.argmax(1)]
    if RANK == 0:
        pd.DataFrame({"id": teste["id"].astype(str), "model": previsto}).to_parquet(saida / "submission.parquet", index=False)
        np.save(saida / "logits_teste.npy", logits)
        contagem = pd.Series(previsto).value_counts()
        log(f"submission.parquet com {len(previsto)} linhas em {time.time() - t0:.0f}s; classes previstas: "
            f"{contagem.min()} a {contagem.max()} por classe ({len(contagem)} de {len(classes)})")
    if MUNDO > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    from transformers import AutoTokenizer

    saida = Path(CFG["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    if cuda:
        dispositivo, tipo = f"cuda:{LOCAL}", torch.float16  # T4 não tem bf16 nativo
        torch.cuda.set_device(LOCAL)
    elif MUNDO == 1 and torch.backends.mps.is_available() and os.environ.get("TREINO_MPS"):
        dispositivo, tipo = "mps", torch.bfloat16
    else:
        dispositivo, tipo = "cpu", torch.float32
    if MUNDO > 1:
        backend = CFG["backend"] or ("nccl" if cuda else "gloo")
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=CFG["timeout_min"]))
        log(f"distribuído: backend {backend}, {MUNDO} processos")
    if RANK == 0:
        preparar_ambiente()
    if MUNDO > 1:
        dist.barrier()
    torch.manual_seed(0)
    np.random.seed(0)

    base = achar_base()
    tok = AutoTokenizer.from_pretrained(CFG["tokenizer"] or base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if CFG["etapa"] == 3:
        return inferir(tok, base, pad, dispositivo, tipo, saida)
    ids, y, ids_of, y_of, classes, prompts = carregar_dados(tok, com_pool=CFG["etapa"] != 2)
    log(f"{np.mean([len(i) for i in ids]) if ids else 0:.0f} tokens por texto do pool em média; {len(ids_of)} textos oficiais")
    t_carga = time.time()
    leitor = montar_modelo(base, dispositivo, tipo, len(classes))
    leitor.classes = classes  # vai junto com as cabeças salvas: a inferência não precisa do train.csv
    if MUNDO > 1:  # mesmas cabeças e mesmo LoRA em todas as GPUs, mesmo que a inicialização divirja
        for p in sum(parametros_treinaveis(leitor), []):
            dist.broadcast(p.data, 0)
    resultado = {"config": CFG, "classes": classes, "mundo": MUNDO, "carregamento_s": time.time() - t_carga,
                 "dispositivo": dispositivo}
    if cuda:
        resultado["gpu"] = torch.cuda.get_device_name(LOCAL)

    if CFG["etapa"] == 2:
        executar_etapa2(leitor, ids_of, y_of, prompts, classes, pad, dispositivo, tipo, resultado)
        if RANK == 0:
            (saida / "resultado.json").write_text(json.dumps(resultado, indent=2, default=float))
        if MUNDO > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    treinar(leitor, ids, y, ids_of, y_of, pad, dispositivo, tipo, resultado)

    # as GPUs têm de terminar com os mesmos parâmetros treináveis
    soma = torch.stack([p.detach().double().sum() for p in sum(parametros_treinaveis(leitor), [])]).sum().reshape(1)
    somas = [torch.zeros_like(soma) for _ in range(MUNDO)]
    if MUNDO > 1:
        dist.all_gather(somas, soma)
    else:
        somas[0] = soma
    resultado["diferenca_params_entre_gpus"] = float(max(s.item() for s in somas) - min(s.item() for s in somas))
    salvar(leitor)

    idx = np.arange(len(ids_of))[: CFG["avaliar_final_n"] or len(ids_of)]
    logits = prever(leitor, [ids_of[i] for i in idx], pad, dispositivo, tipo)
    yv = y_of[idx]
    previsto = logits.argmax(1)
    familia = np.array([c.split("-")[0] for c in classes])
    resultado["final"] = {
        "textos": int(len(idx)), "balanced_accuracy": nota(yv, logits),
        "acuracia_familia": float((familia[previsto] == familia[yv]).mean()),
        "recall_por_classe": {classes[c]: float((previsto[yv == c] == c).mean()) for c in np.unique(yv)},
    }
    if RANK == 0:
        np.save(saida / "logits_oficial.npy", logits)
        np.save(saida / "y_oficial.npy", yv)
        (saida / "resultado.json").write_text(json.dumps(resultado, indent=2))
    log("FINAL balanced accuracy nos oficiais:", round(resultado["final"]["balanced_accuracy"], 4),
        "| família:", round(resultado["final"]["acuracia_familia"], 4),
        "| passos pulados:", resultado["passos_pulados"], "| diferença entre GPUs:", resultado["diferenca_params_entre_gpus"])
    if MUNDO > 1:
        dist.barrier()
        dist.destroy_process_group()


def lancar() -> int:
    """Lança um processo por GPU com o ambiente que o init_process_group espera (endereço fixo em 127.0.0.1: o
    torchrun --standalone trava no macOS ao resolver o hostname). Se um processo cair, derruba os outros."""
    import shutil
    import socket

    if CFG["processos"]:
        n = CFG["processos"]
    else:
        listagem = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
        n = max(1, sum(1 for linha in (listagem.stdout.splitlines() if listagem else []) if linha.startswith("GPU ")))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        porta = s.getsockname()[1]
    env = dict(os.environ, WORLD_SIZE=str(n), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(porta), TREINO_CFG=json.dumps(CFG))
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    print(f"lançando {n} processo(s)", flush=True)
    procs = [subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                              env=dict(env, RANK=str(i), LOCAL_RANK=str(i))) for i in range(n)]
    while True:
        codigos = [p.poll() for p in procs]
        if all(c is not None for c in codigos):
            break
        if any(c not in (None, 0) for c in codigos):
            print(f"um processo terminou com erro {codigos}: derrubando os outros", flush=True)
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            break
        time.sleep(2)
    codigos = [p.wait() for p in procs]
    print("códigos de saída:", codigos, flush=True)
    return 0 if all(c == 0 for c in codigos) else 1


if __name__ == "__main__":
    if "RANK" in os.environ:
        main()
    else:
        sys.exit(lancar())
