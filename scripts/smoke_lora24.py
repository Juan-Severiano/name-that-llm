"""Smoke de velocidade do LoRA24: primeiras 24 camadas de um Qwen3, leitura em várias profundidades, em T4.

Mede tokens/s, memória de pico e passos com perda ou gradiente não finito, por GPU e por tamanho de micro-lote,
com a arquitetura da receita pública (LoRA r16 em todas as lineares, 8 profundidades com LayerNorm, cabeça linear,
gradient checkpointing). Não avalia acurácia: serve para decidir se vale gastar a cota de GPU no treino de verdade.

Um worker por GPU roda ao mesmo tempo, o que também testa o treino em duas T4 (cada uma com sua cópia das camadas).
Local, com um modelo minúsculo: SMOKE_CFG='{"base": "...", "tokenizer": "...", "pool": "..."}' python scripts/smoke_lora24.py
No Kaggle: notebook gerado por scripts/montar_notebook_smoke_lora24.py (internet desligada).
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CFG = {
    "base": "",  # pasta do modelo; vazio = procurar em /kaggle/input pelo base_spec
    "base_spec": {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36},
    "tokenizer": "",  # vazio = o da base
    "keep": 24,  # camadas mantidas
    "layers": [2, 4, 6, 8, 12, 16, 20, 24],  # profundidades lidas (hidden_states[l])
    "r": 16,
    "max_len": 384,
    "micro_lotes": [4, 8, 16],
    "passos": 20,  # passos medidos por tamanho de micro-lote
    "aquecimento": 3,  # passos antes de começar a cronometrar
    "n_textos": 2000,
    "pool": "",  # vazio = procurar ntl_open_pool.parquet em /kaggle/input
    "tempo_maximo_s": 900,  # teto do treino de cada worker (fora o carregamento)
    "saida": "/kaggle/working",
    "gpus": 0,  # 0 = todas as visíveis
    "usar_mps": False,  # sem CUDA, usa a GPU do Mac (bf16); padrão é CPU, que serve só ao teste de mecânica
    "textos_projecao": [35000, 223840],
}
CFG_JSON = "{}"  # o gerador do notebook troca esta linha
CFG.update(json.loads(os.environ.get("SMOKE_CFG", CFG_JSON)))

T0 = time.time()
GPU_TAG = os.environ.get("SMOKE_GPU")  # definido só nos workers


def log(*args):
    rotulo = "" if GPU_TAG is None else f" gpu{GPU_TAG}"
    print(f"[{time.time() - T0:6.0f}s{rotulo}]", *args, flush=True)


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


def achar_pool() -> str:
    return CFG["pool"] or str(next(Path("/kaggle/input").rglob("ntl_open_pool.parquet")))


def carregar_dados(tok):
    pool = pd.read_parquet(achar_pool(), columns=["text", "model", "n_words"])
    pool = pool[pool.n_words.between(30, 300)]
    classes = sorted(pool.model.unique())
    por_classe = max(1, CFG["n_textos"] // len(classes))
    partes = [g.sample(min(len(g), por_classe), random_state=0) for _, g in pool.groupby("model")]
    amostra = pd.concat(partes).sample(frac=1, random_state=0).reset_index(drop=True)
    bruto = tok([t.strip() for t in amostra.text], add_special_tokens=False)["input_ids"]
    ids = [x[: CFG["max_len"] - 1] + [tok.eos_token_id] for x in bruto]
    y = amostra.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    return ids, y, classes


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
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModel

    mc = AutoConfig.from_pretrained(base)
    mc.num_hidden_layers, mc.use_cache = CFG["keep"], False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[: CFG["keep"]]
    if dispositivo.startswith("cuda"):
        trunk = AutoModel.from_pretrained(base, config=mc, dtype=tipo, device_map={"": 0})
    else:
        trunk = AutoModel.from_pretrained(base, config=mc, dtype=tipo).to(dispositivo)
    trunk.norm = nn.Identity()  # a receita pública lê a camada 24 sem a normalização final
    trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trunk.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=CFG["r"], lora_alpha=2 * CFG["r"], lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trunk = get_peft_model(trunk, lora)
    for p in trunk.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # só o adaptador treina, em fp32
    leitor = Leitor(trunk, CFG["layers"], mc.hidden_size, n_classes)
    leitor.norms.to(dispositivo)
    leitor.head.to(dispositivo)
    treinaveis = sum(p.numel() for p in leitor.parameters() if p.requires_grad)
    log(f"modelo {base} com {CFG['keep']} camadas em {dispositivo}/{tipo}, treináveis {treinaveis / 1e6:.1f} M")
    return leitor


def lotes(ids, tamanho, rng):
    ordem = rng.permutation(len(ids))
    saida = []
    for i in range(0, len(ids), tamanho * 50):
        bloco = sorted(ordem[i : i + tamanho * 50], key=lambda j: len(ids[j]))  # tamanhos parecidos juntos
        saida += [bloco[k : k + tamanho] for k in range(0, len(bloco), tamanho)]
    rng.shuffle(saida)
    return [b for b in saida if len(b) == tamanho] or saida


def montar_lote(ids, indices, pad, dispositivo):
    maior = max(len(ids[i]) for i in indices)
    x = torch.full((len(indices), maior), pad, dtype=torch.long)
    m = torch.zeros((len(indices), maior), dtype=torch.long)
    for k, i in enumerate(indices):
        x[k, : len(ids[i])] = torch.tensor(ids[i])
        m[k, : len(ids[i])] = 1
    return x.to(dispositivo), m.to(dispositivo)


def sincronizar(dispositivo):
    if dispositivo.startswith("cuda"):
        torch.cuda.synchronize()
    elif dispositivo == "mps":
        torch.mps.synchronize()


def sem_memoria(erro) -> bool:
    """CUDA levanta OutOfMemoryError; o MPS levanta RuntimeError com 'out of memory' na mensagem."""
    return isinstance(erro, torch.cuda.OutOfMemoryError) or "out of memory" in str(erro).lower()


def medir(leitor, ids, y, mb, dispositivo, tipo, pad, deadline):
    """Treina `aquecimento + passos` micro-lotes de tamanho mb e cronometra os últimos `passos`."""
    nucleo = [p for n, p in leitor.named_parameters() if p.requires_grad and n.startswith("trunk")]
    cabeca = [p for n, p in leitor.named_parameters() if p.requires_grad and not n.startswith("trunk")]
    todos = nucleo + cabeca
    otimizador = torch.optim.AdamW([{"params": nucleo, "lr": 1e-4}, {"params": cabeca, "lr": 1e-3}], weight_decay=0.0)
    cuda = dispositivo.startswith("cuda")
    escala = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=(tipo == torch.float16 and cuda))
    tipo_disp = dispositivo.split(":")[0]
    total = CFG["aquecimento"] + CFG["passos"]
    fila = lotes(ids, mb, np.random.default_rng(mb))
    fila = (fila * math.ceil(total / len(fila)))[:total]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    r = {"micro_lote": mb, "oom": False, "pulados_gradiente": 0, "perda_nao_finita": 0, "passos_medidos": 0}
    perdas, t_ini, real, preench = [], None, 0, 0
    leitor.train()
    for k, indices in enumerate(fila):
        if time.time() > deadline:
            log(f"mb={mb}: tempo máximo, parando no passo {k}")
            break
        x, m = montar_lote(ids, indices, pad, dispositivo)
        yb = torch.as_tensor(y[indices], dtype=torch.long, device=dispositivo)
        if k == CFG["aquecimento"]:
            sincronizar(dispositivo)
            t_ini, real, preench = time.time(), 0, 0
        try:
            with torch.autocast(device_type=tipo_disp, dtype=tipo, enabled=tipo != torch.float32):
                logits = leitor(x, m)
            perda = torch.nn.functional.cross_entropy(logits.float(), yb)
            if not torch.isfinite(perda):
                r["perda_nao_finita"] += 1
                otimizador.zero_grad(set_to_none=True)
                continue
            escala.scale(perda).backward()
            escala.unscale_(otimizador)
            norma = torch.nn.utils.clip_grad_norm_(todos, 1.0)
            if torch.isfinite(norma):
                escala.step(otimizador)
            else:
                r["pulados_gradiente"] += 1
            escala.update()
            otimizador.zero_grad(set_to_none=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as erro:
            if not sem_memoria(erro):
                raise
            log(f"mb={mb}: sem memória no passo {k}")
            r["oom"] = True
            otimizador.zero_grad(set_to_none=True)
            del x, m
            if cuda:
                torch.cuda.empty_cache()
            break
        perdas.append(float(perda))
        if k >= CFG["aquecimento"]:
            real += int(m.sum())
            preench += x.numel()
            r["passos_medidos"] += 1
    sincronizar(dispositivo)
    if t_ini is not None and r["passos_medidos"]:
        dt = time.time() - t_ini
        r.update(s_por_passo=dt / r["passos_medidos"], tokens_reais_por_s=real / dt, tokens_com_pad_por_s=preench / dt)
    if cuda:
        r["pico_gb"] = torch.cuda.max_memory_allocated() / 1e9
    elif dispositivo == "mps":
        r["pico_gb"] = torch.mps.driver_allocated_memory() / 1e9  # inclui cache do driver: é o que o macOS enxerga
    if perdas:
        r["perda_primeiros3"], r["perda_ultimos3"] = float(np.mean(perdas[:3])), float(np.mean(perdas[-3:]))
    log("resultado", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    return r


def worker():
    from transformers import AutoTokenizer

    gpu = int(GPU_TAG)
    torch.manual_seed(gpu)
    cuda = torch.cuda.is_available()
    if cuda:
        dispositivo, tipo = "cuda:0", torch.float16  # T4 não tem bf16 nativo
    elif CFG["usar_mps"] and torch.backends.mps.is_available():
        dispositivo, tipo = "mps", torch.bfloat16
    else:
        dispositivo, tipo = "cpu", torch.float32
    info = {"gpu": gpu, "torch": torch.__version__, "dispositivo": dispositivo}
    if cuda:
        info["nome_gpu"] = torch.cuda.get_device_name(0)
        info["memoria_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    try:
        import transformers, peft

        info["transformers"], info["peft"] = transformers.__version__, peft.__version__
    except Exception as e:  # noqa: BLE001
        info["versoes_erro"] = repr(e)
    log("ambiente", json.dumps(info))
    base = achar_base()
    tok = AutoTokenizer.from_pretrained(CFG["tokenizer"] or base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids, y, classes = carregar_dados(tok)
    info["tokens_medios_por_texto"] = float(np.mean([len(i) for i in ids]))
    log(f"{len(ids)} textos, {info['tokens_medios_por_texto']:.0f} tokens em média, {len(classes)} classes")
    t_carga = time.time()
    leitor = montar_modelo(base, dispositivo, tipo, len(classes))
    info["carregamento_s"] = time.time() - t_carga
    if cuda:
        info["memoria_apos_carregar_gb"] = torch.cuda.memory_allocated() / 1e9
    elif dispositivo == "mps":
        info["memoria_apos_carregar_gb"] = torch.mps.current_allocated_memory() / 1e9
    deadline = time.time() + CFG["tempo_maximo_s"]
    medidas = []
    for mb in CFG["micro_lotes"]:
        medidas.append(medir(leitor, ids, y, mb, dispositivo, tipo, pad, deadline))
        if medidas[-1]["oom"]:
            break  # tamanhos maiores só pioram
    (Path(CFG["saida"]) / f"smoke_gpu{gpu}.json").write_text(json.dumps({"info": info, "medidas": medidas}, indent=2))
    log("worker pronto")


def launcher():
    preparar_ambiente()
    saida = Path(CFG["saida"])
    saida.mkdir(parents=True, exist_ok=True)
    listagem = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True) if shutil_which("nvidia-smi") else None
    n_gpus = sum(1 for linha in (listagem.stdout.splitlines() if listagem else []) if linha.startswith("GPU "))
    if listagem:
        log(listagem.stdout.strip())
    n = CFG["gpus"] or max(1, n_gpus)
    log(f"{n} worker(s), um por GPU" if n_gpus else "sem GPU: 1 worker na CPU (ensaio local)")
    procs = []
    for i in range(n):
        env = dict(os.environ, SMOKE_GPU=str(i), SMOKE_CFG=json.dumps(CFG))
        if n_gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve())], env=env))
    codigos = [p.wait() for p in procs]
    log("códigos de saída dos workers", codigos)

    por_gpu = []
    for i in range(n):
        arq = saida / f"smoke_gpu{i}.json"
        if arq.exists():
            por_gpu.append(json.loads(arq.read_text()))
    resumo = {"codigos": codigos, "gpus": []}
    total_tok_s = 0.0
    tokens_medios = None
    for g in por_gpu:
        validas = [m for m in g["medidas"] if m.get("tokens_reais_por_s") and not m["oom"]]
        melhor = max(validas, key=lambda m: m["tokens_reais_por_s"]) if validas else None
        tokens_medios = g["info"].get("tokens_medios_por_texto", tokens_medios)
        resumo["gpus"].append({"info": g["info"], "medidas": g["medidas"], "melhor": melhor})
        total_tok_s += melhor["tokens_reais_por_s"] if melhor else 0.0
    resumo["tokens_reais_por_s_total"] = total_tok_s
    if total_tok_s and tokens_medios:
        resumo["projecao_horas_1_epoca"] = {
            str(n_txt): round(n_txt * tokens_medios / total_tok_s / 3600, 4) for n_txt in CFG["textos_projecao"]}
    (saida / "smoke_resultado.json").write_text(json.dumps(resumo, indent=2))
    log("RESUMO", json.dumps({k: v for k, v in resumo.items() if k != "gpus"}))
    for g in resumo["gpus"]:
        log("melhor da gpu", g["info"].get("gpu"), json.dumps(g["melhor"]))


def shutil_which(nome):
    import shutil

    return shutil.which(nome)


if __name__ == "__main__":
    worker() if GPU_TAG is not None else launcher()
