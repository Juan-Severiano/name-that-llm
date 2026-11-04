"""Frozen Qwen3-4B/28 camadas: piloto, extração retomável e seleção CPU.

MPS exige --gpu-liberada e só deve ser usado após liberação do coordenador.
Não treina LoRA, não publica e não submete. A CV usa apenas os 120 prompts
de desenvolvimento; salva probabilidades dos outros 30 sem avaliar seus labels.
"""

import os
for _nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_nome] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import gc
import hashlib
import json
import re
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "data/modelos/Qwen3-4B-Base"
SAIDA = RAIZ / "runs/experimentos_qwen4b"
REVISAO = "1cfa9a7208912126459214e8b04321603b3df60c"
REVISOES = {
    "Qwen/Qwen3-4B": REVISAO,
    "Qwen/Qwen3-4B-Base": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
}
CAMADAS = (2, 4, 8, 12, 20, 28)
MANTER = 28
MAX_LEN = 384


def agora():
    return datetime.now(ZoneInfo("America/Fortaleza")).isoformat()


def salvar(path, dado):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dado, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(vals):
    return hashlib.sha256("\n".join(map(str, vals)).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parametros_config(c, n_camadas=None, norm_final=True):
    h, v, m = c.hidden_size, c.vocab_size, c.intermediate_size
    d = c.head_dim
    q, kv = c.num_attention_heads * d, c.num_key_value_heads * d
    por_camada = 2 * h * q + 2 * h * kv + 3 * h * m + 2 * h + 2 * d
    if c.attention_bias:
        por_camada += q + 2 * kv + h
    return v * h + (n_camadas or c.num_hidden_layers) * por_camada + (h if norm_final else 0)


def configurar(path=None, manter=MANTER):
    path = MODELO if path is None else path
    dados = json.loads((path / "config.json").read_text())
    dados["num_hidden_layers"] = manter
    dados["use_cache"] = False
    dados.pop("layer_types", None)
    return Qwen3Config.from_dict(dados)


def tokenizar(tok, textos, max_len=MAX_LEN):
    crus = tok(list(textos), add_special_tokens=False, truncation=False)["input_ids"]
    eos = tok.eos_token_id
    assert eos is not None
    completos = [list(x) if x and x[-1] == eos else [*x, eos] for x in crus]
    tamanhos = np.asarray([len(x) for x in completos], dtype=np.int32)
    ids = [x if len(x) <= max_len else [*x[:max_len - 1], eos] for x in completos]
    return ids, tamanhos


def orcamento(hash_pesos=False):
    c = AutoConfig.from_pretrained(MODELO, local_files_only=True)
    origem = json.loads((MODELO / "origem.json").read_text())
    assert origem["repo"] in REVISOES and origem["revision"] == REVISOES[origem["repo"]]
    index = json.loads((MODELO / "model.safetensors.index.json").read_text())
    nomes = sorted(set(index["weight_map"].values()))
    shards = []
    for nome in nomes:
        p = MODELO / nome
        d = {"nome": nome, "presente": p.exists(), "bytes": p.stat().st_size if p.exists() else None}
        if hash_pesos and p.exists():
            d["sha256"] = sha256(p)
            esperado_origem = origem.get("hashes_pesos_esperados", {}).get(nome)
            if esperado_origem:
                assert d["sha256"] == esperado_origem, f"Hash incorreto na origem oficial: {nome}"
                d["sha256_validado_origem_hf"] = True
            met = MODELO / ".cache/huggingface/download" / f"{nome}.metadata"
            if met.exists():
                linhas = met.read_text().splitlines()
                esperado = linhas[1] if len(linhas) > 1 else ""
                if re.fullmatch(r"[0-9a-f]{64}", esperado):
                    assert d["sha256"] == esperado, f"Hash incorreto: {nome}"
                    d["sha256_validado_cache_hf"] = True
        shards.append(d)
    total = parametros_config(c) * 2
    index_bytes = index["metadata"]["total_size"]
    if total != index_bytes:
        # O index oficial Base tem total_size 655.360B maior até que os arquivos
        # físicos. A API HF da revisão pinada informa a contagem real dos tensors.
        assert origem.get("parametros_safetensors_hf", -1) * 2 == total, (total, index_bytes)
    df = pd.read_csv(RAIZ / "data/train.csv", usecols=["id", "text"], keep_default_na=False)
    tok = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
    tok_prov = origem.get("tokenizer_provenance")
    if tok_prov:
        assert sha256(MODELO / "tokenizer_config.json") == tok_prov["canonical_sha256"]
        assert tok.eos_token_id == tok_prov["effective_eos_token_id"]
    input_ids, tamanhos = tokenizar(tok, df.text.tolist())
    h_tokens = hashlib.sha256()
    for seq in input_ids:
        h_tokens.update(np.asarray([len(seq), *seq], dtype=np.int32).tobytes())
    d = {"verificado_em": agora(), "modelo": origem["repo"], "revision": origem["revision"],
         "fonte_modelo": origem.get("fonte_kaggle", origem["repo"]),
         "revision_nota": "Revisão HF dos pesos; config/tokenizer locais identificados pelos hashes abaixo.",
         "camadas_originais": c.num_hidden_layers, "camadas_mantidas": MANTER,
         "camadas_extraidas": CAMADAS, "hidden_size": c.hidden_size,
         "parametros_originais_sem_lm_head": parametros_config(c),
         "index_total_size_declarado": index_bytes,
         "tensor_bytes_derivados_config": total, "index_total_size_consistente": total == index_bytes,
         "parametros_truncados_sem_norm": parametros_config(c, MANTER, False),
         "bytes_pesos_bf16_truncados": parametros_config(c, MANTER, False) * 2,
         "gib_pesos_bf16_truncados": parametros_config(c, MANTER, False) * 2 / 1024**3,
         "bytes_features_float16": len(df) * len(CAMADAS) * c.hidden_size * 2,
         "max_len": MAX_LEN, "maior_texto_com_eos": int(tamanhos.max()),
         "textos_truncados": int((tamanhos > MAX_LEN).sum()), "textos_total": len(df),
         "token_length_quantiles": np.percentile(tamanhos, [0, 25, 50, 75, 95, 100]).tolist(),
         "ids_sha256": digest(df.id), "textos_sha256": digest(df.text),
         "train_csv_sha256": sha256(RAIZ / "data/train.csv"),
         "eos_token_id": tok.eos_token_id, "input_ids_sha256": h_tokens.hexdigest(),
         "tokenizer_provenance": tok_prov,
         "tokenizer_files_sha256": {p.name: sha256(p) for p in MODELO.iterdir() if p.name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")},
         "config_sha256": sha256(MODELO / "config.json"),
         "index_sha256": sha256(MODELO / "model.safetensors.index.json"), "shards": shards,
         "nota_memoria": "Bytes de parâmetros não incluem buffers, ativações, alocações temporárias e overhead MPS; o piloto mede uso real."}
    salvar(SAIDA / "orcamento.json", d)
    return d


def media_sem_primeiro(hidden, attention_mask):
    mask = attention_mask.clone().float()
    mask[:, 0] = 0
    return (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


class CapturarCamadas:
    """Pool em hooks; nunca armazena/empilha todas as ativações por camada."""
    def __init__(self, modelo, camadas):
        self.camadas = tuple(camadas)
        self.mask = None
        self.valores = {}
        self.handles = [modelo.layers[c - 1].register_forward_hook(self._hook(c)) for c in self.camadas]

    def _hook(self, camada):
        def hook(_modulo, _entrada, saida):
            hidden = saida[0] if isinstance(saida, tuple) else saida
            self.valores[camada] = media_sem_primeiro(hidden, self.mask)
        return hook

    def iniciar(self, mask):
        self.mask = mask
        self.valores = {}

    def resultado(self):
        assert set(self.valores) == set(self.camadas)
        # Só 6 vetores já agregados, não [batch, layers, tokens, hidden].
        return torch.cat([self.valores[c].unsqueeze(1) for c in self.camadas], dim=1)

    def fechar(self):
        for h in self.handles:
            h.remove()
        self.valores = {}
        self.mask = None


def armazenamento(pasta, manifesto):
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / "manifesto.json"
    if arq.exists():
        existente = json.loads(arq.read_text())
        assert existente == manifesto, "Manifesto diferente: não reaproveitar cache de outro modelo/protocolo/ordem de ids."
        E = np.lib.format.open_memmap(pasta / "features.npy", mode="r+")
        done = np.lib.format.open_memmap(pasta / "concluidas.npy", mode="r+")
        assert tuple(E.shape) == tuple(manifesto["shape"]) and E.dtype == np.float16
        assert done.shape == (manifesto["shape"][0],) and done.dtype == np.bool_
        assert np.isfinite(E[np.asarray(done)]).all(), "Checkpoint marcado concluído contém NaN/Inf."
    else:
        E = np.lib.format.open_memmap(pasta / "features.npy", mode="w+", dtype=np.float16, shape=tuple(manifesto["shape"]))
        E[:] = np.nan
        E.flush()
        done = np.lib.format.open_memmap(pasta / "concluidas.npy", mode="w+", dtype=np.bool_, shape=(manifesto["shape"][0],))
        done[:] = False
        done.flush()
        salvar(arq, manifesto)
    return E, done


def finalizar_completo(E, done, manifesto, resumo):
    """Valida e publica conclusão mesmo após queda entre a última linha e o JSON."""
    assert tuple(E.shape) == tuple(manifesto["shape"]) and E.dtype == np.float16
    assert done.shape == (E.shape[0],) and done.dtype == np.bool_ and done.all()
    assert np.isfinite(E).all(), "Checkpoint concluído contém NaN/Inf."
    final = resumo | {"finalizado_em": agora(), "modo": "completo", "concluidas_total": int(done.sum()),
                      "linhas_total": E.shape[0], "shape": list(E.shape), "dtype_features": "float16",
                      "cache_validado_finito": True}
    salvar(SAIDA / "extracao_completa.json", final)
    return final


def selftest():
    # Só RNG CPU; torch.manual_seed também toca RNGs de outros backends.
    torch.random.default_generator.manual_seed(20261103)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, use_cache=False,
                      pad_token_id=0, eos_token_id=2, tie_word_embeddings=True)
    pasta = SAIDA / "selftest_tiny"
    pasta.mkdir(exist_ok=True)
    base = Qwen3ForCausalLM(cfg).cpu().eval()
    assert sum(p.numel() for p in base.model.parameters()) == parametros_config(cfg)
    base.save_pretrained(pasta, safe_serialization=True)
    pequena = configurar(pasta, manter=3)
    modelo, info = AutoModel.from_pretrained(pasta, config=pequena, local_files_only=True,
                                            dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"] and not info["mismatched_keys"]
    assert len(modelo.layers) == 3
    modelo.norm = torch.nn.Identity()
    modelo.requires_grad_(False).eval()
    assert sum(p.numel() for p in modelo.parameters()) == parametros_config(cfg, 3, False)
    cap = CapturarCamadas(modelo, (1, 2, 3))
    ids = torch.tensor([[1, 4, 7, 2, 0], [1, 8, 2, 0, 0]])
    mask = (ids != 0).long()
    with torch.inference_mode():
        cap.iniciar(mask)
        out = modelo(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        features = cap.resultado()
        assert out.past_key_values is None
        assert len(out.hidden_states) == 4
        referencia = torch.stack([media_sem_primeiro(out.hidden_states[c], mask) for c in (1, 2, 3)], dim=1)
        torch.testing.assert_close(features, referencia, atol=1e-6, rtol=1e-6)
        vazio = media_sem_primeiro(torch.randn(1, 3, 4), torch.tensor([[1, 0, 0]]))
        assert torch.isfinite(vazio).all() and not vazio.any()
        modelo.to(dtype=torch.bfloat16)
        cap.iniciar(mask)
        bf16 = modelo(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        bf16_features = cap.resultado()
        bf16_ref = torch.stack([media_sem_primeiro(bf16.hidden_states[c], mask) for c in (1, 2, 3)], dim=1)
        torch.testing.assert_close(bf16_features, bf16_ref, atol=1e-6, rtol=1e-6)
    cap.fechar()
    pasta_cp = SAIDA / "selftest_checkpoint"
    m = {"shape": [4, 3, 8], "ids_sha256": "tiny"}
    E, done = armazenamento(pasta_cp, m)
    E[2] = np.ones((3, 8), dtype=np.float16); E.flush()
    done[2] = True; done.flush()
    del E, done
    E, done = armazenamento(pasta_cp, m)
    assert done[2] and np.all(E[2] == 1) and not done[0]
    recusou = False
    try:
        armazenamento(pasta_cp, m | {"ids_sha256": "ordem-diferente"})
    except AssertionError:
        recusou = True
    assert recusou
    bloqueou_gpu = False
    try:
        extrair(argparse.Namespace(device="mps", gpu_liberada=False))
    except RuntimeError as e:
        bloqueou_gpu = "MPS só após liberação" in str(e)
    assert bloqueou_gpu
    d = {"verificado_em": agora(), "device": "cpu", "truncamento_carga": "4->3 camadas sem missing keys",
         "norm_final": "Identity", "pooling": "hooks igual hidden_states indexados, máscara ignora primeiro/padding, EOS incluso",
         "diferenca_maxima": float((features - referencia).abs().max()), "checkpoint_retomado": True,
         "bf16_pooling_diferenca_maxima": float((bf16_features - bf16_ref).abs().max()),
         "ordem_diferente_recusada": True, "guard_gpu_sem_liberacao": True,
         "dtype_model_cpu_testados": ["float32", "bfloat16"], "dtype_features": "float16", "gpu_usada": False}
    salvar(SAIDA / "selftest.json", d)
    print(json.dumps(d, ensure_ascii=False, indent=2), flush=True)


def extrair(args):
    if args.device == "mps" and not args.gpu_liberada:
        raise RuntimeError("MPS só após liberação do coordenador e --gpu-liberada.")
    b = orcamento(hash_pesos=True)
    assert all(s["presente"] for s in b["shards"]), "Download ainda incompleto."
    assert not b["textos_truncados"], "MAX_LEN=384 trunca textos; revisar protocolo antes de executar."
    assert b["bytes_pesos_bf16_truncados"] < 7_000_000_000
    df = pd.read_csv(RAIZ / "data/train.csv", usecols=["id", "text"], keep_default_na=False)
    tok = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
    ids, tamanhos = tokenizar(tok, df.text.tolist())
    manifesto = {"modelo": b["modelo"], "revision": b["revision"], "fonte_modelo": b["fonte_modelo"],
                 "revision_nota": b["revision_nota"], "keep_layers": MANTER,
                 "layers": CAMADAS, "max_len": MAX_LEN, "append_eos": True,
                 "pooling": "masked mean excluding token0, including EOS", "final_norm": "Identity",
                 "dtype_model": "bfloat16", "dtype_features": "float16", "batch_size": 1, "use_cache": False,
                 "shape": [len(df), len(CAMADAS), b["hidden_size"]], "ids_sha256": digest(df.id),
                 "train_csv_sha256": b["train_csv_sha256"],
                 "textos_sha256": digest(df.text), "config_sha256": b["config_sha256"],
                 "eos_token_id": b["eos_token_id"], "input_ids_sha256": b["input_ids_sha256"],
                 "tokenizer_provenance": b["tokenizer_provenance"],
                 "tokenizer_files_sha256": b["tokenizer_files_sha256"],
                 "index_sha256": b["index_sha256"], "shard_hashes": [{"nome": s["nome"], "sha256": s["sha256"]} for s in b["shards"]]}
    # JSON serializa tuplas como listas; normalizar antes da comparação de retomada.
    manifesto = json.loads(json.dumps(manifesto))
    E, done = armazenamento(SAIDA, manifesto)
    np.save(SAIDA / "ids.npy", df.id.to_numpy(dtype=str))
    np.save(SAIDA / "comprimentos_tokens.npy", tamanhos)
    np.save(SAIDA / "indices_originais.npy", np.arange(len(df)))
    ordem = np.argsort(tamanhos, kind="stable")
    escolhidas = ordem[np.linspace(0, len(ordem) - 1, args.pilot, dtype=int)] if args.pilot else ordem
    faltam = [int(i) for i in escolhidas if not done[i]]
    if not faltam:
        if not args.pilot:
            final = finalizar_completo(E, done, manifesto,
                                       {"device": args.device, "novas_linhas": 0, "retomada_ja_completa": True,
                                        "carga_segundos": 0.0, "inferencia_segundos": 0.0})
            print("FINAL_EXTRACAO", json.dumps(final), flush=True)
        print("Todas as linhas solicitadas já estão no checkpoint.", flush=True)
        return
    print("ANTES_CARGA", json.dumps({k: b[k] for k in ("gib_pesos_bf16_truncados", "maior_texto_com_eos", "textos_truncados")}), flush=True)
    inicio_carga = time.monotonic()
    config = configurar()
    modelo, info = AutoModel.from_pretrained(MODELO, config=config, local_files_only=True,
                                            dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    assert not info["missing_keys"] and not info["mismatched_keys"] and not info.get("error_msgs")
    for chave in info["unexpected_keys"]:
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", chave)
        assert (m and int(m[1]) >= MANTER) or chave == "lm_head.weight", chave
    modelo.norm = torch.nn.Identity()
    modelo.requires_grad_(False).eval()
    assert len(modelo.layers) == MANTER and all(p.dtype == torch.bfloat16 for p in modelo.parameters())
    assert sum(p.numel() * p.element_size() for p in modelo.parameters()) == b["bytes_pesos_bf16_truncados"]
    modelo.to(args.device)
    cap = CapturarCamadas(modelo, CAMADAS)
    carga_segundos = time.monotonic() - inicio_carga
    tempos, tokens = [], []
    max_driver = 0
    inicio = time.monotonic()
    try:
        with torch.inference_mode():
            for i in faltam:
                entrada = torch.tensor([ids[i]], device=args.device, dtype=torch.long)
                mask = torch.ones_like(entrada)
                cap.iniciar(mask)
                if args.device == "mps":
                    torch.mps.synchronize()
                t0 = time.monotonic()
                out = modelo(input_ids=entrada, attention_mask=mask, use_cache=False, output_hidden_states=False)
                assert out.past_key_values is None
                f = cap.resultado().to(dtype=torch.float16, device="cpu").numpy()[0]
                assert f.shape == tuple(E.shape[1:]) and np.isfinite(f).all()
                E[i] = f; E.flush()
                done[i] = True; done.flush()
                if args.device == "mps":
                    torch.mps.synchronize()
                    max_driver = max(max_driver, torch.mps.driver_allocated_memory())
                tempos.append(time.monotonic() - t0); tokens.append(len(ids[i]))
                del out, entrada, mask, f
                progresso = {"concluidas": int(done.sum()), "total": len(df), "nova_linha": i,
                             "tokens": len(ids[i]), "segundos_linha": tempos[-1], "mps_driver_bytes_observado_max": max_driver,
                             "segundos_sem_carga": time.monotonic() - inicio}
                salvar(SAIDA / "progresso.json", progresso)
                print("LINHA", json.dumps(progresso), flush=True)
    finally:
        cap.fechar()
        del modelo
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    segundos_por_token = sum(tempos[1:] or tempos) / sum(tokens[1:] or tokens)
    restantes = tamanhos[~np.asarray(done)]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    resumo = {"finalizado_em": agora(), "modo": "piloto" if args.pilot else "completo", "device": args.device,
              "novas_linhas": len(tempos), "concluidas_total": int(done.sum()), "linhas_total": len(df),
              "carga_segundos": carga_segundos, "inferencia_segundos": sum(tempos), "rss_max_mb": rss,
              "mps_driver_bytes_observado_max": max_driver, "segundos_por_token_sem_primeira_linha": segundos_por_token,
              "eta_restante_segundos_por_tokens": float(restantes.sum() * segundos_por_token),
              "eta_nota": "Extrapolação por tokens do piloto, não tempo garantido; alocações MPS são máximos observados entre linhas, não pico instantâneo."}
    if args.pilot:
        salvar(SAIDA / "piloto.json", resumo)
    else:
        resumo = finalizar_completo(E, done, manifesto, resumo)
    print("FINAL_EXTRACAO", json.dumps(resumo), flush=True)


def cv():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    out = SAIDA / "cv"
    out.mkdir(exist_ok=True)
    if (out / "selecao_congelada.json").exists():
        print("CV já concluída e seleção congelada; não repetir.", flush=True)
        return
    done = np.load(SAIDA / "concluidas.npy", allow_pickle=False)
    assert done.all(), f"Extração incompleta: {done.sum()}/{len(done)}"
    E = np.load(SAIDA / "features.npy", mmap_mode="r", allow_pickle=False)
    m = json.loads((SAIDA / "manifesto.json").read_text())
    df = pd.read_csv(RAIZ / "data/train.csv", keep_default_na=False)
    assert digest(df.id) == m["ids_sha256"] and digest(df.text) == m["textos_sha256"]
    assert sha256(RAIZ / "data/train.csv") == m["train_csv_sha256"]
    split = np.load(RAIZ / "runs/experimentos_estilo/split.npz", allow_pickle=False)
    dev, hold, classes = split["desenvolvimento"], split["holdout"], split["classes"]
    assert set(df.prompt_id.iloc[dev]).isdisjoint(df.prompt_id.iloc[hold])
    folds = list(GroupKFold(3).split(dev, df.model.iloc[dev], df.prompt_id.iloc[dev]))
    configs = [(camada, c) for camada in CAMADAS for c in (.001, .01, .1)]
    salvar(out / "protocolo.json", {"criado_em": agora(), "configs": configs, "n_configs": 18,
                                    "split": "120 prompts dev, GroupKFold3; labels dos 30 prompts reservados não avaliados",
                                    "historico": "Holdout já exposto historicamente por outras campanhas; esta seleção usa só CV dev.",
                                    "features_manifest_sha256": sha256(SAIDA / "manifesto.json")})
    def clf(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
    resultados, oofs = [], {}
    with threadpool_limits(limits=1):
        for camada, c in configs:
            inicio = time.monotonic()
            X = E[:, CAMADAS.index(camada)].astype(np.float32)
            nome = f"qwen4b_l{camada}_C{c:g}"
            p = np.empty((len(dev), len(classes)))
            notas = []
            for tr, va in folds:
                modelo = clf(c).fit(X[dev[tr]], df.model.iloc[dev[tr]])
                assert np.array_equal(modelo.classes_, classes)
                p[va] = modelo.predict_proba(X[dev[va]])
                notas.append(float(balanced_accuracy_score(df.model.iloc[dev[va]], classes[p[va].argmax(1)])))
            nota = float(balanced_accuracy_score(df.model.iloc[dev], classes[p.argmax(1)]))
            r = {"nome": nome, "camada": camada, "C": c, "balanced_accuracy_dev": nota, "dobras": notas, "segundos": time.monotonic() - inicio}
            oofs[nome] = p.astype(np.float32)
            resultados.append(r)
            salvar(out / "resultados_dev.json", resultados)
            print("CV", json.dumps(r), flush=True)
        np.savez_compressed(out / "inner_oof.npz", indices=dev, classes=classes, **oofs)
        melhor = max(resultados, key=lambda r: r["balanced_accuracy_dev"])
        salvar(out / "selecao_congelada.json", {"selecionado": melhor, "congelado_em": agora(), "holdout_avaliado": False})
        X = E[:, CAMADAS.index(melhor["camada"])].astype(np.float32)
        modelo = clf(melhor["C"]).fit(X[dev], df.model.iloc[dev])
        p = modelo.predict_proba(X[hold])
        np.savez_compressed(out / "probas_holdout_sem_avaliacao.npz", indices=hold, classes=classes, probas=p)
        salvar(out / "fim.json", {"holdout_labels_avaliados": False, "candidatos": len(configs), "linhas_holdout_previstas": len(hold), "features_sha256": sha256(SAIDA / "features.npy")})


def selecionar_blend():
    """Congela peso usando somente OOF dev; não acessa targets holdout."""
    from sklearn.metrics import balanced_accuracy_score

    out = SAIDA / "blend"
    out.mkdir(exist_ok=True)
    congelado = out / "selecao_congelada.json"
    if congelado.exists():
        print("Blend já congelado; seleção preservada.", flush=True)
        return
    cv = SAIDA / "cv"
    fusao = RAIZ / "runs/experimentos_fusao"
    selecionado = json.loads((cv / "selecao_congelada.json").read_text())["selecionado"]
    nome = selecionado["nome"]
    split = np.load(RAIZ / "runs/experimentos_estilo/split.npz", allow_pickle=False)
    dev, hold, classes = split["desenvolvimento"], split["holdout"], split["classes"]
    z4 = np.load(cv / "inner_oof.npz", allow_pickle=False)
    z5 = np.load(fusao / "inner_oof.npz", allow_pickle=False)
    for z in (z4, z5):
        assert np.array_equal(z["indices"], dev)
        assert np.array_equal(z["classes"], classes)
    p4, p5 = z4[nome].astype(np.float64), z5["mean5"].astype(np.float64)
    for p in (p4, p5):
        assert p.shape == (len(dev), len(classes)) and np.isfinite(p).all()
        assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    df = pd.read_csv(RAIZ / "data/train.csv", usecols=["id", "prompt_id", "model"], keep_default_na=False)
    assert set(df.prompt_id.iloc[dev]).isdisjoint(df.prompt_id.iloc[hold])
    # Somente esta fatia de labels participa da seleção.
    y_dev = df.model.iloc[dev].to_numpy()
    pesos = (0., .25, .5, .75, 1.)
    resultados = []
    for peso in pesos:
        p = (1 - peso) * p5 + peso * p4
        resultados.append({"peso_lr4b": peso, "balanced_accuracy_dev": float(
            balanced_accuracy_score(y_dev, classes[p.argmax(1)]))})
    melhor = max(resultados, key=lambda r: r["balanced_accuracy_dev"])
    assert abs(resultados[-1]["balanced_accuracy_dev"] - selecionado["balanced_accuracy_dev"]) < 1e-12
    np.savez_compressed(out / "inner_oof.npz", indices=dev, classes=classes,
                        lr4b=p4, mean5=p5,
                        selecionado=(1 - melhor["peso_lr4b"]) * p5 + melhor["peso_lr4b"] * p4)
    salvar(congelado, {
        "congelado_em": agora(), "lr4b": selecionado, "selecionado": melhor,
        "candidatos": resultados, "formula": "(1-peso_lr4b)*mean5 + peso_lr4b*lr4b",
        "criterio": "pooled inner OOF balanced accuracy; empate prefere menor peso_lr4b",
        "labels_holdout_avaliados": False, "historico":
        "30 prompts reservados já expostos por campanhas históricas; esta escolha usa apenas os 120 prompts dev. Camada/C também selecionados no mesmo dev, sem nesting adicional do peso.",
        "bootstrap_planejado": {"unidade": "prompt_id", "n": 1500, "seed": 20261103,
                                  "intervalo_percentil": [2.5, 97.5]},
        "fontes_sha256": {str(p.relative_to(RAIZ)): sha256(p) for p in (
            cv / "selecao_congelada.json", cv / "inner_oof.npz",
            cv / "probas_holdout_sem_avaliacao.npz", fusao / "inner_oof.npz",
            fusao / "holdout_predictions.npz", RAIZ / "data/train.csv")},
        "ids_dev_sha256": digest(df.id.iloc[dev]), "ids_holdout_sha256": digest(df.id.iloc[hold]),
    })
    print(json.dumps({"selecionado": melhor, "candidatos": resultados}, ensure_ascii=False), flush=True)


def avaliar_blend():
    """Avaliação única após congelamento do candidato, com bootstrap pareado."""
    from sklearn.metrics import balanced_accuracy_score, log_loss, recall_score

    out = SAIDA / "blend"
    resultado = out / "resultados_holdout.json"
    if resultado.exists():
        print("Holdout já avaliado; resultado preservado.", flush=True)
        return
    congelado = out / "selecao_congelada.json"
    s = json.loads(congelado.read_text())
    for p, esperado in s["fontes_sha256"].items():
        assert sha256(RAIZ / p) == esperado, f"Fonte alterada após congelamento: {p}"
    split = np.load(RAIZ / "runs/experimentos_estilo/split.npz", allow_pickle=False)
    dev, hold, classes = split["desenvolvimento"], split["holdout"], split["classes"]
    z4 = np.load(SAIDA / "cv/probas_holdout_sem_avaliacao.npz", allow_pickle=False)
    z5 = np.load(RAIZ / "runs/experimentos_fusao/holdout_predictions.npz", allow_pickle=False)
    for z in (z4, z5):
        assert np.array_equal(z["indices"], hold) and np.array_equal(z["classes"], classes)
    p4, p5 = z4["probas"].astype(np.float64), z5["baseline_mean5"].astype(np.float64)
    peso = s["selecionado"]["peso_lr4b"]
    probas = {"mean5": p5, "lr4b_solo": p4, "blend_selecionado": (1 - peso) * p5 + peso * p4}
    for p in probas.values():
        assert p.shape == (len(hold), len(classes)) and np.isfinite(p).all()
        assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    # Primeira avaliação dos labels reservados nesta campanha LR4B.
    df = pd.read_csv(RAIZ / "data/train.csv", usecols=["id", "prompt_id", "model"], keep_default_na=False)
    assert digest(df.id.iloc[hold]) == s["ids_holdout_sha256"]
    assert set(df.prompt_id.iloc[dev]).isdisjoint(df.prompt_id.iloc[hold])
    y, grupos = df.model.iloc[hold].to_numpy(), df.prompt_id.iloc[hold].to_numpy()
    prompts = np.unique(grupos)
    n = np.asarray([[(y[grupos == g] == c).sum() for c in classes] for g in prompts])
    plano = s["bootstrap_planejado"]
    draws = np.random.default_rng(plano["seed"]).integers(0, len(prompts), size=(plano["n"], len(prompts)))
    denominador = n[draws].sum(1)
    assert (denominador > 0).all()
    resultados, amostras = {}, {}
    for nome, p in probas.items():
        pred = classes[p.argmax(1)]
        corretos = np.asarray([[((y == c) & (pred == y) & (grupos == g)).sum()
                                for c in classes] for g in prompts])
        amostras[nome] = (corretos[draws].sum(1) / denominador).mean(1)
        resultados[nome] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "balanced_accuracy_ci95_prompt": np.percentile(amostras[nome], [2.5, 97.5]).tolist(),
            "log_loss": float(log_loss(y, p, labels=classes)),
            "recall_por_classe": dict(zip(classes.tolist(), recall_score(y, pred, labels=classes, average=None).tolist())),
        }
    for nome in ("lr4b_solo", "blend_selecionado"):
        resultados[nome]["delta_vs_mean5"] = resultados[nome]["balanced_accuracy"] - resultados["mean5"]["balanced_accuracy"]
        resultados[nome]["delta_vs_mean5_ci95_prompt"] = np.percentile(amostras[nome] - amostras["mean5"], [2.5, 97.5]).tolist()
    np.savez_compressed(out / "probas_holdout.npz", indices=hold, classes=classes,
                        ids=df.id.iloc[hold].to_numpy(dtype=str), prompts=grupos.astype(str), **probas)
    salvar(resultado, {"avaliado_em": agora(), "selecao_sha256": sha256(congelado),
                       "peso_congelado_lr4b": peso, "linhas": len(hold), "prompts": len(prompts),
                       "resultados": resultados, "bootstrap": plano,
                       "historico": s["historico"], "promover_automaticamente": False})
    print(json.dumps({nome: {k: v for k, v in r.items() if k != "recall_por_classe"}
                      for nome, r in resultados.items()}, ensure_ascii=False), flush=True)


def main():
    global MODELO, SAIDA
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["selftest", "budget", "extract", "cv", "blend_select", "blend_eval"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--gpu-liberada", action="store_true")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--hash-pesos", action="store_true")
    parser.add_argument("--model-path", "--modelo-path", type=Path, default=MODELO)
    parser.add_argument("--out", "--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    MODELO = args.model_path.resolve()
    destino = args.out.resolve()
    assert destino.is_relative_to((RAIZ / "runs/experimentos_qwen4b").resolve()), "Saída precisa ficar no diretório de experimentos Qwen4B."
    SAIDA = destino
    SAIDA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.modo == "selftest":
        selftest()
    elif args.modo == "budget":
        print(json.dumps(orcamento(args.hash_pesos), ensure_ascii=False, indent=2), flush=True)
    elif args.modo == "extract":
        assert args.pilot >= 0
        extrair(args)
    elif args.modo == "cv":
        cv()
    elif args.modo == "blend_select":
        selecionar_blend()
    else:
        avaliar_blend()


if __name__ == "__main__":
    main()
