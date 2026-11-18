"""Ensaio do treino LoRA24 em 1 e 2 processos na CPU (gloo), com um Qwen3 minúsculo de pesos aleatórios.

Confere a mecânica: filtro do pool sem vazamento dos oficiais, laço de treino, soma de gradientes entre processos
(os parâmetros têm de terminar idênticos), avaliação distribuída, salvamento e resumo em JSON. Não mede acurácia.
Pula se faltar o tokenizador local ou o peft.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
TOKENIZADOR = RAIZ / "data/modelos/Qwen3-0.6B"
PALAVRAS = "a casa o gato sol rio mar luz vento pedra livro tempo chuva fogo terra ar".split()
CLASSES = [f"modelo-{c:02d}" for c in range(18)]


def modelo_minusculo(destino: Path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    Qwen3ForCausalLM(cfg).save_pretrained(destino)


def textos(rng, n):
    return " ".join(rng.choice(PALAVRAS, n))


def dados_falsos(pasta: Path):
    rng = np.random.default_rng(0)
    pool = []
    for classe in CLASSES:
        for k in range(20):
            n = int(rng.integers(30, 120))
            pool.append({"text": textos(rng, n), "model": classe, "n_words": n, "flag_kb_proxy_after_ruling": k == 0})
    pd.DataFrame(pool).to_parquet(pasta / "pool.parquet")
    oficial = [{"id": f"id{i}", "prompt_id": f"p{i // 18}", "text": textos(rng, 40) + f" oficial{i}", "model": CLASSES[i % 18]}
               for i in range(54)]
    pd.DataFrame(oficial).to_csv(pasta / "train.csv", index=False)


@pytest.mark.skipif(not TOKENIZADOR.exists(), reason="tokenizador local do Qwen3-0.6B ausente")
@pytest.mark.parametrize("processos", [1, 2])
def test_treino_na_cpu_com_modelo_minusculo(tmp_path, processos):
    pytest.importorskip("peft")
    modelo_minusculo(tmp_path / "base")
    dados_falsos(tmp_path)
    saida = tmp_path / "saida"
    cfg = {"base": str(tmp_path / "base"), "tokenizer": str(TOKENIZADOR), "pool": str(tmp_path / "pool.parquet"),
           "oficial": str(tmp_path / "train.csv"), "keep": 4, "layers": [1, 2, 3, 4], "n_pool_por_classe": 8,
           "micro_lote": 2, "max_passos": 6, "aquecimento_frac": 0.5, "avaliar_cada": 3, "avaliar_n": 20,
           "salvar_cada": 3, "saida": str(saida), "timeout_min": 2, "processos": processos, "perfil_passos": True}
    r = subprocess.run([sys.executable, str(RAIZ / "scripts/treino_lora24.py")], capture_output=True, text=True,
                       timeout=420, env={"TREINO_CFG": json.dumps(cfg), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-3500:]
    res = json.loads((saida / "resultado.json").read_text())
    assert res["mundo"] == processos and res["passos"] == 6 and res["passos_pulados"] == 0
    assert res["diferenca_params_entre_gpus"] < 1e-9
    assert [a["passo"] for a in res["avaliacoes"]] == [3, 6]
    assert res["final"]["textos"] == 54 and 0.0 <= res["final"]["balanced_accuracy"] <= 1.0
    assert np.isfinite(res["perda_final"]) and res["tempo_avaliacao_s"] > 0
    perfil = res["perfil_passos"]  # 6 passos, os 3 primeiros fora (aquecimento)
    assert perfil["passos_medidos"] == 3 and all(perfil[k] >= 0 for k in ("fwbw_s", "espera_barreira_s", "all_reduce_e_otimizador_s"))
    assert np.load(saida / "logits_oficial.npy").shape == (54, 18)
    assert (saida / "adaptador/adapter_config.json").exists() and (saida / "cabecas.pt").exists()
    # o filtro tirou as linhas do proxy pós-regra (1 por classe) e sobrou o bastante para a amostra balanceada
    assert "pool: 360 linhas, 342 após filtros, 144 na amostra balanceada (8 a 8 por classe)" in r.stdout


def test_lotes_pareados_por_comprimento_tiram_a_espera_entre_gpus():
    sys.path.insert(0, str(RAIZ / "scripts"))
    import treino_lora24 as t

    rng = np.random.default_rng(0)
    ids = [[1] * int(n) for n in rng.integers(30, 400, 1600)]
    comprimento = lambda b: max(len(ids[i]) for i in b)  # noqa: E731

    def espera(fila):  # tempo do passo = lote mais longo do par, contra o tempo médio dos dois
        c = np.array([comprimento(b) for b in fila], dtype=float)[: len(fila) // 2 * 2].reshape(-1, 2)
        return c.max(1).mean() / c.mean() - 1

    sem = t.lotes(ids, 8, np.random.default_rng(1), 1)
    com = t.lotes(ids, 8, np.random.default_rng(1), 2)
    assert espera(sem) > 0.10 and espera(com) < 0.03
    usados = [i for b in com for i in b]  # nenhum texto repetido, todos os lotes completos
    assert len(usados) == len(set(usados)) == len(com) * 8 and all(len(b) == 8 for b in com)
    assert len(com) % 2 == 0 and abs(len(com) - len(sem)) <= 1


def rodar_script(cfg, tmp_path):
    r = subprocess.run([sys.executable, str(RAIZ / "scripts/treino_lora24.py")], capture_output=True, text=True, timeout=420,
                       env={"TREINO_CFG": json.dumps(cfg), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-3500:]
    return r


@pytest.mark.skipif(not TOKENIZADOR.exists(), reason="tokenizador local do Qwen3-0.6B ausente")
def test_etapa2_retoma_a_etapa1_e_mede_as_mesmas_passagens_antes_e_depois(tmp_path):
    pytest.importorskip("peft")
    modelo_minusculo(tmp_path / "base")
    dados_falsos(tmp_path)
    comum = {"base": str(tmp_path / "base"), "tokenizer": str(TOKENIZADOR), "oficial": str(tmp_path / "train.csv"), "keep": 4,
             "layers": [1, 2, 3, 4], "micro_lote": 2, "timeout_min": 2, "processos": 2, "avaliar_cada": 0, "salvar_cada": 0}
    etapa1 = tmp_path / "etapa1"
    rodar_script(comum | {"pool": str(tmp_path / "pool.parquet"), "n_pool_por_classe": 8, "max_passos": 4, "saida": str(etapa1)},
                 tmp_path)
    etapa2 = tmp_path / "etapa2"
    rodar_script(comum | {"etapa": 2, "inicio": str(etapa1), "logits_etapa1": str(etapa1 / "logits_oficial.npy"),
                          "dobras": [0, 1], "n_dobras": 3, "epocas": 1, "lr_nucleo": 5e-5, "lr_cabeca": 5e-4,
                          "final_todos": True, "saida": str(etapa2)}, tmp_path)
    res = json.loads((etapa2 / "resultado.json").read_text())
    assert res["retomada"]["concordancia_argmax"] == 1.0  # carregou o adaptador e as cabeças da etapa 1
    assert [d["dobra"] for d in res["dobras"]] == [0, 1]
    assert all(d["avaliacao"] == 18 and d["treino"] == 36 and d["passos_pulados"] == 0 for d in res["dobras"])
    p = res["pareado"]
    assert p["passagens"] == 36 and p["prompts"] == 2 and len(p["ic95_delta_por_prompt"]) == 2
    assert abs(p["delta"] - (p["ba_depois"] - p["ba_antes"])) < 1e-12
    assert res["final_todos"]["treino"] == 54
    assert (etapa2 / "adaptador_etapa2/adapter_config.json").exists() and (etapa2 / "cabecas_etapa2.pt").exists()


@pytest.mark.skipif(not TOKENIZADOR.exists(), reason="tokenizador local do Qwen3-0.6B ausente")
def test_inferencia_gera_o_mesmo_submission_com_1_e_2_processos_sem_ler_o_train(tmp_path):
    pytest.importorskip("peft")
    modelo_minusculo(tmp_path / "base")
    dados_falsos(tmp_path)
    comum = {"base": str(tmp_path / "base"), "tokenizer": str(TOKENIZADOR), "oficial": str(tmp_path / "train.csv"), "keep": 4,
             "layers": [1, 2, 3, 4], "micro_lote": 2, "timeout_min": 2, "processos": 2, "avaliar_cada": 0, "salvar_cada": 0}
    etapa1 = tmp_path / "etapa1"
    rodar_script(comum | {"pool": str(tmp_path / "pool.parquet"), "n_pool_por_classe": 8, "max_passos": 4, "saida": str(etapa1)},
                 tmp_path)
    final = tmp_path / "final"  # ajuste final: todos os oficiais, sem dobras
    rodar_script(comum | {"etapa": 2, "inicio": str(etapa1), "logits_etapa1": str(etapa1 / "logits_oficial.npy"), "dobras": [],
                          "epocas": 1, "final_todos": True, "saida": str(final)}, tmp_path)
    res = json.loads((final / "resultado.json").read_text())
    assert res["dobras"] == [] and "pareado" not in res and res["final_todos"]["treino"] == 54

    rng = np.random.default_rng(7)  # o texto 3 é vazio de propósito
    pd.DataFrame({"id": [f"t{i}" for i in range(7)], "text": [textos(rng, 50) if i != 3 else "" for i in range(7)]}).to_csv(
        tmp_path / "test.csv", index=False)
    saidas = {}
    for n in (1, 2):
        s = tmp_path / f"inferencia{n}"
        rodar_script(comum | {"etapa": 3, "inicio": str(final), "sufixo_inicio": "_etapa2", "teste": str(tmp_path / "test.csv"),
                              "oficial": "/nao/existe/train.csv", "processos": n, "saida": str(s)}, tmp_path)
        saidas[n] = pd.read_parquet(s / "submission.parquet")
    a, b = saidas[1], saidas[2]
    assert list(a.columns) == ["id", "model"] and list(a.id) == [f"t{i}" for i in range(7)]
    assert a.equals(b)  # a divisão entre processos não muda nenhuma previsão
    assert set(a.model) <= set(CLASSES) and a.model.map(type).eq(str).all()


@pytest.mark.skipif(not TOKENIZADOR.exists(), reason="tokenizador local do Qwen3-0.6B ausente")
def test_etapa_1b_continua_numa_segunda_amostra_sem_repetir_a_primeira(tmp_path):
    pytest.importorskip("peft")
    modelo_minusculo(tmp_path / "base")
    dados_falsos(tmp_path)  # 19 textos por classe depois do filtro do proxy
    comum = {"base": str(tmp_path / "base"), "tokenizer": str(TOKENIZADOR), "oficial": str(tmp_path / "train.csv"), "keep": 4,
             "layers": [1, 2, 3, 4], "micro_lote": 2, "timeout_min": 2, "processos": 2, "avaliar_cada": 0, "salvar_cada": 0,
             "pool": str(tmp_path / "pool.parquet")}
    etapa1 = tmp_path / "etapa1"
    rodar_script(comum | {"n_pool_por_classe": 8, "max_passos": 4, "saida": str(etapa1)}, tmp_path)
    etapa1b = tmp_path / "etapa1b"
    # 19 - 8 = 11 textos novos por classe: pedir 15 obriga a reaproveitar 4 da primeira amostra em cada classe
    r = rodar_script(comum | {"retomar": True, "inicio": str(etapa1), "logits_etapa1": str(etapa1 / "logits_oficial.npy"),
                              "pool_excluir_primeira": True, "n_pool_primeira": 8, "n_pool_por_classe": 15, "max_passos": 4,
                              "saida": str(etapa1b)}, tmp_path)
    assert "segunda amostra: 198 textos novos e 72 reaproveitados da primeira" in r.stdout  # 18 x 11 novos, 18 x 4 reaproveitados
    res = json.loads((etapa1b / "resultado.json").read_text())
    assert res["retomada"]["concordancia_argmax"] == 1.0  # carregou o adaptador e as cabeças da etapa 1
    p = res["final"]["pareado_vs_etapa1"]
    assert p["delta"] == pytest.approx(p["ba_depois"] - p["ba_antes"]) and len(p["ic95_delta_por_prompt"]) == 2
    assert res["passos"] == 4 and res["passos_pulados"] == 0 and res["diferenca_params_entre_gpus"] < 1e-9
