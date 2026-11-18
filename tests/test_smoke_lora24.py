"""Ensaio do smoke LoRA24 na CPU, com um Qwen3 minúsculo de pesos aleatórios (mesmo layout de checkpoint do real).

Confere a mecânica (truncar camadas, LoRA, gradient checkpointing, leitura em várias profundidades, laço de
treino, resumo em JSON), não a velocidade nem a acurácia. Pula se faltar o tokenizador local ou o peft.
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


def modelo_minusculo(destino: Path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    Qwen3ForCausalLM(cfg).save_pretrained(destino)


def pool_falso(destino: Path):
    rng = np.random.default_rng(0)
    palavras = "a casa o gato sol rio mar luz vento pedra livro tempo chuva fogo terra ar".split()
    linhas = []
    for classe in range(18):
        for _ in range(8):
            n = int(rng.integers(30, 120))
            linhas.append({"text": " ".join(rng.choice(palavras, n)), "model": f"modelo-{classe:02d}", "n_words": n})
    pd.DataFrame(linhas).to_parquet(destino)


@pytest.mark.skipif(not TOKENIZADOR.exists(), reason="tokenizador local do Qwen3-0.6B ausente")
def test_smoke_roda_na_cpu_com_modelo_minusculo(tmp_path):
    pytest.importorskip("peft")
    base = tmp_path / "base"
    modelo_minusculo(base)
    pool = tmp_path / "pool.parquet"
    pool_falso(pool)
    cfg = {"base": str(base), "tokenizer": str(TOKENIZADOR), "pool": str(pool), "keep": 4, "layers": [1, 2, 3, 4],
           "micro_lotes": [2, 4], "passos": 3, "aquecimento": 1, "n_textos": 90, "tempo_maximo_s": 300,
           "saida": str(tmp_path), "textos_projecao": [1000]}
    r = subprocess.run([sys.executable, str(RAIZ / "scripts/smoke_lora24.py")], capture_output=True, text=True,
                       env={"SMOKE_CFG": json.dumps(cfg), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    resumo = json.loads((tmp_path / "smoke_resultado.json").read_text())
    assert resumo["codigos"] == [0]
    melhor = resumo["gpus"][0]["melhor"]
    assert melhor and melhor["tokens_reais_por_s"] > 0 and not melhor["oom"]
    assert melhor["perda_nao_finita"] == 0 and melhor["passos_medidos"] == 3
    assert resumo["projecao_horas_1_epoca"]["1000"] > 0
    assert len(resumo["gpus"][0]["medidas"]) == 2  # os dois tamanhos de micro-lote
