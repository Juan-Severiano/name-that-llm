"""Embeddings do pool público (subconjunto conservador, balanceado por modelo) com os mesmos encoders e camadas
do pipeline, guardados em lotes retomáveis em data/emb_pool/<encoder>/.

O conservador já exclui prompts parecidos com os 150 oficiais (limiar 0,35), proxies e duplicatas, então os
2.699 textos oficiais servem de teste. Uso: PYTHONPATH=src uv run python scripts/pool_embeddings.py <encoder|subconjunto>
  subconjunto: sorteia N textos por modelo (semente fixa) e grava data/pool_subconjunto.parquet
  encoder: bge-small-en-v1.5 | roberta-base | deberta-v3-base | Qwen3-0.6B"""

import os
import sys
import time

import numpy as np
import pandas as pd

from name_that_llm.data import DATA
from name_that_llm.pipeline import CAMADAS, extrair

CONJUNTO = os.environ.get("POOL_CONJUNTO", "subconjunto")  # "resto" = o conservador menos o subconjunto
SUBCONJUNTO = DATA / f"pool_{CONJUNTO}.parquet"
POR_MODELO = 2300
LOTE = 2000


def montar_resto() -> None:
    pool = pd.read_parquet(DATA / "open_pool/pool_conservador.parquet", columns=["id", "text", "model", "prompt_canon", "source_dataset"])
    pool = pool[pool.text.str.split().str.len().between(30, 300)]
    resto = pool[~pool.id.isin(pd.read_parquet(DATA / "pool_subconjunto.parquet").id)].sample(frac=1, random_state=2).reset_index(drop=True)
    resto.to_parquet(DATA / "pool_resto.parquet", index=False)
    print("resto:", len(resto), resto.model.value_counts().min(), "a", resto.model.value_counts().max(), "por modelo")


def montar_subconjunto() -> None:
    pool = pd.read_parquet(DATA / "open_pool/pool_conservador.parquet", columns=["id", "text", "model", "prompt_canon", "source_dataset"])
    pool = pool[pool.text.str.split().str.len().between(30, 300)]  # os oficiais têm 36 a 274 palavras
    amostra = pool.groupby("model", group_keys=False).sample(n=POR_MODELO, random_state=20261103)
    amostra = amostra.sample(frac=1, random_state=1).reset_index(drop=True)
    amostra.to_parquet(SUBCONJUNTO, index=False)
    print(amostra.model.value_counts().describe().round(0).to_dict(), "| prompts:", amostra.prompt_canon.nunique(),
          "| fontes:", amostra.source_dataset.nunique(), "| linhas:", len(amostra))


def extrair_encoder(nome: str) -> None:
    repo = next(r for r in CAMADAS if r.split("/")[-1] == nome)
    camadas = CAMADAS[repo]
    textos = pd.read_parquet(SUBCONJUNTO).text.tolist()
    pasta = DATA / ("emb_pool" if CONJUNTO == "subconjunto" else f"emb_pool_{CONJUNTO}") / nome
    pasta.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(textos), LOTE):
        arq = pasta / f"lote_{i // LOTE:03d}.npy"
        if arq.exists():
            continue
        t0 = time.time()
        E = extrair(DATA / "modelos" / nome, textos[i : i + LOTE])
        np.save(arq, np.concatenate([E[:, c] for c in camadas], axis=1))
        print(f"{nome} lote {i // LOTE + 1}/{-(-len(textos) // LOTE)} em {time.time() - t0:.0f}s", flush=True)
    np.save(pasta.parent / f"{nome}.npy", np.concatenate([np.load(pasta / f"lote_{k:03d}.npy") for k in range(-(-len(textos) // LOTE))]))
    print(f"{nome}: pronto", flush=True)


if __name__ == "__main__":
    {"subconjunto": montar_subconjunto, "resto": montar_resto}.get(sys.argv[1], lambda: extrair_encoder(sys.argv[1]))()
