"""Embeddings de encoders públicos: média dos tokens em cada camada, guardada em cache.

Uso: PYTHONPATH=src uv run python -m name_that_llm.embeddings BAAI/bge-small-en-v1.5
"""

import sys
import time

import numpy as np

from name_that_llm.data import DATA, carregar_teste, carregar_treino
from name_that_llm.pipeline import extrair

CACHE = DATA / "emb"


def caminho(modelo: str, conjunto: str):
    return CACHE / f"{modelo.replace('/', '__')}.{conjunto}.npy"


def main(modelo: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for conjunto, df in (("treino", carregar_treino()), ("teste", carregar_teste())):
        t0 = time.time()
        e = extrair(modelo, df["text"].tolist())
        np.save(caminho(modelo, conjunto), e)
        print(f"{modelo} {conjunto}: {e.shape} em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
