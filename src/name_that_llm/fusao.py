"""Fusão por média de probabilidades fora da dobra: TF-IDF + embeddings de vários encoders."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from name_that_llm.avaliar_embeddings import lr, nota, probabilidades_oof
from name_that_llm.baseline import RUNS, montar
from name_that_llm.data import carregar_treino
from name_that_llm.embeddings import caminho
from name_that_llm.pipeline import CAMADAS


def em_cache(nome: str, calcular) -> np.ndarray:
    arq = RUNS / f"oof_{nome}.npy"
    if arq.exists():
        return np.load(arq)
    p = calcular()
    np.save(arq, p)
    return p


def main() -> None:
    df = carregar_treino()
    RUNS.mkdir(exist_ok=True)
    oof = {}
    oof["tfidf"] = em_cache("tfidf", lambda: probabilidades_oof(df, df, lambda: montar(LogisticRegression(C=20, max_iter=2000))))
    for modelo, camadas in CAMADAS.items():
        E = np.load(caminho(modelo, "treino")).astype(np.float32)
        X = np.concatenate([E[:, c] for c in camadas], axis=1)
        nome = modelo.split("/")[-1]
        oof[nome] = em_cache(nome, lambda: probabilidades_oof(X, df, lr))
    for nome, p in oof.items():
        print(f"{nome:22s} {nota(p, df):.4f}", flush=True)
    todos = np.mean(list(oof.values()), axis=0)
    print(f"{'media de todos':22s} {nota(todos, df):.4f}")
    for fora in oof:
        resto = np.mean([p for n, p in oof.items() if n != fora], axis=0)
        print(f"{'sem ' + fora:22s} {nota(resto, df):.4f}")

if __name__ == "__main__":
    main()
