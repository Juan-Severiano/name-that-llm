"""Valida embeddings (uma camada por vez) com regressão logística, agrupando por prompt."""

import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from name_that_llm.data import carregar_treino
from name_that_llm.embeddings import caminho
from name_that_llm.pipeline import lr_embeddings as lr


def probabilidades_oof(X: np.ndarray, df, classificador, dobras: int = 5) -> np.ndarray:
    """Probabilidades fora da dobra, na ordem das classes ordenadas."""
    p = np.zeros((len(df), df["model"].nunique()))
    for treino, valid in GroupKFold(n_splits=dobras).split(df, df["model"], df["prompt_id"]):
        parte = X.iloc if hasattr(X, "iloc") else X  # DataFrame (texto) ou matriz (embeddings)
        m = classificador().fit(parte[treino], df["model"].iloc[treino])
        p[valid] = m.predict_proba(parte[valid])
    return p


def nota(p: np.ndarray, df) -> float:
    classes = np.sort(df["model"].unique())
    return balanced_accuracy_score(df["model"], classes[p.argmax(axis=1)])


def main(modelo: str) -> None:
    df = carregar_treino()
    E = np.load(caminho(modelo, "treino")).astype(np.float32)
    for camada in range(0, E.shape[1], 2):
        print(f"{modelo} camada {camada:2d}: {nota(probabilidades_oof(E[:, camada], df, lr), df):.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
