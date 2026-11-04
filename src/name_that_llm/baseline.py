"""Baseline: TF-IDF de palavras + caracteres e um classificador linear.

Validação agrupada por prompt (GroupKFold): o teste da competição usa prompts novos,
então nenhum prompt do treino pode aparecer na dobra de validação.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from name_that_llm.data import carregar_treino
from name_that_llm.pipeline import montar_tfidf as montar

RUNS = Path(__file__).resolve().parents[2] / "runs"


def validar(pipe: Pipeline, df, dobras: int = 5) -> dict:
    """Balanced accuracy por dobra, agrupando por prompt_id."""
    notas, previsto = [], np.empty(len(df), dtype=object)
    for treino, valid in GroupKFold(n_splits=dobras).split(df, df["model"], df["prompt_id"]):
        m = clone(pipe).fit(df.iloc[treino], df["model"].iloc[treino])
        previsto[valid] = m.predict(df.iloc[valid])
        notas.append(balanced_accuracy_score(df["model"].iloc[valid], previsto[valid]))
    return {"por_dobra": notas, "media": float(np.mean(notas)), "previsto": previsto}


CANDIDATOS = {
    "linearsvc_c0.5": lambda: LinearSVC(C=0.5),
    "logreg_c20": lambda: LogisticRegression(C=20, max_iter=2000),
}


def main() -> None:
    df = carregar_treino()
    resultado = {}
    for nome, fabrica in CANDIDATOS.items():
        t0 = time.time()
        r = validar(montar(fabrica()), df)
        resultado[nome] = {"media": r["media"], "por_dobra": r["por_dobra"], "segundos": time.time() - t0}
        print(f"{nome}: balanced acc {r['media']:.4f}  dobras {np.round(r['por_dobra'], 3)}  {time.time() - t0:.0f}s", flush=True)
    RUNS.mkdir(exist_ok=True)
    (RUNS / "baseline.json").write_text(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    sys.exit(main())
