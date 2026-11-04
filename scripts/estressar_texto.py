"""Ataques de formatação em um único fold interno; não consulta o holdout final."""

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from name_that_llm.pipeline import montar_tfidf


def main():
    raiz = Path(__file__).resolve().parents[1]
    protocolo = json.loads((raiz / "runs/validacao_pesada/split.json").read_text())
    df = pd.read_csv(raiz / "data/train.csv", keep_default_na=False)
    fold = protocolo["inner_folds"][0]
    treino = df.iloc[fold["treino_indices"]]
    valid = df.iloc[fold["valid_indices"]]
    assert not set(treino.prompt_id) & set(valid.prompt_id)
    inicio = time.monotonic()
    modelo = montar_tfidf(LogisticRegression(C=20, max_iter=2000)).fit(treino, treino.model)
    aspas = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "…": "..."})
    ataques = {
        "preservado": lambda t: t,
        "strip": str.strip,
        "espacos_normalizados": lambda t: re.sub(r"\s+", " ", t).strip(),
        "minusculas": str.lower,
        "aspas_tracos_ascii": lambda t: unicodedata.normalize("NFKC", t).translate(aspas),
        "sem_pontuacao": lambda t: re.sub(r"[^\w\s]", "", t),
        "primeiras_64_palavras": lambda t: " ".join(t.split()[:64]),
        "ultimas_64_palavras": lambda t: " ".join(t.split()[-64:]),
    }
    resultados = {}
    base = None
    for nome, transformar in ataques.items():
        entrada = pd.DataFrame({"text": valid.text.map(transformar)})
        p = modelo.predict_proba(entrada)
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)
        pred = modelo.classes_[p.argmax(1)]
        if base is None:
            base = pred.copy()
        resultados[nome] = {
            "balanced_accuracy": balanced_accuracy_score(valid.model, pred),
            "fracao_previsoes_alteradas": float(np.mean(pred != base)),
        }
        print(nome, resultados[nome], flush=True)
    saida = {"protocolo": "inner_fold0, sem holdout externo", "n_treino": len(treino), "n_valid": len(valid),
             "resultados": resultados, "segundos": time.monotonic() - inicio}
    (raiz / "runs/validacao_pesada/estresse_formatacao.json").write_text(json.dumps(saida, indent=2) + "\n")


if __name__ == "__main__":
    main()
