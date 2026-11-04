"""Constrói os artefatos de treino que o notebook lê em vez do train.csv.

O Kaggle roda o notebook com o dataset da competição trocado por uma versão oculta, sem garantia de que o
train.csv exista. Os artefatos ficam num dataset privado: rótulos, pesos do TF-IDF e as camadas escolhidas
dos embeddings de cada encoder (não os textos).

Uso: HF_HUB_OFFLINE=1 PYTHONPATH=src uv run python scripts/preparar_artefatos.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from name_that_llm.pipeline import CAMADAS, extrair, montar_tfidf


def exportar_tfidf(pipe, destino: Path) -> None:
    """Vocabulário (na ordem das colunas), idf e pesos da regressão logística, sem pickle do sklearn."""
    palavras, caracteres = (pipe.named_steps["atributos"].named_transformers_[n] for n in ("palavras", "caracteres"))
    vocabulario = {n: sorted(v.vocabulary_, key=v.vocabulary_.get) for n, v in (("palavras", palavras), ("caracteres", caracteres))}
    (destino / "tfidf_vocab.json").write_text(json.dumps(vocabulario, ensure_ascii=False))
    clf = pipe.named_steps["clf"]
    np.savez_compressed(
        destino / "tfidf.npz",
        idf_palavras=palavras.idf_,
        idf_caracteres=caracteres.idf_,
        coef=clf.coef_,
        intercepto=clf.intercept_,
    )


def construir(treino, pasta_modelos: Path, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    y = treino["model"].to_numpy()
    (destino / "rotulos.json").write_text(json.dumps(y.tolist()))
    exportar_tfidf(montar_tfidf(LogisticRegression(C=20, max_iter=2000)).fit(treino, y), destino)
    for modelo, camadas in CAMADAS.items():
        nome = modelo.split("/")[-1]
        E = extrair(pasta_modelos / nome, treino["text"].tolist())
        np.save(destino / f"emb_{nome}.npy", np.concatenate([E[:, c] for c in camadas], axis=1))
        print(f"{nome}: pronto", flush=True)

