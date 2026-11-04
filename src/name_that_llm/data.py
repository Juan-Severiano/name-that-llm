"""Leitura dos dados da competição."""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def carregar_treino() -> pd.DataFrame:
    """Colunas: id, prompt_id, text, model. Cada prompt tem uma resposta por modelo."""
    return pd.read_csv(DATA / "train.csv", keep_default_na=False)


def carregar_teste() -> pd.DataFrame:
    return pd.read_csv(DATA / "test.csv", keep_default_na=False)
