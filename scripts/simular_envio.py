"""Ensaio do envio: separa 30 prompts do treino como "teste oculto", constrói os artefatos só com os outros
120, roda prever() sem internet e confere o formato contra sample_submission.parquet.

Uso: HF_HUB_OFFLINE=1 PYTHONPATH=src uv run python scripts/simular_envio.py
"""

import json
import os
import time

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from name_that_llm import artefatos
from name_that_llm.baseline import RUNS
from name_that_llm.data import DATA, carregar_treino
from name_that_llm.pipeline import prever

assert os.environ.get("HF_HUB_OFFLINE") == "1", "rode com HF_HUB_OFFLINE=1 para simular o Kaggle sem internet"

df = carregar_treino()
prompts = sorted(df["prompt_id"].unique())
oculto = df["prompt_id"].isin(prompts[::5])  # 30 dos 150 prompts, espalhados
treino, teste = df[~oculto], df[oculto]

t0 = time.time()
artefatos.construir(treino, DATA / "modelos", DATA / "artefatos_simulacao")
segundos_treino = time.time() - t0
t0 = time.time()
sub = prever(teste[["id", "text"]], DATA / "modelos", DATA / "artefatos_simulacao")
segundos = time.time() - t0

modelo_exemplo = pd.read_parquet(DATA / "sample_submission.parquet")
assert list(sub.columns) == list(modelo_exemplo.columns), (list(sub.columns), list(modelo_exemplo.columns))
assert sub["id"].is_unique and set(sub["id"]) == set(teste["id"])
assert set(sub["model"]) <= set(df["model"])
print("dtypes envio:", dict(sub.dtypes.astype(str)), "| exemplo:", dict(modelo_exemplo.dtypes.astype(str)))

nota = balanced_accuracy_score(teste.set_index("id").loc[sub["id"], "model"], sub["model"])
print(f"balanced accuracy no teste simulado ({len(teste)} textos, artefatos {segundos_treino:.0f}s, previsão {segundos:.0f}s): {nota:.4f}")
RUNS.mkdir(exist_ok=True)
(RUNS / "simulacao_envio.json").write_text(json.dumps({"balanced_accuracy": nota, "textos": len(teste), "segundos_previsao": segundos, "segundos_artefatos": segundos_treino}, indent=2))
