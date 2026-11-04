"""Constrói os artefatos de treino com o treino inteiro, em data/artefatos, prontos para subir ao Kaggle.

Uso: HF_HUB_OFFLINE=1 PYTHONPATH=src uv run python scripts/preparar_artefatos.py
"""

import json

from montar_notebook import DATASET_ARTEFATOS

from name_that_llm.artefatos import construir
from name_that_llm.data import DATA, carregar_treino

LEIA_ME = """Artefatos de treino do notebook name-that-llm-submissao: rótulos, pesos do TF-IDF (npz e json) e as camadas
escolhidas dos embeddings de cada encoder para os 2.699 textos de treino da competição Name That LLM.
Não contém os textos. Existe porque a reexecução do Kaggle troca o dataset da competição por uma versão oculta."""

construir(carregar_treino(), DATA / "modelos", DATA / "artefatos")
pasta = DATA / "artefatos"
(pasta / "dataset-metadata.json").write_text(
    json.dumps({"title": DATASET_ARTEFATOS.split("/")[1], "id": DATASET_ARTEFATOS, "licenses": [{"name": "other"}]}, indent=2) + "\n"
)
(pasta / "LEIA-ME.md").write_text(LEIA_ME + "\n")
print(f"{sum(f.stat().st_size for f in pasta.iterdir()) / 1e6:.0f} MB em {pasta}")
