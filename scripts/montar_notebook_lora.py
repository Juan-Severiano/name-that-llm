"""Gera notebook_lora/ (notebook de TREINO com GPU e internet) a partir de scripts/treino_lora.py.

Separado dos notebooks de envio: a internet fica ligada só aqui, para baixar o Qwen3 e instalar o peft.
Uso: PYTHONPATH=src:scripts uv run python scripts/montar_notebook_lora.py [--smoke | --medio | --completo]"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "notebook_lora"
USUARIO = "juanseveriano"
NOME = "ntl-lora-treino"
DATASET_POOL = f"{USUARIO}/name-that-llm-pool-conservador"

PERFIS = {
    # mede velocidade e confere o fluxo: 3 mil textos do pool, 10 minutos de treino, avalia em 600 oficiais
    "smoke": {"modelo": "Qwen/Qwen3-1.7B", "max_textos_pool": 3000, "tempo_maximo_s": 600, "avaliar_max_textos": 600},
    # rodada média: 40 mil textos do pool (~1,5 h na T4) e depois os oficiais
    "medio": {"modelo": "Qwen/Qwen3-1.7B", "max_textos_pool": 40000, "epocas_pool": 1, "epocas_oficial": 2, "tempo_maximo_s": 7000,
              "salvar_cada": 300},
    # etapa 1 no pool conservador inteiro e etapa 2 nos oficiais, com tempo máximo de 7 horas
    "completo": {"modelo": "Qwen/Qwen3-1.7B", "max_textos_pool": 0, "epocas_pool": 1, "epocas_oficial": 2, "tempo_maximo_s": 25200,
                 "salvar_cada": 300},
}

INTRO = """# Name That LLM: treino LoRA

Qwen3 como classificador dos 18 modelos, com LoRA. Etapa 1: pool público conservador (dataset
`{pool}`, 115 mil textos de 32 fontes da comunidade, CC-BY/CC-BY-SA/CC0), medida nos 2.699 textos oficiais.
Etapa 2: textos oficiais, menos 30 prompts reservados. Esse notebook usa internet (baixa o modelo) e por isso
não serve para envio; salva o adaptador para o notebook de envio, que roda sem internet.
Código: `scripts/treino_lora.py`.
"""


def celula(tipo, fonte):
    base = {"cell_type": tipo, "metadata": {}, "source": fonte.splitlines(keepends=True)}
    return base | ({"execution_count": None, "outputs": []} if tipo == "code" else {})


def notebook(perfil: str) -> dict:
    codigo = (RAIZ / "scripts/treino_lora.py").read_text()
    assert codigo.count('CFG_JSON = "{}"') == 1
    codigo = codigo.replace('CFG_JSON = "{}"', f"CFG_JSON = {json.dumps(json.dumps(PERFIS[perfil]))}")
    return {"cells": [celula("markdown", INTRO.format(pool=DATASET_POOL)), celula("code", codigo)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 4}


def metadata() -> dict:
    return {"id": f"{USUARIO}/{NOME}", "title": NOME, "code_file": f"{NOME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true", "dataset_sources": [DATASET_POOL], "competition_sources": ["name-that-llm"],
            "kernel_sources": [], "model_sources": []}


if __name__ == "__main__":
    perfil = next((p for p in PERFIS if f"--{p}" in sys.argv), "smoke")
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / f"{NOME}.ipynb").write_text(json.dumps(notebook(perfil), ensure_ascii=False, indent=1) + "\n")
    (SAIDA / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print(f"escrito {SAIDA} (perfil {perfil}); nada enviado")
