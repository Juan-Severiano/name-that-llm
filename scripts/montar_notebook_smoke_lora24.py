"""Gera notebook_smoke_lora24/: notebook de GPU (T4 x2) que mede a velocidade do LoRA24 do Qwen3-8B-Base.

Internet desligada: o modelo (qwen-lm/qwen-3/Transformers/8b-base/1) e o pool aberto entram como entradas do
Kaggle. O código é o de scripts/smoke_lora24.py, embutido numa célula que o grava em disco e o executa.
Uso: PYTHONPATH=src:scripts uv run python scripts/montar_notebook_smoke_lora24.py   (nada é enviado)
"""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "notebook_smoke_lora24"
USUARIO = "juanseveriano"
NOME = "ntl-smoke-lora24"
POOL = "crtproduction/name-that-llm-open-pool"
MODELO = "qwen-lm/qwen-3/Transformers/8b-base/1"
CONFIG = {}  # os padrões do script já são o smoke (micro-lotes 4/8/16, 20 passos cada, 900 s de treino por GPU)

INTRO = """# Name That LLM: smoke de velocidade do LoRA24 (Qwen3-8B-Base, T4 x2)

Mede tokens/s, memória de pico e passos com gradiente não finito de um treino LoRA nas primeiras 24 camadas do
Qwen3-8B-Base, com a arquitetura da receita pública (LoRA r16, leitura em 8 profundidades, cabeça linear).
Um worker por GPU roda ao mesmo tempo. Não avalia acurácia; serve para decidir se vale gastar a cota de GPU
no treino de verdade. Entradas: modelo `{modelo}` e dataset `{pool}`. Internet desligada.
Código: `scripts/smoke_lora24.py`.
"""

CELULA = """import json
import subprocess
import sys
from pathlib import Path

SCRIPT = {script!r}
caminho = Path("/kaggle/working/smoke_lora24.py")
caminho.parent.mkdir(parents=True, exist_ok=True)
caminho.write_text(SCRIPT)
processo = subprocess.Popen([sys.executable, str(caminho)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for linha in processo.stdout:
    print(linha, end="")
codigo = processo.wait()
print("código de saída:", codigo)
resultado = Path("/kaggle/working/smoke_resultado.json")
if resultado.exists():
    print(json.dumps(json.loads(resultado.read_text()), indent=2)[:6000])
assert codigo == 0
"""


def celula(tipo, fonte):
    base = {"cell_type": tipo, "metadata": {}, "source": fonte.splitlines(keepends=True)}
    return base | ({"execution_count": None, "outputs": []} if tipo == "code" else {})


def notebook() -> dict:
    codigo = (RAIZ / "scripts/smoke_lora24.py").read_text()
    assert codigo.count('CFG_JSON = "{}"') == 1
    codigo = codigo.replace('CFG_JSON = "{}"', f"CFG_JSON = {json.dumps(json.dumps(CONFIG))}")
    return {"cells": [celula("markdown", INTRO.format(modelo=MODELO, pool=POOL)),
                      celula("code", CELULA.format(script=codigo))],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 4}


def metadata() -> dict:
    return {"id": f"{USUARIO}/{NOME}", "title": NOME, "code_file": f"{NOME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "false", "dataset_sources": [POOL], "competition_sources": [],
            "kernel_sources": [], "model_sources": [MODELO], "machine_shape": "NvidiaTeslaT4"}


if __name__ == "__main__":
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / f"{NOME}.ipynb").write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n")
    (SAIDA / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print(f"escrito {SAIDA}; nada enviado")
