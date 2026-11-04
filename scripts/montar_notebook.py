"""Gera notebook/name-that-llm-submissao.ipynb a partir de src/name_that_llm/pipeline.py.

O notebook do Kaggle roda sem internet e não tem o pacote name_that_llm: o código do pipeline entra
inteiro numa célula, então há uma fonte só. Também escreve o kernel-metadata.json do `kaggle kernels push`.

Uso: PYTHONPATH=src uv run python scripts/montar_notebook.py
"""

import ast
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
USUARIO = "juanseveriano"
NOME = "name-that-llm-submissao"
DATASET = f"{USUARIO}/name-that-llm-encoders"
DATASET_ARTEFATOS = f"{USUARIO}/name-that-llm-artefatos"

INTRO = """# Name That LLM: envio

Qual dos 18 LLMs escreveu cada texto. Média das probabilidades de cinco componentes: TF-IDF (palavras e
caracteres) e quatro encoders (bge-small, roberta-base, deberta-v3-base e Qwen3-0.6B), de cada um a média dos
tokens em várias camadas. Sem internet: os encoders vêm do dataset `{encoders}` e o treino, já processado
(rótulos, TF-IDF ajustado e embeddings do treino), do dataset `{artefatos}`, porque o Kaggle reexecuta o notebook
com o dataset da competição trocado por uma versão oculta. Cada componente que falhar é pulado e o
`submission.parquet` é regravado a cada componente pronto. O código da célula seguinte é
`src/name_that_llm/pipeline.py`, copiado pelo `scripts/montar_notebook.py`.
"""

EXECUCAO = '''import sklearn
import transformers

ENTRADA = Path("/kaggle/input")
SEM_CONFIANCA = "gpt-5.5"  # um dos 18 rótulos: só entra se nem os rótulos do treino puderem ser lidos


def achar(sufixo: str) -> Path | None:
    return next((p for p in ENTRADA.rglob("*") if p.as_posix().endswith(sufixo)), None)


def salvar(sub: pd.DataFrame) -> None:
    assert sub["id"].is_unique and sub["model"].notna().all()
    sub.to_parquet("submission.parquet", index=False)


print("sklearn", sklearn.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available())
print("arquivos em /kaggle/input:", sorted({p.name for p in ENTRADA.rglob("*") if p.is_file()})[:40])

teste = pd.read_csv(achar("test.csv"), keep_default_na=False)
print("test.csv:", teste.shape, "| train.csv na execução:", achar("train.csv") is not None)

salvar(pd.DataFrame({"id": teste["id"], "model": SEM_CONFIANCA}))  # valor de partida: sempre há um arquivo válido
try:
    artefatos = achar("rotulos.json").parent
    pasta_modelos = next(
        (p.parent.parent for m in CAMADAS if (p := achar(f"{m.split('/')[-1]}/config.json")) is not None),
        ENTRADA,
    )
    previsao = prever(teste, pasta_modelos, artefatos, salvar)
    salvar(previsao)
except Exception:
    traceback.print_exc()  # o notebook termina normalmente: vale o último arquivo gravado
print(pd.read_parquet("submission.parquet")["model"].value_counts())
'''


def codigo_do_pipeline() -> str:
    """O pipeline.py sem a docstring do módulo."""
    texto = (RAIZ / "src" / "name_that_llm" / "pipeline.py").read_text()
    modulo = ast.parse(texto)
    return "\n".join(texto.splitlines()[modulo.body[0].end_lineno :]).strip() + "\n"


def celula(tipo: str, fonte: str) -> dict:
    base = {"cell_type": tipo, "metadata": {}, "source": fonte.splitlines(keepends=True)}
    return base | ({"execution_count": None, "outputs": []} if tipo == "code" else {})


def notebook() -> dict:
    return {
        "cells": [
            celula("markdown", INTRO.format(encoders=DATASET, artefatos=DATASET_ARTEFATOS)),
            celula("code", codigo_do_pipeline()),
            celula("code", EXECUCAO),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def metadata() -> dict:
    return {
        "id": f"{USUARIO}/{NOME}",
        "title": NOME,
        "code_file": f"{NOME}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [DATASET, DATASET_ARTEFATOS],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [],
        "model_sources": [],
    }


if __name__ == "__main__":
    pasta = RAIZ / "notebook"
    pasta.mkdir(exist_ok=True)
    (pasta / f"{NOME}.ipynb").write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n")
    (pasta / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print(f"escrito em {pasta.relative_to(RAIZ)}/")
