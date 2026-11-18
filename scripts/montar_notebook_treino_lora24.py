"""Gera o notebook de TREINO do LoRA24 (Qwen3-8B-Base, T4 x2, internet desligada) em um de dois perfis.

  --check   40 passos em ~1.000 textos do pool: prova o NCCL nas duas T4, a escala e o fluxo inteiro (~5 min).
  --etapa1  etapa 1 completa: 34.992 textos balanceados, 1 época (~2.190 passos, ~1,5 h), avaliada nos 2.699 oficiais.
  --etapa1b continua a etapa 1 por mais uma época numa segunda amostra do pool (35 mil textos que não repetem a primeira,
            exceto os que faltarem em classes pequenas), avaliada nos mesmos 2.699 oficiais, com ganho pareado (~1,9 h).
  --etapa2  retoma a saída do notebook da etapa 1 e ajusta nos oficiais, reservando prompts inteiros (2 dobras), medindo
            as mesmas passagens antes e depois (~35 min).
  --final   retoma a etapa 1 e ajusta em todos os oficiais, sem avaliação (~16 min); gera o modelo para enviar.
  --inferencia  prevê o teste com a saída do notebook final e grava submission.parquet (a reexecução do placar usa este).

O modelo (qwen-lm/qwen-3/Transformers/8b-base/1), o pool aberto e os dados da competição entram como entradas do
Kaggle. O código é o de scripts/treino_lora24.py, embutido numa célula que o grava em disco e o executa.
Uso: PYTHONPATH=src:scripts uv run python scripts/montar_notebook_treino_lora24.py --check   (nada é enviado)
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
USUARIO = "juanseveriano"
POOL = "crtproduction/name-that-llm-open-pool"
MODELO = "qwen-lm/qwen-3/Transformers/8b-base/1"
COMPETICAO = "name-that-llm"

PERFIS = {
    "check": {"nome": "ntl-lora24-check", "datasets": [POOL], "kernels": [],
              "cfg": {"n_pool_por_classe": 60, "max_passos": 40, "avaliar_cada": 20, "avaliar_n": 144,
                      "avaliar_final_n": 360, "salvar_cada": 0, "tempo_maximo_s": 900, "perfil_passos": True},
              "titulo": "check curto (40 passos)"},
    "etapa1": {"nome": "ntl-lora24-etapa1", "cfg": {}, "titulo": "etapa 1 completa (~35 mil textos, 1 época)",
               "datasets": [POOL], "kernels": []},
    "etapa1b": {"nome": "ntl-lora24-etapa1b", "titulo": "etapa 1b (mais uma época, segunda amostra do pool)",
                "datasets": [POOL], "kernels": [f"{USUARIO}/ntl-lora24-etapa1"], "intro": "etapa1b",
                "cfg": {"retomar": True, "pool_excluir_primeira": True, "lr_nucleo": 1e-4, "lr_cabeca": 5e-4,
                        "tempo_maximo_s": 12600}},
    "etapa2": {"nome": "ntl-lora24-etapa2", "titulo": "etapa 2 (ajuste nos oficiais, prompts reservados)", "datasets": [],
               "kernels": [f"{USUARIO}/ntl-lora24-etapa1"], "intro": "etapa2",
               "cfg": {"etapa": 2, "epocas": 2, "lr_nucleo": 5e-5, "lr_cabeca": 5e-4, "dobras": [0, 1], "avaliar_cada": 0,
                       "salvar_cada": 0, "micro_lote": 8, "tempo_maximo_s": 5400}},
    "final": {"nome": "ntl-lora24-final", "titulo": "ajuste final (todos os oficiais)", "datasets": [],
              "kernels": [f"{USUARIO}/ntl-lora24-etapa1"], "intro": "final",
              "cfg": {"etapa": 2, "epocas": 2, "lr_nucleo": 5e-5, "lr_cabeca": 5e-4, "dobras": [], "final_todos": True,
                      "avaliar_cada": 0, "salvar_cada": 0, "micro_lote": 8, "tempo_maximo_s": 3600}},
    "inferencia": {"nome": "ntl-lora24-inferencia", "titulo": "inferência no teste", "datasets": [],
                   "kernels": [f"{USUARIO}/ntl-lora24-final"], "intro": "inferencia",
                   "cfg": {"etapa": 3, "sufixo_inicio": "_etapa2", "avaliar_cada": 0, "salvar_cada": 0,
                           "tempo_maximo_s": 3600}},
}

INTRO = """# Name That LLM: treino LoRA24, {titulo}

Qwen3-8B-Base nas primeiras 24 camadas, LoRA r16, leitura em 8 profundidades, cabeça linear, em duas T4 com dados
paralelos (um processo por GPU, gradientes somados por all_reduce). Treina na amostra balanceada do pool público
(`{pool}`, sem as linhas geradas pelo proxy do Kaggle depois da regra do host de 28/10) e mede nos 2.699 textos
oficiais, que nunca entram no treino. Entradas: modelo `{modelo}`, dataset `{pool}` e a competição. Internet desligada.
Código: `scripts/treino_lora24.py`.
"""

INTRO_ETAPA1B = """# Name That LLM: treino LoRA24, {titulo}

Continua a etapa 1: retoma o adaptador e as cabeças da saída do notebook `juanseveriano/ntl-lora24-etapa1` e treina mais
uma época em outra amostra balanceada do pool (`{pool}`), com as mesmas filtragens. A amostra não repete a primeira,
exceto os textos que faltarem em classes com pouco pool (reaproveitados da primeira). Taxas 1e-4 no LoRA e 5e-4 nas
cabeças, com novo aquecimento. Mede nos mesmos 2.699 oficiais, que nunca entram no treino, e compara com a etapa 1
passagem a passagem (ganho pareado com intervalo por prompt). Entradas: modelo `{modelo}`, o pool, a etapa 1 e a
competição. Internet desligada. Código: `scripts/treino_lora24.py`.
"""

INTRO_ETAPA2 = """# Name That LLM: treino LoRA24, {titulo}

Retoma o adaptador e as cabeças da etapa 1 (saída do notebook `juanseveriano/ntl-lora24-etapa1`, que entra como
entrada) e ajusta por 2 épocas nos textos oficiais (taxas 5e-5 no LoRA e 5e-4 nas cabeças). Reserva prompts inteiros:
em cada dobra, 30 dos 150 prompts ficam de fora do ajuste e as mesmas passagens são medidas antes e depois. Antes de
treinar, confere que a retomada reproduz os logits salvos da etapa 1. Entradas: modelo `{modelo}`, a etapa 1 e a
competição. Internet desligada. Código: `scripts/treino_lora24.py`.
"""

INTRO_FINAL = """# Name That LLM: treino LoRA24, {titulo}

Retoma o adaptador e as cabeças da etapa 1 (saída do notebook `juanseveriano/ntl-lora24-etapa1`) e ajusta por 2 épocas
em TODOS os 2.699 textos oficiais, com as mesmas taxas da etapa 2 (5e-5 no LoRA e 5e-4 nas cabeças). Não há como avaliar
este modelo: o ganho foi medido antes, com prompts reservados, no notebook da etapa 2. Salva `adaptador_etapa2/` e
`cabecas_etapa2.pt` (com a lista de classes) para o notebook de inferência. Entradas: modelo `{modelo}`, a etapa 1 e a
competição. Internet desligada. Código: `scripts/treino_lora24.py`.
"""

INTRO_INFERENCIA = """# Name That LLM: LoRA24, {titulo}

Prevê o teste com o modelo final (saída do notebook `juanseveriano/ntl-lora24-final`): Qwen3-8B-Base nas 24 primeiras
camadas com o adaptador da etapa 1 ajustado nos oficiais, em duas T4, uma fatia dos textos por GPU. Lê as classes dos
pesos salvos e não usa o `train.csv`, porque a reexecução do Kaggle troca a competição por uma versão oculta. Grava
`submission.parquet` (id, model) pela classe de maior logit, sem balanceamento. Entradas: modelo `{modelo}`, o ajuste
final e a competição. Internet desligada. Código: `scripts/treino_lora24.py` (etapa 3).
"""

INTROS = {"padrao": INTRO, "etapa1b": INTRO_ETAPA1B, "etapa2": INTRO_ETAPA2, "final": INTRO_FINAL, "inferencia": INTRO_INFERENCIA}

CELULA = """import json
import subprocess
import sys
from pathlib import Path

SCRIPT = {script!r}
caminho = Path("/kaggle/working/treino_lora24.py")
caminho.parent.mkdir(parents=True, exist_ok=True)
caminho.write_text(SCRIPT)
processo = subprocess.Popen([sys.executable, str(caminho)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for linha in processo.stdout:
    print(linha, end="")
codigo = processo.wait()
print("código de saída:", codigo)
resultado = Path("/kaggle/working/resultado.json")
if resultado.exists():
    res = json.loads(resultado.read_text())
    print(json.dumps({{k: v for k, v in res.items() if k not in ("config", "historico_perda")}}, indent=2)[:6000])
envio = Path("/kaggle/working/submission.parquet")
if envio.exists():
    import pandas as pd

    sub = pd.read_parquet(envio)
    print("submission.parquet:", sub.shape, list(sub.columns), sub.model.value_counts().to_dict())
assert codigo == 0
"""


def celula(tipo, fonte):
    base = {"cell_type": tipo, "metadata": {}, "source": fonte.splitlines(keepends=True)}
    return base | ({"execution_count": None, "outputs": []} if tipo == "code" else {})


def notebook(perfil: str) -> dict:
    codigo = (RAIZ / "scripts/treino_lora24.py").read_text()
    assert codigo.count('CFG_JSON = "{}"') == 1
    codigo = codigo.replace('CFG_JSON = "{}"', f"CFG_JSON = {json.dumps(json.dumps(PERFIS[perfil]['cfg']))}")
    intro = INTROS[PERFIS[perfil].get("intro", "padrao")].format(titulo=PERFIS[perfil]["titulo"], pool=POOL, modelo=MODELO)
    return {"cells": [celula("markdown", intro), celula("code", CELULA.format(script=codigo))],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 4}


def metadata(perfil: str) -> dict:
    nome = PERFIS[perfil]["nome"]
    return {"id": f"{USUARIO}/{nome}", "title": nome, "code_file": f"{nome}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "false", "dataset_sources": PERFIS[perfil]["datasets"],
            "competition_sources": [COMPETICAO], "kernel_sources": PERFIS[perfil]["kernels"], "model_sources": [MODELO],
            "machine_shape": "NvidiaTeslaT4"}


if __name__ == "__main__":
    perfil = next((p for p in PERFIS if f"--{p}" in sys.argv), None)
    if perfil is None:
        sys.exit("use --check, --etapa1, --etapa1b, --etapa2, --final ou --inferencia")
    nome = PERFIS[perfil]["nome"]
    saida = RAIZ / f"notebook_lora24_{perfil}"
    saida.mkdir(exist_ok=True)
    (saida / f"{nome}.ipynb").write_text(json.dumps(notebook(perfil), ensure_ascii=False, indent=1) + "\n")
    (saida / "kernel-metadata.json").write_text(json.dumps(metadata(perfil), indent=2) + "\n")
    print(f"escrito {saida} (perfil {perfil}); nada enviado")
