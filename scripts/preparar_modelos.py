"""Copia os encoders do cache do Hugging Face para data/modelos/<nome>, prontos para subir ao Kaggle.

Só lê o cache local (sem internet). Junta os arquivos de todos os snapshots do modelo (o safetensors do
deberta-v3-base veio de um snapshot separado) e pula os .bin quando existe safetensors. Grava também o
tokenizer.json de cada modelo, para o Kaggle não precisar de sentencepiece nem de tiktoken.
"""

import json
import shutil
from pathlib import Path

from montar_notebook import DATASET
from transformers import AutoTokenizer

from name_that_llm.data import DATA
from name_that_llm.pipeline import CAMADAS

HUB = Path.home() / ".cache" / "huggingface" / "hub"

for repo in CAMADAS:
    snapshots = (HUB / ("models--" + repo.replace("/", "--")) / "snapshots").iterdir()
    arquivos = {f.name: f for s in snapshots for f in s.iterdir()}
    if "model.safetensors" in arquivos:
        arquivos.pop("pytorch_model.bin", None)
    destino = DATA / "modelos" / repo.split("/")[-1]
    destino.mkdir(parents=True, exist_ok=True)
    for nome, origem in arquivos.items():
        shutil.copyfile(origem.resolve(), destino / nome)
    if "tokenizer.json" not in arquivos:
        AutoTokenizer.from_pretrained(destino).save_pretrained(destino)
    tamanho = sum(f.stat().st_size for f in destino.iterdir()) / 1e6
    print(f"{repo} -> {destino.relative_to(DATA.parent)}: {tamanho:.0f} MB  {sorted(f.name for f in destino.iterdir())}")

LEIA_ME = """Encoders usados pelo notebook name-that-llm-submissao, copiados sem alteração dos repositórios originais
do Hugging Face (só foi gerado o tokenizer.json do deberta-v3-base): {repos}.
Licenças: veja a página de cada modelo."""

pasta = DATA / "modelos"
(pasta / "dataset-metadata.json").write_text(
    json.dumps({"title": DATASET.split("/")[1], "id": DATASET, "licenses": [{"name": "other"}]}, indent=2) + "\n"
)
(pasta / "LEIA-ME.md").write_text(LEIA_ME.format(repos=", ".join(CAMADAS)) + "\n")
