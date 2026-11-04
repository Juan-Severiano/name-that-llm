"""Ensaio local do treino LoRA: poucos textos, modelo pequeno, tempo curto. Só confere que o código roda de ponta a ponta.
Uso: HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts uv run python scripts/ensaio_treino_lora.py"""

import tempfile

import treino_lora as t
from name_that_llm.data import DATA

with tempfile.TemporaryDirectory() as tmp:
    t.CFG.update({"modelo": str(DATA / "modelos" / "Qwen3-0.6B"), "max_textos_pool": 96, "epocas_pool": 1, "epocas_oficial": 1,
                  "max_len": 128, "micro_lote": 4, "acumular": 2, "tempo_maximo_s": 40, "avaliar_max_textos": 64, "salvar_cada": 5,
                  "saida": tmp, "entrada": str(DATA)})
    t.main()
    import os
    print(sorted(os.listdir(tmp)))
