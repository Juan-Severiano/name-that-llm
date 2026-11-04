"""CLI/API Kaggle com credencial local no ambiente, sem imprimi-la.

  .venv/bin/python scripts/kaggle_local.py competitions submissions name-that-llm -v
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def autenticar():
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.startswith("KAGGLE_"):
            os.environ[key] = value.strip().strip('"').strip("'")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


if __name__ == "__main__":
    autenticar()
    executable = ROOT / ".venv/bin/kaggle"
    os.execv(executable, [str(executable), *sys.argv[1:]])
