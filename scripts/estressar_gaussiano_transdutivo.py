"""Verifica contrato transdutivo, sem medir acurácia nem selecionar hiperparâmetros.

Usa features existentes, sem GPU. Trocar a ordem das linhas deve preservar o
resultado após alinhar os IDs. Partir o conjunto em blocos é um diagnóstico:
remove contexto entre respostas e NÃO equivale à inferência do lote completo.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from name_that_llm import gaussiano as g
from preparar_gaussiano import cache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gaussiano_candidato"


def subset(matrices, indices):
    return g.Matrizes(matrices.ids[indices], {k: v[indices] for k, v in matrices.vistas.items()}, matrices.semantica[indices])


def main():
    started = time.monotonic()
    df = pd.read_csv(ROOT / "data/train.csv", keep_default_na=False)[["id", "text"]]
    matrices = cache(df)
    model = g.carregar(OUT / "artefatos")
    # Mesmo conjunto do holdout para conferir contratos, mas modelo full-fit.
    # Não reportar acurácia desta execução: há exposição de treino por desenho.
    split = json.loads((ROOT / "runs/validacao_pesada/split.json").read_text())
    indices = np.asarray(split["holdout_indices"])
    df = df.iloc[indices].reset_index(drop=True)
    matrices = subset(matrices, indices)
    reference = g.prever(model, df, matrices)
    permutations = []
    for seed in (11, 29, 53, 101, 211):
        order = np.random.default_rng(seed).permutation(len(df))
        inverse = np.argsort(order)
        result = g.prever(model, df.iloc[order].reset_index(drop=True), subset(matrices, order))
        aligned = result.probabilidades[inverse]
        delta = float(np.max(np.abs(reference.probabilidades - aligned)))
        same_labels = bool(np.array_equal(aligned.argmax(axis=1), reference.probabilidades.argmax(axis=1)))
        ari = adjusted_rand_score(reference.grupos, result.grupos[inverse])
        assert same_labels and ari == 1.0 and delta < 1e-5, (seed, delta, ari)
        permutations.append({"seed": seed, "maxdiff": delta, "same_predictions": same_labels, "cluster_ari": ari})
    chunk_checks = []
    for size in (1, 16, 64):
        probabilities = []
        for start in range(0, len(df), size):
            rows = np.arange(start, min(start + size, len(df)))
            probabilities.append(g.prever(model, df.iloc[rows].reset_index(drop=True), subset(matrices, rows)).probabilidades)
        probabilities = np.concatenate(probabilities)
        chunk_checks.append({"chunk_rows": size,
                             "fraction_predictions_changed": float(np.mean(probabilities.argmax(axis=1) != reference.probabilidades.argmax(axis=1))),
                             "max_probability_difference": float(np.max(np.abs(probabilities - reference.probabilidades)))})
    report = {"rows": len(df), "seconds": time.monotonic() - started,
              "permutations": permutations, "chunking_diagnostic": chunk_checks,
              "accuracy_measured": False, "gpu": False,
              "contract": "Encode in batches; infer groups and probabilities on the complete test set. Chunked classification is not equivalent."}
    (OUT / "contrato_transdutivo.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
