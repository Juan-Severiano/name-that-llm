"""Treina nos textos do pool e avalia nos 2.699 oficiais (nunca vistos no treino do pool); depois mede
pool + oficial em validação por prompt (as mesmas 5 dobras do OOF) contra só oficial.
Uso: PYTHONPATH=src uv run python scripts/avaliar_pool.py <encoder> [C ...]"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from name_that_llm.baseline import RUNS
from name_that_llm.data import DATA, carregar_treino
from name_that_llm.pipeline import lr_embeddings  # noqa: F401  (mesmo cabeçalho do pipeline de envio)

nome = sys.argv[1]
Cs = [float(c) for c in sys.argv[2:]] or [0.05, 0.5]
oficial = carregar_treino()
pool = pd.read_parquet(DATA / "pool_subconjunto.parquet")
Xp = np.load(DATA / "emb_pool" / f"{nome}.npy").astype(np.float32)
Xo = np.load(DATA / "artefatos" / f"emb_{nome}.npy").astype(np.float32)
classes = np.sort(oficial.model.unique())
assert Xp.shape[0] == len(pool) and Xo.shape[0] == len(oficial) and Xp.shape[1] == Xo.shape[1], (Xp.shape, Xo.shape)
yp, yo = pool.model.to_numpy(), oficial.model.to_numpy()


def modelo(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300))


melhor = (0.0, None)
for C in Cs:
    t0 = time.time()
    P = modelo(C).fit(Xp, yp).predict_proba(Xo)
    nota = balanced_accuracy_score(yo, classes[P.argmax(1)])
    print(f"{nome} pool->oficial C={C}: {nota:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    if nota > melhor[0]:
        melhor = (nota, C, P)
np.save(RUNS / f"pool_para_oficial_{nome}.npy", melhor[2])

C = melhor[1]
oof = np.zeros((len(oficial), 18))
for treino, valid in GroupKFold(5).split(oficial, yo, oficial.prompt_id):
    m = modelo(C).fit(np.vstack([Xp, Xo[treino]]), np.concatenate([yp, yo[treino]]))
    oof[valid] = m.predict_proba(Xo[valid])
np.save(RUNS / f"pool_mais_oficial_oof_{nome}.npy", oof)
so_oficial = np.load(RUNS / f"oof_{nome}.npy")
print(f"{nome} oficial 5 dobras SO oficial (camadas do pipeline, C=0.05): {balanced_accuracy_score(yo, classes[so_oficial.argmax(1)]):.4f}"
      f" | pool+oficial (C={C}): {balanced_accuracy_score(yo, classes[oof.argmax(1)]):.4f}", flush=True)
