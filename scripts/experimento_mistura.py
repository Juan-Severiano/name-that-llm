"""Mistura de log-probabilidades (média dos 5 componentes, Gaussian e LR em características centradas por grupo)
seguida da atribuição húngara. Uso: PYTHONPATH=src uv run python scripts/experimento_mistura.py"""

import itertools

import numpy as np
from sklearn.metrics import balanced_accuracy_score

exec(open("scripts/experimento_transdutivo.py").read().split("def autotreino")[0])
from name_that_llm.atribuicao import agrupar, atribuir, distancias  # noqa: E402

# LR simples centrada por grupo, fora da dobra, com grupos do teste inferidos dentro da dobra (como no teste real)
LS = np.zeros((len(df), 18))
for dobra in range(5):
    teste, treino = np.where(dobras == dobra)[0], np.where(dobras != dobra)[0]
    g_te = agrupar(distancias(df.text.iloc[teste].tolist(), semantica[teste]))
    clf = LogisticRegression(C=0.5, max_iter=300).fit(centrar(X[treino], df.prompt_id.to_numpy()[treino]), y[treino])
    LS[teste] = logp(clf, centrar(X[teste], g_te))
np.save("runs/oof_lr_centrada.npy", LS)

LB, LG = np.log(z["baseline"] + 1e-9), np.log(z["gaussian"] + 1e-9)
grupos = agrupar(distancias(df.text.tolist(), semantica))  # lote inteiro: 150 prompts, como um teste grande
oraculo = np.unique(df.prompt_id, return_inverse=True)[1]
print("argmax: base %.4f gaussian %.4f lr-centrada %.4f" % tuple(balanced_accuracy_score(y, L.argmax(1)) for L in (LB, LG, LS)))
linhas = []
for a, b in itertools.product(np.arange(0, 1.01, 0.25), repeat=2):
    c = 1 - a - b
    if c < -1e-9:
        continue
    L = a * LB + b * LG + c * LS
    linhas.append((balanced_accuracy_score(y, atribuir(L, grupos)), balanced_accuracy_score(y, atribuir(L, oraculo)),
                   balanced_accuracy_score(y, L.argmax(1)), a, b, c))
for r in sorted(linhas, reverse=True)[:8]:
    print("inferido %.4f | oraculo %.4f | argmax %.4f | pesos base %.2f gauss %.2f lr %.2f" % r)
