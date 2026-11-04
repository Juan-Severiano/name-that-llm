"""Autotreino com atribuição: rótulos do húngaro nos textos de teste entram no treino e o classificador é refeito.
Simula o teste com cada dobra de 30 prompts. Uso: PYTHONPATH=src uv run python scripts/experimento_transdutivo.py"""

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from name_that_llm.atribuicao import agrupar, atribuir, distancias
from name_that_llm.data import carregar_treino
from name_that_llm.embeddings import caminho
from name_that_llm.pipeline import CAMADAS

df = carregar_treino()
classes = np.sort(df.model.unique())
y = np.searchsorted(classes, df.model.to_numpy())
z = np.load("runs/gaussiano_validacao_completa/oof.npz", allow_pickle=True)
dobras = z["fold_ids"]

# características: camadas escolhidas de cada encoder, padronizadas, PCA 256 por encoder
partes = []
for modelo, camadas in CAMADAS.items():
    E = np.load(caminho(modelo, "treino")).astype(np.float32)
    X = StandardScaler().fit_transform(np.concatenate([E[:, c] for c in camadas], axis=1))
    partes.append(PCA(256, random_state=0).fit_transform(X) / np.sqrt(X.shape[1]) * 10)
X = np.concatenate(partes, axis=1)
semantica = np.load(caminho("BAAI/bge-small-en-v1.5", "treino"))[:, 12].astype(np.float32)
print("X", X.shape, flush=True)


def centrar(X, g):
    out = X.copy()
    for k in np.unique(g):
        m = g == k
        out[m] -= X[m].mean(axis=0)
    return out


def logp(clf, Xc):
    return np.log(clf.predict_proba(Xc) + 1e-9)


def autotreino(Xtr, ytr, Xte, g_te, L0, rodadas=3, pedacos=5, peso=1.0, semente=0):
    """Validação cruzada por grupo dentro do teste: o texto nunca é previsto por um modelo que viu o próprio
    pseudo-rótulo (senão o classificador só repete o que lhe foi dado)."""
    rng = np.random.default_rng(semente)
    ids = np.unique(g_te)
    pedaco = dict(zip(ids, rng.permutation(len(ids)) % pedacos))
    p_texto = np.array([pedaco[g] for g in g_te])
    L, saida = L0, []
    for _ in range(rodadas):
        pseudo = atribuir(L, g_te)
        L = np.zeros_like(L0)
        for k in range(pedacos):
            fora = p_texto != k
            clf = LogisticRegression(C=0.5, max_iter=300).fit(
                np.vstack([Xtr, Xte[fora]]), np.concatenate([ytr, pseudo[fora]]),
                sample_weight=np.concatenate([np.ones(len(ytr)), np.full(fora.sum(), peso)]))
            L[~fora] = logp(clf, Xte[~fora])
        saida.append(L.copy())
    return saida


for dobra in range(5):
    t0 = time.time()
    teste = np.where(dobras == dobra)[0]
    treino = np.where(dobras != dobra)[0]
    D = distancias(df.text.iloc[teste].tolist(), semantica[teste])
    g_te = agrupar(D)
    Xtr, Xte = centrar(X[treino], df.prompt_id.to_numpy()[treino]), centrar(X[teste], g_te)
    clf = LogisticRegression(C=0.5, max_iter=300).fit(Xtr, y[treino])
    L0 = logp(clf, Xte)
    res = {"argmax": balanced_accuracy_score(y[teste], L0.argmax(1)), "hungaro": balanced_accuracy_score(y[teste], atribuir(L0, g_te))}
    for i, L in enumerate(autotreino(Xtr, y[treino], Xte, g_te, L0), 1):
        res[f"auto{i}"] = balanced_accuracy_score(y[teste], atribuir(L, g_te))
    print(f"dobra {dobra}: " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()) + f"   ({time.time() - t0:.0f}s)", flush=True)
