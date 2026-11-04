"""Cada modelo escreve no máximo um texto por prompt: dentro de um grupo de textos do mesmo prompt, os rótulos
são todos diferentes. Mede quanto isso rende com grupos inferidos do texto (sem prompt_id), usando as
probabilidades fora da dobra que já existem. Uso: PYTHONPATH=src uv run python scripts/experimento_atribuicao.py"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import normalize

from name_that_llm.data import carregar_treino
from name_that_llm.embeddings import caminho

df = carregar_treino()
classes = np.sort(df.model.unique())
y = df.model.to_numpy()
P = np.mean([np.load(f"runs/oof_{n}.npy") for n in ("tfidf", "bge-small-en-v1.5", "roberta-base", "deberta-v3-base", "Qwen3-0.6B")], axis=0)
L = np.log(P + 1e-9)

T = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=100_000, dtype=np.float32).fit_transform(df.text)
sim_t = (T @ T.T).toarray()
E = np.load(caminho("BAAI/bge-small-en-v1.5", "treino"))[:, 12].astype(np.float32)
E = normalize(E - E.mean(axis=0))
sim_b = E @ E.T


def atribuir(grupos: np.ndarray) -> np.ndarray:
    """Hungria em cada grupo de até 18 textos; grupos maiores ficam no argmax."""
    pred = classes[L.argmax(axis=1)].astype(object)
    for g in np.unique(grupos):
        idx = np.where(grupos == g)[0]
        if 1 < len(idx) <= len(classes):
            r, c = linear_sum_assignment(-L[idx])
            pred[idx[r]] = classes[c]
    return pred


def relatorio(nome: str, grupos: np.ndarray) -> None:
    verdade = df.prompt_id.to_numpy()
    mesmo = grupos[:, None] == grupos[None]
    igual = verdade[:, None] == verdade[None]
    np.fill_diagonal(mesmo, False)
    prec = (mesmo & igual).sum() / max(mesmo.sum(), 1)
    rec = (mesmo & igual).sum() / igual.sum() * 1.0 - 0  # igual já exclui diagonal abaixo
    n_ig = igual.sum() - len(df)
    rec = (mesmo & igual).sum() / n_ig
    tam = np.bincount(np.unique(grupos, return_inverse=True)[1])
    nota = balanced_accuracy_score(y, atribuir(grupos))
    print(f"{nome:34s} grupos={len(tam):4d} maior={tam.max():3d} prec={prec:.3f} rec={rec:.3f} ARI={adjusted_rand_score(verdade, grupos):.3f}  ->  acc={nota:.4f}")




def cortar_com_teto(D: np.ndarray, corte: float, teto: int = 18) -> np.ndarray:
    """Corte da árvore de ligação média; grupo com mais de `teto` textos é recortado em cortes menores."""
    rotulos = np.zeros(len(D), dtype=int)
    proximo = 0
    pilha = [(np.arange(len(D)), corte)]
    while pilha:
        idx, c = pilha.pop()
        if len(idx) <= 1:
            rotulos[idx], proximo = proximo, proximo + 1
            continue
        sub = D[np.ix_(idx, idx)]
        g = fcluster(linkage(squareform(sub, checks=False), method="average"), c, criterion="distance")
        for k in np.unique(g):
            m = idx[g == k]
            if len(m) > teto and c > 0.2:
                pilha.append((m, c - 0.03))
            else:
                rotulos[m], proximo = proximo, proximo + 1
    return rotulos


def distancia(peso: float) -> np.ndarray:
    D = np.clip(1 - (peso * sim_t + (1 - peso) * sim_b), 0, 2)
    np.fill_diagonal(D, 0)
    return D


z = np.load("runs/gaussiano_validacao_completa/oof.npz", allow_pickle=True)
assert (z["ids"] == df.id.to_numpy().astype(str)).all() and (z["classes"] == classes).all()
oraculo = np.unique(df.prompt_id, return_inverse=True)[1]
D = distancia(0.5)
for nome, prob in (("media 5 componentes", P), ("gaussian", z["gaussian"]), ("robusto", z["robust"])):
    L = np.log(prob + 1e-9)
    print(f"== {nome}: argmax {balanced_accuracy_score(y, classes[L.argmax(1)]):.4f}")
    relatorio("  oraculo", oraculo)
    for c in (0.65, 0.7, 0.75):
        relatorio(f"  corte {c} com teto 18", cortar_com_teto(D, c))


print("\n== sensibilidade ao numero de prompts no lote (probabilidades gaussian)")
L = np.log(z["gaussian"] + 1e-9)
rng = np.random.default_rng(0)
todos = np.unique(df.prompt_id)
for n_prompts in (40, 75, 110, 150):
    for corte in (0.7, 0.75, 0.8):
        notas, aris = [], []
        for s in range(3 if n_prompts < 150 else 1):
            escolhidos = rng.choice(todos, n_prompts, replace=False)
            idx = np.where(df.prompt_id.isin(escolhidos))[0]
            g = np.full(len(df), -1)
            g[idx] = cortar_com_teto(D[np.ix_(idx, idx)], corte)
            pred = classes[L.argmax(1)].astype(object)
            for k in np.unique(g[idx]):
                m = np.where(g == k)[0]
                if 1 < len(m) <= 18:
                    r, c = linear_sum_assignment(-L[m])
                    pred[m[r]] = classes[c]
            notas.append(balanced_accuracy_score(y[idx], pred[idx]))
            aris.append(adjusted_rand_score(df.prompt_id.to_numpy()[idx], g[idx]))
        base = balanced_accuracy_score(y[idx], classes[L.argmax(1)][idx])
        print(f"prompts={n_prompts:3d} corte={corte}: acc {np.mean(notas):.4f} (argmax ~{base:.3f})  ARI {np.mean(aris):.3f}")
