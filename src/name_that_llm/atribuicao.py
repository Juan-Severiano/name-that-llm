"""Cada modelo escreve no máximo um texto por prompt, então os rótulos de um grupo de textos do mesmo prompt
são todos diferentes. Infere os grupos pelo texto (sem prompt_id) e escolhe, em cada um, a atribuição de
rótulos distintos que maximiza a soma dos log-probabilidades (algoritmo húngaro).

Só usa numpy, scipy e sklearn: vira parte do notebook do Kaggle.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

CORTE = 0.8  # distância máxima de ligação média dentro de um grupo
TETO = 18  # um texto por modelo: grupo maior que isso mistura prompts


def distancias(textos: list[str], semantica: np.ndarray) -> np.ndarray:
    """Média da distância de cosseno em TF-IDF de conteúdo (sem stopwords) e em embeddings semânticos."""
    try:
        T = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=100_000, dtype=np.float32).fit_transform(textos)
        sim_t = (T @ T.T).toarray()
    except ValueError:  # vocabulário vazio
        sim_t = np.zeros((len(textos), len(textos)), dtype=np.float32)
    E = np.asarray(semantica, dtype=np.float32)
    E = normalize(E - E.mean(axis=0))
    D = np.clip(1 - 0.5 * (sim_t + E @ E.T), 0, 2).astype(np.float64)
    np.fill_diagonal(D, 0)
    return D


def agrupar(D: np.ndarray, corte: float = CORTE, teto: int = TETO) -> np.ndarray:
    """Corte da árvore de ligação média; grupo com mais de `teto` textos é recortado com cortes menores."""
    rotulos = np.zeros(len(D), dtype=int)
    if len(D) < 2:
        return rotulos
    proximo = 0
    pilha = [(np.arange(len(D)), corte)]
    while pilha:
        idx, c = pilha.pop()
        if len(idx) <= 1:
            rotulos[idx], proximo = proximo, proximo + 1
            continue
        g = fcluster(linkage(squareform(D[np.ix_(idx, idx)], checks=False), method="average"), c, criterion="distance")
        for k in np.unique(g):
            membros = idx[g == k]
            if len(membros) > teto and c > 0.2:
                pilha.append((membros, c - 0.03))
            else:
                rotulos[membros], proximo = proximo, proximo + 1
    return rotulos


def atribuir(logp: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    """Índice de classe de cada texto: húngaro por grupo; grupo de um texto ou com mais classes que texto sobra no argmax."""
    pred = logp.argmax(axis=1)
    for g in np.unique(grupos):
        idx = np.where(grupos == g)[0]
        if 1 < len(idx) <= logp.shape[1]:
            linhas, colunas = linear_sum_assignment(-logp[idx])
            pred[idx[linhas]] = colunas
    return pred


def refinar(base, gauss, textos, semantica, grupos_gauss, menor_grupo: int = 4, peso_gauss: float = 0.7):
    """Mistura os log-probabilidades (base e Gaussian; grupo pequeno do Gaussian só usa a base) e aplica o
    húngaro nos grupos inferidos. Devolve (log-probabilidades misturados, índice de classe por texto)."""
    L = np.log(np.clip(base, 1e-9, None))
    LG = np.log(np.clip(gauss, 1e-9, None))
    _, inv, contagem = np.unique(grupos_gauss, return_inverse=True, return_counts=True)
    usa = (contagem[inv] >= menor_grupo)[:, None]
    L = np.where(usa, (1 - peso_gauss) * L + peso_gauss * LG, L)
    return L, atribuir(L, agrupar(distancias(textos, semantica)))
