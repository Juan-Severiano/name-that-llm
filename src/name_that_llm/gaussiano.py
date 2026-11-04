"""Candidato local de Gaussian compartilhado por prompt, sem dependência de HF/torch.

Configuração congelada na campanha: cinco vistas, PCA residual até 128 dimensões,
temperatura 3, clusters mixed-content/BGE com average linkage e corte 0.60.
Treino usa prompt_id; inferência seleciona exclusivamente id/text. Não há quotas,
Sinkhorn, repulsão de classes ou suposição de uma resposta de cada classe.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import softmax
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.extmath import randomized_svd


VERSAO = 1
SEMENTE = 20261103
VISTAS = {
    "bge-small-en-v1.5": {"backbone": "BAAI/bge-small-en-v1.5", "camadas": [2, 4, 6]},
    "roberta-base": {"backbone": "FacebookAI/roberta-base", "camadas": [2, 4, 6, 8, 10, 12]},
    "deberta-v3-base": {"backbone": "microsoft/deberta-v3-base", "camadas": [4, 8, 10, 11, 12]},
    "Qwen3-0.6B": {"backbone": "Qwen/Qwen3-0.6B", "camadas": [0, 4, 6, 18, 20]},
    "StyleDistance_L12": {"backbone": "StyleDistance/styledistance", "camadas": [12]},
}
CONFIG = {"dimensao_maxima": 128, "temperatura": 3., "limiar_grupo": .60,
          "pca_iteracoes": 4, "two_way_iteracoes": 8, "cavity_iteracoes": 6,
          "semente": SEMENTE, "lambda_minimo": 1e-4, "classe_unica_por_grupo": False,
          "quota_classes": False, "repel": False, "sinkhorn": False}


@dataclass
class Matrizes:
    """IDs acompanham obrigatoriamente a ordem das linhas de todas as matrizes."""
    ids: np.ndarray
    vistas: dict[str, np.ndarray]
    semantica: np.ndarray


@dataclass
class Modelo:
    classes: np.ndarray
    treino_ids: np.ndarray
    parametros: dict[str, dict[str, np.ndarray]]
    metadados: dict


@dataclass
class Resultado:
    submissao: pd.DataFrame
    probabilidades: np.ndarray
    grupos: np.ndarray
    diagnostico: dict


def checksum_ids(ids):
    return hashlib.sha256("\n".join(np.asarray(ids, dtype=str)).encode()).hexdigest()


def _sha(arquivo):
    h = hashlib.sha256()
    with Path(arquivo).open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def _ids_textos(df):
    if not {"id", "text"} <= set(df.columns):
        raise ValueError("dados precisam de id e text")
    ids = df["id"]
    if ids.isna().any() or ids.duplicated().any() or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("ids precisam ser strings não vazias, únicas e completas")
    if not all(isinstance(t, str) for t in df["text"]):
        raise ValueError("text precisa conter strings, incluindo string vazia quando necessário")
    return ids.to_numpy(dtype=str), df["text"].tolist()


def _matriz(a, linhas, nome):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != linhas or a.shape[1] < 1 or a.dtype.kind not in "fiu":
        raise ValueError(f"shape/dtype inválido em {nome}")
    if not np.isfinite(a).all():
        raise ValueError(f"valores não finitos em {nome}")
    with np.errstate(over="ignore", invalid="ignore"):
        convertido = np.asarray(a, dtype=np.float32)
    if not np.isfinite(convertido).all():
        raise ValueError(f"conversão float32 não finita em {nome}")
    return convertido


def _validar_matrizes(ids, matrizes, modelo=None):
    if not isinstance(matrizes, Matrizes):
        raise TypeError("features precisam de Matrizes com IDs explícitos")
    if not np.array_equal(np.asarray(matrizes.ids, dtype=str), ids):
        raise ValueError("ordem/identidade dos IDs das matrizes difere dos dados")
    if set(matrizes.vistas) != set(VISTAS):
        raise ValueError("vistas ausentes ou inesperadas; configuração de cinco vistas é congelada")
    vistas = {k: _matriz(matrizes.vistas[k], len(ids), k) for k in VISTAS}
    semantica = _matriz(matrizes.semantica, len(ids), "semantica")
    if modelo is not None:
        if semantica.shape[1] != modelo.metadados["dimensao_semantica"]:
            raise ValueError("dimensão semântica difere do modelo")
        for k, x in vistas.items():
            if x.shape[1] != len(modelo.parametros[k]["media"]):
                raise ValueError(f"dimensão da vista {k} difere do modelo")
    return vistas, semantica


def _centrar(X, grupos):
    _, inverso, contagens = np.unique(grupos, return_inverse=True, return_counts=True)
    medias = np.zeros((len(contagens), X.shape[1]), dtype=X.dtype)
    np.add.at(medias, inverso, X)
    medias /= contagens[:, None]
    return X - medias[inverso]


def _padronizar(X, parametros):
    # Mesma aritmética float32 in-place do StandardScaler.transform usado na pesquisa.
    X = np.asarray(X, dtype=np.float32).copy()
    X -= np.asarray(parametros["media"], dtype=np.float32)
    X /= np.asarray(parametros["escala"], dtype=np.float32)
    return X


def _ajustar_vista(X, y, grupos, classes):
    escala = StandardScaler().fit(X)
    Z = _padronizar(X, {"media": escala.mean_, "escala": escala.scale_})
    residual = _centrar(Z, grupos)
    d = min(CONFIG["dimensao_maxima"], len(X)-len(np.unique(grupos))-1, X.shape[1])
    if d < 1:
        raise ValueError("amostras insuficientes para PCA residual dentro dos prompts")
    _, _, vt = randomized_svd(residual, n_components=d, n_iter=CONFIG["pca_iteracoes"], random_state=SEMENTE)
    V = vt.T
    Z = np.asarray(Z @ V, dtype=np.float64)
    _, inv, contagem = np.unique(grupos, return_inverse=True, return_counts=True)
    efeito = np.zeros((len(contagem), d))
    mu = np.zeros((classes, d))
    for _ in range(CONFIG["two_way_iteracoes"]):
        R = Z - efeito[inv]
        for c in range(classes):
            mu[c] = R[y == c].mean(axis=0)
        efeito[:] = 0
        np.add.at(efeito, inv, Z - mu[y])
        efeito /= contagem[:, None]
        deslocamento = efeito.mean(axis=0)
        efeito -= deslocamento
        mu += deslocamento
    E = Z - mu[y] - efeito[inv]
    Se_raw = E.T @ E / max(len(X)-len(contagem)-classes+1, 1)
    ae = float(ledoit_wolf_shrinkage(E, assume_centered=True))
    Se = (1-ae)*Se_raw + ae*np.trace(Se_raw)/d*np.eye(d)
    Se += np.eye(d)*max(float(np.trace(Se)/d*1e-7), 1e-9)
    P = efeito - efeito.mean(axis=0)
    Sp_raw = P.T @ P/max(len(P)-1, 1) - np.mean(1/contagem)*Se_raw
    ap = float(ledoit_wolf_shrinkage(P, assume_centered=True))
    tr = max(float(np.trace(Sp_raw)), 1e-8*float(np.trace(Se)))
    Sp = (1-ap)*Sp_raw + ap*tr/d*np.eye(d)
    Li = np.linalg.inv(np.linalg.cholesky(Se))
    M = Li @ Sp @ Li.T
    valores, Q = np.linalg.eigh((M+M.T)/2)
    lam = np.maximum(valores, CONFIG["lambda_minimo"])
    A = Q.T @ Li
    return {"media": escala.mean_, "escala": escala.scale_, "V": V,
            "A": A, "prototipos": mu @ A.T, "lambda": lam}, {"dimensao": d, "shrink_residual": ae, "shrink_prompt": ap}


def fit(treino: pd.DataFrame, matrizes: Matrizes) -> Modelo:
    ids, _ = _ids_textos(treino)
    if len(ids) == 0 or not {"model", "prompt_id"} <= set(treino.columns):
        raise ValueError("treino precisa de linhas, model e prompt_id")
    if not all(isinstance(v, str) and v for v in treino["model"]) or treino["prompt_id"].isna().any():
        raise ValueError("rótulos/prompts inválidos")
    classes = np.unique(treino["model"].to_numpy(dtype=str))
    if len(classes) < 2:
        raise ValueError("treino precisa de pelo menos duas classes")
    y = np.searchsorted(classes, treino["model"].to_numpy(dtype=str))
    grupos = treino["prompt_id"].to_numpy()
    vistas, sem = _validar_matrizes(ids, matrizes)
    parametros, infos = {}, {}
    for k, x in vistas.items():
        parametros[k], infos[k] = _ajustar_vista(x, y, grupos, len(classes))
    metadados = {"versao": VERSAO, "config": dict(CONFIG), "vistas": json.loads(json.dumps(VISTAS)),
                 "classes": classes.tolist(), "treino_linhas": len(ids), "treino_prompts": len(np.unique(grupos)),
                 "treino_ids_sha256": checksum_ids(ids), "dimensao_semantica": sem.shape[1],
                 "semantica": {"backbone": "BAAI/bge-small-en-v1.5", "camada": 12, "pooling": "mean tokens"},
                 "ajuste": infos, "limites": ["grupos k=1/k=3 podem regredir", "probabilidades T3 subconfidentes", "ganho local não garante leaderboard"]}
    modelo = Modelo(classes, ids, parametros, metadados)
    _validar_modelo(modelo)
    return modelo


def _grupos(textos, semantica):
    if len(textos) < 2:
        return np.arange(len(textos)), {"sem_vocabulario": not any(t.strip() for t in textos)}
    sem_vocabulario = False
    try:
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=100_000, dtype=np.float32).fit_transform(textos)
        distancia = np.clip(1-(X @ X.T).toarray(), 0, 1)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        sem_vocabulario = True
        distancia = np.ones((len(textos), len(textos)))
    E = np.asarray(semantica, dtype=np.float64)
    E = normalize(E-E.mean(axis=0))
    distancia = .5*(distancia+np.clip(1-E @ E.T, 0, 2))
    np.fill_diagonal(distancia, 0)
    grupos = fcluster(linkage(squareform(distancia, checks=False), method="average"), CONFIG["limiar_grupo"], criterion="distance")
    return np.unique(grupos, return_inverse=True)[1], {"sem_vocabulario": sem_vocabulario}


def _probabilidades(X, parametros, grupos):
    Z = _padronizar(X, parametros) @ parametros["V"]
    U = np.asarray(Z, dtype=np.float64) @ parametros["A"].T
    m, lam = parametros["prototipos"], parametros["lambda"]
    R = U[:, None] - m[None]
    p = softmax(-.5*np.sum(R**2/(1+lam), axis=2)/CONFIG["temperatura"], axis=1)
    for g in np.unique(grupos):
        ii = np.flatnonzero(grupos == g)
        if len(ii) < 2:
            continue
        q = p[ii].copy()
        v = lam/(1+(len(ii)-1)*lam)
        for _ in range(CONFIG["cavity_iteracoes"]):
            erros = U[ii]-q @ m
            prompt = (erros.sum(axis=0)[None]-erros)*v
            R = U[ii, None]-prompt[:, None]-m[None]
            q = softmax(-.5*np.sum(R**2/(1+v), axis=2)/CONFIG["temperatura"], axis=1)
        p[ii] = q
    return p


def prever(modelo: Modelo, teste: pd.DataFrame, matrizes: Matrizes) -> Resultado:
    """Somente id/text são lidos de teste; prompt_id/model extras são ignorados."""
    _validar_modelo(modelo)
    ids, textos = _ids_textos(teste)
    vistas, sem = _validar_matrizes(ids, matrizes, modelo)
    if len(ids) == 0:
        return Resultado(pd.DataFrame({"id": ids, "model": np.array([], dtype=str)}), np.empty((0, len(modelo.classes))), np.empty(0, dtype=int), {"vazio": True})
    grupos, diagnostico = _grupos(textos, sem)
    p = np.mean([_probabilidades(vistas[k], modelo.parametros[k], grupos) for k in VISTAS], axis=0)
    if p.shape != (len(ids), len(modelo.classes)) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
        raise ValueError("probabilidades inválidas; nenhuma previsão constante de fallback será criada")
    p /= p.sum(axis=1, keepdims=True)
    saida = pd.DataFrame({"id": ids, "model": modelo.classes[p.argmax(axis=1)]})
    if saida["id"].tolist() != ids.tolist() or not set(saida["model"]) <= set(modelo.classes):
        raise ValueError("IDs/classes de saída inconsistentes")
    _, inv, n = np.unique(grupos, return_inverse=True, return_counts=True)
    diagnostico.update({"linhas": len(ids), "grupos": len(n), "fracao_em_grupos": float(np.mean(n[inv] > 1)),
                       "classes_da_chamada_usadas": False, "prompt_id_da_chamada_usado": False})
    return Resultado(saida, p, grupos, diagnostico)


def _validar_modelo(modelo):
    m = modelo.metadados
    if m.get("versao") != VERSAO or m.get("config") != CONFIG or m.get("vistas") != VISTAS:
        raise ValueError("versão/configuração/vistas do modelo inválidas")
    classes = np.asarray(modelo.classes)
    if classes.ndim != 1 or classes.dtype.kind not in "US" or len(classes) < 2 or not np.array_equal(classes, np.unique(classes)) or any(not c for c in classes):
        raise ValueError("classes precisam ser strings únicas na ordem canônica")
    if classes.tolist() != m.get("classes"):
        raise ValueError("ordem das classes difere dos metadados")
    ids = np.asarray(modelo.treino_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(ids) != m.get("treino_linhas") or len(np.unique(ids)) != len(ids) or checksum_ids(ids) != m.get("treino_ids_sha256"):
        raise ValueError("identidade/ordem do treino inconsistente")
    if set(modelo.parametros) != set(VISTAS):
        raise ValueError("parâmetros de vistas ausentes/inesperados")
    for k, p in modelo.parametros.items():
        if set(p) != {"media", "escala", "V", "A", "prototipos", "lambda"}:
            raise ValueError(f"parâmetros incompletos em {k}")
        if any(np.asarray(v).dtype.kind not in "fiu" or not np.isfinite(v).all() for v in p.values()):
            raise ValueError(f"parâmetros não finitos/não numéricos em {k}")
        D = len(p["media"])
        d = len(p["lambda"])
        if p["media"].shape != (D,) or p["escala"].shape != (D,) or p["V"].shape != (D,d) or p["A"].shape != (d,d) or p["prototipos"].shape != (len(classes),d) or not 1 <= d <= CONFIG["dimensao_maxima"]:
            raise ValueError(f"shapes de parâmetros inconsistentes em {k}")
        if np.any(p["escala"] <= 0) or np.any(p["lambda"] <= 0):
            raise ValueError(f"escala/covariância inválida em {k}")


def salvar(modelo: Modelo, destino):
    _validar_modelo(modelo)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    arrays = {"classes": np.asarray(modelo.classes, dtype=str), "treino_ids": np.asarray(modelo.treino_ids, dtype=str)}
    for i, k in enumerate(VISTAS):
        arrays.update({f"v{i}_{n}": np.asarray(v) for n, v in modelo.parametros[k].items()})
    temporario = destino / "parametros.npz.tmp"
    with temporario.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temporario.replace(destino / "parametros.npz")
    metadata = dict(modelo.metadados) | {"npz_sha256": _sha(destino / "parametros.npz"), "formato": "numpy arrays, allow_pickle=False"}
    temporario = destino / "metadados.json.tmp"
    temporario.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    temporario.replace(destino / "metadados.json")


def carregar(origem) -> Modelo:
    origem = Path(origem)
    metadata = json.loads((origem / "metadados.json").read_text())
    if _sha(origem / "parametros.npz") != metadata.get("npz_sha256"):
        raise ValueError("checksum NPZ inválido")
    with np.load(origem / "parametros.npz", allow_pickle=False) as arq:
        esperados={"classes","treino_ids"}|{f"v{i}_{n}" for i in range(len(VISTAS)) for n in ("media","escala","V","A","prototipos","lambda")}
        if set(arq.files)!=esperados:
            raise ValueError("campos NPZ ausentes/inesperados")
        # Preservar C/F-order: mudar o layout altera a ordem de acumulação de BLAS
        # no produto float32 e quebraria paridade estrita apesar dos mesmos valores.
        parametros = {k: {n: arq[f"v{i}_{n}"].copy(order="K") for n in ("media", "escala", "V", "A", "prototipos", "lambda")} for i, k in enumerate(VISTAS)}
        modelo = Modelo(arq["classes"].copy(), arq["treino_ids"].copy(), parametros, metadata)
    _validar_modelo(modelo)
    return modelo
