"""Inferência do envio: lê os artefatos de treino e prevê os textos ocultos.

Só importa bibliotecas de terceiros: o código vira uma célula do notebook do Kaggle
(scripts/montar_notebook.py), onde o pacote name_that_llm não existe. O treino fica em artefatos.py.
Cada componente falha sozinho: o notebook de uma competição de código roda sem log visível.
"""

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.special import expit, softmax
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoModel, AutoTokenizer

TOKENS = r"(?u)\b\w+\b|[^\w\s]"
PARAMETROS_TFIDF = {
    "palavras": {"token_pattern": TOKENS, "ngram_range": (1, 2)},
    "caracteres": {"analyzer": "char", "ngram_range": (1, 5)},
}

# Camadas de cada encoder que entram na regressão logística (escolhidas em fusao.py, validação por prompt).
# A pasta do modelo, local ou no Kaggle, tem o nome depois da barra.
CAMADAS = {
    "BAAI/bge-small-en-v1.5": [2, 4, 6],
    "FacebookAI/roberta-base": [2, 4, 6, 8, 10, 12],
    "microsoft/deberta-v3-base": [4, 8, 10, 11, 12],
    "Qwen/Qwen3-0.6B": [0, 4, 6, 18, 20],
}


def montar_tfidf(classificador) -> Pipeline:
    """Maiúsculas, pontuação e espaços ficam como estão: são o sinal de estilo."""
    atributos = ColumnTransformer(
        [
            (nome, TfidfVectorizer(lowercase=False, min_df=min_df, sublinear_tf=True, **PARAMETROS_TFIDF[nome]), "text")
            for nome, min_df in (("palavras", 2), ("caracteres", 3))
        ]
    )
    return Pipeline([("atributos", atributos), ("clf", classificador)])


def _bloco_tfidf(textos: list[str], termos: list[str], idf: np.ndarray, **parametros) -> sp.csr_matrix:
    contagens = CountVectorizer(vocabulary=termos, lowercase=False, **parametros).transform(textos).astype(np.float64)
    contagens.data = 1 + np.log(contagens.data)  # sublinear_tf
    return normalize(contagens @ sp.diags(idf))


def proba_tfidf(textos: list[str], artefatos: Path) -> np.ndarray:
    """TF-IDF e regressão logística refeitos só com vocabulário, idf e pesos: um pickle do sklearn 1.9 não
    abre no sklearn 1.6 do Kaggle."""
    vocabulario = json.loads((artefatos / "tfidf_vocab.json").read_text())
    with np.load(artefatos / "tfidf.npz", allow_pickle=False) as pesos:
        X = sp.hstack(
            [_bloco_tfidf(textos, vocabulario[n], pesos[f"idf_{n}"], **PARAMETROS_TFIDF[n]) for n in ("palavras", "caracteres")]
        ).tocsr()
        escores = X @ pesos["coef"].T + pesos["intercepto"]
    if escores.shape[1] == 1:
        positiva = expit(escores[:, 0])
        return np.column_stack((1 - positiva, positiva))
    return softmax(escores, axis=1)


def lr_embeddings() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))


@torch.no_grad()
def extrair(modelo: str | Path, textos: list[str], lote: int = 16, max_len: int = 512) -> np.ndarray:
    """Devolve [n_textos, n_camadas, dim] (float16): média dos tokens, sem padding, por camada."""
    if torch.cuda.is_available():
        dispositivo = "cuda"
    elif torch.backends.mps.is_available():
        dispositivo = "mps"
    else:
        dispositivo = "cpu"
    tok = AutoTokenizer.from_pretrained(str(modelo))
    enc = AutoModel.from_pretrained(str(modelo)).to(dispositivo).eval()
    ordem = np.argsort([len(t) for t in textos])  # textos de tamanho parecido no mesmo lote
    saida = [None] * len(textos)
    for i in range(0, len(textos), lote):
        idx = ordem[i : i + lote]
        b = tok([textos[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dispositivo)
        camadas = torch.stack(enc(**b, output_hidden_states=True).hidden_states, dim=1)  # [b, L, T, D]
        m = b["attention_mask"][:, None, :, None].to(camadas.dtype)
        media = (camadas * m).sum(dim=2) / m.sum(dim=2)
        for j, v in zip(idx, media.float().cpu().numpy().astype(np.float16)):
            saida[j] = v
    return np.stack(saida)


def _proba_encoder(modelo: str, camadas: list[int], textos: list[str], pasta_modelos: Path, artefatos: Path, y: np.ndarray):
    nome = modelo.split("/")[-1]
    E = extrair(pasta_modelos / nome, textos)
    X = np.concatenate([E[:, c] for c in camadas], axis=1).astype(np.float32)
    treino = np.load(artefatos / f"emb_{nome}.npy").astype(np.float32)
    return lr_embeddings().fit(treino, y).predict_proba(X)


def prever(
    teste: pd.DataFrame,
    pasta_modelos: str | Path,
    artefatos: str | Path,
    salvar=None,
) -> pd.DataFrame:
    """Média das probabilidades de cinco componentes: TF-IDF e quatro encoders. Devolve id e model.

    Um componente que levanta exceção é pulado (a pilha vai para a saída). Depois de cada um que termina,
    `salvar` recebe a previsão com os componentes prontos até ali: se algo morrer adiante, o arquivo vale.
    """
    pasta_modelos, artefatos = Path(pasta_modelos), Path(artefatos)
    y = np.array(json.loads((artefatos / "rotulos.json").read_text()))
    classes = np.unique(y)
    textos = teste["text"].tolist()

    componentes = {"tfidf": lambda: proba_tfidf(textos, artefatos)} | {
        m: (lambda m=m, c=c: _proba_encoder(m, c, textos, pasta_modelos, artefatos, y)) for m, c in CAMADAS.items()
    }
    probas = {}
    for nome, calcular in componentes.items():
        try:
            probas[nome] = calcular()
        except Exception:
            print(f"{nome}: FALHOU, pulado", flush=True)
            traceback.print_exc()
            continue
        print(f"{nome}: pronto ({len(probas)}/{len(componentes)})", flush=True)
        if salvar:
            salvar(_votar(probas, classes, teste["id"]))
    if not probas:
        raise RuntimeError("nenhum componente funcionou")
    return _votar(probas, classes, teste["id"])


def _votar(probas: dict, classes: np.ndarray, ids) -> pd.DataFrame:
    media = np.mean(list(probas.values()), axis=0)
    return pd.DataFrame({"id": np.asarray(ids), "model": classes[media.argmax(axis=1)]})
