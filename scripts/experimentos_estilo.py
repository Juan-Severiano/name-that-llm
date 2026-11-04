"""Experimentos locais de estilo; nenhuma alteração na inferência de produção.

Seleção em GroupKFold(3), dentro de 80% dos prompts. O holdout de 20% é
avaliado uma única vez, depois da seleção. Todos os dados já participaram
historicamente de experiências anteriores: este é um diagnóstico novo, não
um teste virgem. CPU, uma thread; execução retomável por candidato.
"""

import os

for _nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_nome] = "1"

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "runs/experimentos_estilo"
SEMENTE = 20261103
PALAVRAS = re.compile(r"\b\w+(?:['’]\w+)?\b", flags=re.UNICODE)
TOKENS = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
FUNCIONAIS = """a an the and or but if when while whereas although though because since as until unless before after then than that this these those is are was were be been being am can could may might must shall should will would do does did have has had having it its it's they them their theirs themselves we us our ours ourselves you your yours yourself he him his she her hers i me my mine who whom whose which what where why how not no nor neither either both all any some many much few more most less least each every other another such so very too quite rather just only even still yet also almost already ever never always often sometimes usually perhaps maybe really indeed however therefore moreover furthermore nevertheless thus hence instead otherwise for to of in on at by with from into through throughout over under between among against about around beyond within without up down out off again once now here there together apart back own same enough one two first second next last finally ultimately overall whether themselves itself himself herself also let lets don't doesn't didn't isn't aren't wasn't weren't won't wouldn't can't cannot couldn't shouldn't need needs ought upon toward towards like unlike per via versus versus""".split()
FUNCIONAIS = tuple(sorted(set(FUNCIONAIS)))
FUNC_SET = set(FUNCIONAIS)
FRASES = (
    "in conclusion", "in short", "in essence", "ultimately", "however", "moreover", "furthermore",
    "for example", "for instance", "in contrast", "rather than", "not just", "not merely", "not only",
    "more than", "as well as", "at the same time", "on the other hand", "in other words", "it is",
    "it's", "there is", "there are", "it also", "it can", "this means", "this is", "that is",
    "what remains", "instead of", "even as", "as if", "whether", "yet", "because", "by contrast",
)
PONTUACAO = ".,;:!?—–-()[]{}\"'‘’“”/\\…*#%&+=<>_`|"


class Estilometria(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([self.uma(str(t)) for t in X], dtype=np.float64)

    @staticmethod
    def uma(t):
        palavras = PALAVRAS.findall(t)
        minusculas = [p.lower() for p in palavras]
        n, c = max(len(palavras), 1), max(len(t), 1)
        comprimentos = np.array([len(p) for p in palavras] or [0])
        frases = [x for x in re.split(r"[.!?]+", t) if x.strip()]
        tokens_por_frase = np.array([len(PALAVRAS.findall(x)) for x in frases] or [0])
        # Números absolutos e formas normalizadas capturam comprimento e ritmo.
        base = [len(t), len(palavras), len(frases), len(set(minusculas)), len(t.splitlines()), len(t.split("\n\n"))]
        for a in (comprimentos, tokens_por_frase):
            base.extend([a.mean(), a.std(), a.min(), a.max(), *np.percentile(a, [10, 25, 50, 75, 90])])
        base.extend([len(set(minusculas)) / n, sum(len(x) for x in palavras) / c,
                     sum(x.isupper() for x in palavras) / n,
                     sum(x[:1].isupper() for x in palavras) / n,
                     sum(any(ch.isdigit() for ch in x) for x in palavras) / n,
                     sum(x.isupper() for x in t) / c,
                     sum(x.isdigit() for x in t) / c,
                     sum(ord(x) > 127 for x in t) / c,
                     sum(x.isspace() for x in t) / c,
                     len(re.findall(r"\s{2,}", t)) / n,
                     len(re.findall(r"\b\w+['’]\w+\b", t)) / n,
                     len(re.findall(r"\b\w+-\w+\b", t)) / n,
                     len(re.findall(r"[!?]{2,}|\.{2,}|—", t)) / n,
                     float(t.startswith(" ")), float(t.endswith(" ")),
                     float(bool(t) and t[:1] in '\"“'), float(bool(t) and t[-1:] in '\"”'),
                     float(bool(t) and t[-1:] in '.!?'), float(t.endswith(':'))])
        base.extend((comprimentos == i).mean() for i in range(1, 15))
        base.extend(sum(comprimentos >= i) / n for i in (5, 7, 10, 12, 15))
        for p in PONTUACAO:
            posiciones = np.array([i / c for i, x in enumerate(t) if x == p])
            base.extend([len(posiciones), len(posiciones) / n,
                         float(posiciones.mean()) if len(posiciones) else -1,
                         float(posiciones.std()) if len(posiciones) else 0])
        for w in FUNCIONAIS:
            posiciones = [i / n for i, x in enumerate(minusculas) if x == w]
            base.extend([len(posiciones) / n, float(bool(minusculas and minusculas[0] == w)),
                         float(bool(minusculas and minusculas[-1] == w))])
        tl = t.lower()
        base.extend(tl.count(p) / n for p in FRASES)
        for inicio, fin in ((0, 12), (0, 30), (-12, None), (-30, None)):
            tramo = minusculas[inicio:fin]
            m = max(len(tramo), 1)
            base.extend([sum(w in FUNC_SET for w in tramo) / m,
                         np.mean([len(w) for w in tramo]) if tramo else 0])
        return base


def mascara_textos(textos):
    def mascara(t):
        salida = []
        for tok in TOKENS.findall(t):
            low = tok.lower()
            if low in FUNC_SET or not tok[0].isalnum():
                salida.append(low)
            else:
                longitud = min(len(tok), 12)
                sufijo = next((s for s in ("ing", "tion", "ment", "ly", "ed", "s") if low.endswith(s)), "_")
                salida.append(f"W{longitud}{'U' if tok[:1].isupper() else 'l'}{sufijo}")
        return " ".join(salida)
    return [mascara(str(t)) for t in textos]


def bordas_textos(textos):
    salida = []
    for t in textos:
        toks = t.split()
        salida.append(" ".join(toks[:25]) + " ||FIM|| " + " ".join(toks[-25:]))
    return salida


def forma_textos(textos):
    return [re.sub(r"[A-Za-z]+", "w", re.sub(r"\d+", "d", t)) for t in textos]


def tfidf_wordchar(char=(1, 5), word=(1, 2), c=20):
    return make_pipeline(FeatureUnion([
        ("palavra", TfidfVectorizer(lowercase=False, min_df=2, sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=word, max_features=90000)),
        ("caractere", TfidfVectorizer(lowercase=False, min_df=3, sublinear_tf=True,
                                     analyzer="char", ngram_range=char, max_features=180000)),
    ]), LogisticRegression(C=c, max_iter=1500))


def candidatos():
    d = {}
    for c in (.03, .3, 3):
        d[f"estilo_lr_c{c}"] = lambda c=c: make_pipeline(Estilometria(), StandardScaler(), LogisticRegression(C=c, max_iter=1500))
    for c in (1, 10):
        d[f"estilo_rbf_c{c}"] = lambda c=c: make_pipeline(Estilometria(), StandardScaler(), SVC(C=c, gamma="scale"))
    d["estilo_boost"] = lambda: make_pipeline(Estilometria(), HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=7, l2_regularization=10, random_state=SEMENTE))
    d["tfidf_original"] = lambda: tfidf_wordchar()
    d["tfidf_palavra3_char6"] = lambda: tfidf_wordchar(char=(2, 6), word=(1, 3), c=8)
    for nome, analyzer, ng in (("char_2a6", "char", (2, 6)), ("char_4a8", "char", (4, 8)), ("charwb_3a6", "char_wb", (3, 6))):
        d[nome] = lambda analyzer=analyzer, ng=ng: make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=ng, lowercase=False, min_df=3, sublinear_tf=True, max_features=220000), LogisticRegression(C=8, max_iter=1500))
    d["gramatica_mascarada"] = lambda: make_pipeline(FunctionTransformer(mascara_textos), TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\S+", ngram_range=(1, 4), min_df=2, sublinear_tf=True, max_features=150000), LogisticRegression(C=10, max_iter=1500))
    d["bordas_texto"] = lambda: make_pipeline(FunctionTransformer(bordas_textos), TfidfVectorizer(analyzer="char", ngram_range=(1, 6), lowercase=False, min_df=3, sublinear_tf=True, max_features=180000), LogisticRegression(C=15, max_iter=1500))
    d["forma_texto"] = lambda: make_pipeline(FunctionTransformer(forma_textos), TfidfVectorizer(analyzer="char", ngram_range=(1, 8), lowercase=False, min_df=3, sublinear_tf=True, max_features=100000), LogisticRegression(C=10, max_iter=1500))
    for peso in (.5, 1, 2):
        def fabrica(peso=peso):
            texto = tfidf_wordchar().steps[0][1]
            estilo = make_pipeline(Estilometria(), StandardScaler(), Normalizer())
            return make_pipeline(FeatureUnion([("texto", texto), ("estilo", estilo)], transformer_weights={"estilo": peso}), LogisticRegression(C=20, max_iter=1500))
        d[f"tfidf_estilo_peso{peso}"] = fabrica
    return d


def proba(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)
    return softmax(modelo.decision_function(X), axis=1)


def grava(nome, valor):
    (SAIDA / nome).write_text(json.dumps(valor, ensure_ascii=False, indent=2) + "\n")


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    if (SAIDA / "holdout.json").exists():
        print("Seleção e holdout já congelados; preservar resultados existentes.", flush=True)
        return
    df = pd.read_csv(RAIZ / "data/train.csv", keep_default_na=False)
    classes = np.sort(df.model.unique())
    desenvolvimento, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEMENTE).split(df, groups=df.prompt_id))
    dev = df.iloc[desenvolvimento].reset_index(drop=True)
    folds = list(GroupKFold(n_splits=3).split(dev, dev.model, dev.prompt_id))
    assert set(dev.prompt_id).isdisjoint(df.iloc[holdout].prompt_id)
    np.savez_compressed(SAIDA / "split.npz", desenvolvimento=desenvolvimento, holdout=holdout, classes=classes.astype(str))
    grava("protocolo.json", {
        "semente": SEMENTE, "split": "GroupShuffleSplit 20% holdout; GroupKFold 3 seleção",
        "historico_exposicao": "Todo o conjunto já foi usado historicamente na escolha de camadas/experimentos; holdout diagnóstico não é virgem.",
        "prompts_desenvolvimento": sorted(dev.prompt_id.unique()), "prompts_holdout": sorted(df.iloc[holdout].prompt_id.unique()),
        "linhas_desenvolvimento": len(dev), "linhas_holdout": len(holdout),
        "sha256_ids_treino": hashlib.sha256("\n".join(df.id).encode()).hexdigest(),
        "estilo_features": Estilometria().transform([""]).shape[1],
    })
    resultados = []
    configs = candidatos()
    for nome, fabrica in configs.items():
        arquivo = SAIDA / f"dev_{nome}.json"
        if arquivo.exists():
            r = json.loads(arquivo.read_text())
            resultados.append(r)
            print(f"CACHE {nome}: {r['balanced_accuracy']:.4f}", flush=True)
            continue
        t = time.monotonic()
        oof = np.empty((len(dev), len(classes)))
        notas = []
        for k, (tr, va) in enumerate(folds):
            m = fabrica().fit(dev.text.iloc[tr].tolist(), dev.model.iloc[tr])
            assert np.array_equal(m.classes_, classes)
            oof[va] = proba(m, dev.text.iloc[va].tolist())
            notas.append(balanced_accuracy_score(dev.model.iloc[va], classes[oof[va].argmax(1)]))
            print(f"{nome} fold{k+1}: {notas[-1]:.4f} ({time.monotonic()-t:.1f}s)", flush=True)
        r = {"nome": nome, "balanced_accuracy": balanced_accuracy_score(dev.model, classes[oof.argmax(1)]), "dobras": notas, "segundos": time.monotonic()-t,
             "saida_probabilidades": "softmax de decision_function, sem calibração" if nome.startswith("estilo_rbf") else "predict_proba nativo; sem calibração separada"}
        np.save(SAIDA / f"oof_{nome}.npy", oof)
        grava(f"dev_{nome}.json", r)
        resultados.append(r)
        grava("resultados_desenvolvimento.json", sorted(resultados, key=lambda r: -r["balanced_accuracy"]))
    melhor = max(resultados, key=lambda r: r["balanced_accuracy"])
    grava("selecao_congelada.json", {"selecionado": melhor, "criterio": "Maior balanced accuracy OOF em desenvolvimento, antes de avaliar holdout."})
    # Esta trava impede reavaliar/reselecionar com o holdout em reexecuções.
    if (SAIDA / "holdout.json").exists():
        print("HOLDOUT JÁ AVALIADO; não repetir seleção.", flush=True)
        return
    m = configs[melhor["nome"]]().fit(dev.text.tolist(), dev.model)
    p = proba(m, df.text.iloc[holdout].tolist())
    np.save(SAIDA / "probas_holdout_selecionado.npy", p)
    previsto = classes[p.argmax(1)]
    rng = np.random.default_rng(SEMENTE)
    grupo = df.prompt_id.iloc[holdout].to_numpy()
    rotulos = df.model.iloc[holdout].to_numpy()
    unicos = np.unique(grupo)
    boot = []
    for _ in range(1500):
        idx = np.concatenate([np.flatnonzero(grupo == g) for g in rng.choice(unicos, len(unicos), replace=True)])
        boot.append(balanced_accuracy_score(rotulos[idx], previsto[idx]))
    r = {"selecionado": melhor["nome"], "balanced_accuracy": balanced_accuracy_score(rotulos, previsto),
         "log_loss": log_loss(rotulos, p, labels=classes), "ic95_bootstrap_por_prompt": np.percentile(boot, [2.5, 97.5]).tolist(),
         "n": len(holdout), "prompts": len(unicos), "nota": "Holdout consultado uma única vez após congelar a seleção. Histórico de exposição consta no protocolo."}
    grava("holdout.json", r)
    grava("confusao_holdout.json", {"classes": classes.tolist(), "matriz": confusion_matrix(rotulos, previsto, labels=classes).tolist()})
    print("FINAL", json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
