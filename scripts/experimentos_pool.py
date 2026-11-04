"""Curva local pré-declarada com pool conservador e validação por prompt.

Quatro tamanhos (0, ~3k, 9k, 27k), mesmo SGD/logloss de 5 épocas. Hashing
fixo e IDF ajustado só no treino de cada dobra. Nenhum metadado do publicador
é usado como feature ou peso. O domínio oficial tem metade do peso total
quando há pool (e peso por linha >= pool). Holdout só após seleção congelada.
"""

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import hashlib
import gc
import json
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "runs/experimentos_pool"
SEMENTE = 20261103
TAMANHOS = [0, 3000, 9000, 27000]


def grava(nome, d):
    (SAIDA / nome).write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")


def rss_mb():
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return val / (1024 ** 2 if sys.platform == "darwin" else 1024)


def pesos_dominios(n_oficial, n_pool):
    if not n_pool:
        return np.ones(n_oficial)
    total = n_oficial + n_pool
    return np.r_[np.full(n_oficial, total / (2 * n_oficial)), np.full(n_pool, total / (2 * n_pool))]


def features(treinos, validar):
    tr, va = [], []
    for treino, valid in zip(treinos, validar):
        idf = TfidfTransformer(sublinear_tf=True, norm="l2")
        tr.append(idf.fit_transform(treino))
        va.append(idf.transform(valid))
    return sp.hstack(tr, format="csr", dtype=np.float32), sp.hstack(va, format="csr", dtype=np.float32)


def classificador():
    return SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=5, tol=None,
                         average=True, random_state=SEMENTE, shuffle=True, n_jobs=1)


def contar_em_lotes(vetor, textos, lote=1000):
    return sp.vstack([vetor.transform(textos[i:i + lote]) for i in range(0, len(textos), lote)], format="csr")


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    if (SAIDA / "holdout.json").exists():
        print("Seleção e holdout já congelados; não repetir avaliação.", flush=True)
        return
    inicio = time.monotonic()
    df = pd.read_csv(RAIZ / "data/train.csv", keep_default_na=False)
    split = np.load(RAIZ / "runs/experimentos_estilo/split.npz", allow_pickle=False)
    dev_idx, hold_idx, classes = split["desenvolvimento"], split["holdout"], split["classes"]
    dev = df.iloc[dev_idx].reset_index(drop=True)
    auditoria = json.loads((RAIZ / "data/open_pool/auditoria_conservadora.json").read_text())
    assert auditoria["classes_18_preservadas"] and not auditoria["rotulos_oficiais_do_holdout_usados"]
    pool = pd.read_parquet(RAIZ / "data/open_pool/pool_conservador.parquet", columns=["id", "text", "model", "prompt_canon", "source_dataset"])
    assert set(pool.model) == set(classes)
    rng = np.random.default_rng(SEMENTE)
    por_classe = {c: rng.permutation(np.flatnonzero(pool.model.to_numpy() == c)) for c in classes}
    max_por_classe = min(1500, *(len(v) for v in por_classe.values()))
    amostra_max = np.concatenate([v[:max_por_classe] for v in por_classe.values()])
    amostra = pool.iloc[amostra_max].reset_index(drop=True)
    linhas_pool_elegiveis = len(pool)
    del pool
    gc.collect()
    np.savez_compressed(SAIDA / "split_e_amostra.npz", desenvolvimento=dev_idx, holdout=hold_idx,
                        classes=classes, ids_pool=amostra.id.to_numpy(dtype=str), indices_pool_conservador=amostra_max)
    protocolo = {
        "congelado_em": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(),
        "semente": SEMENTE, "tamanhos_pedidos": TAMANHOS,
        "tamanhos_efetivos": {str(n): min(n // len(classes), max_por_classe) * len(classes) for n in TAMANHOS},
        "treino": "120 prompts oficiais; seleção GroupKFold3; pool conservador externo em todas as dobras",
        "metadados_publicador_como_features_ou_pesos": False,
        "features": "Hashing word1-2 + char2-5, 2**18 colunas cada; IDF sublinear/L2 por bloco ajustado no treino de cada dobra",
        "modelo": classificador().get_params(),
        "pesos": "Massa total 50% oficial e 50% pool; média do peso=1; por linha oficial>=pool; não ajustado por score.",
        "historico": "Holdout diagnóstico de 30 prompts já foi exposto em experiências históricas; sem nova consulta nesta curva até seleção congelada.",
        "sha256_pool_conservador": auditoria["sha256_arquivo_treino"],
        "pool_linhas_elegiveis": linhas_pool_elegiveis, "pool_amostra_maxima": len(amostra),
        "aviso": "Surrogate CPU barato de 5 épocas, não mede o potencial de fine-tuning neural com o pool.",
    }
    if (SAIDA / "protocolo.json").exists():
        anterior = json.loads((SAIDA / "protocolo.json").read_text())
        assert anterior["sha256_pool_conservador"] == protocolo["sha256_pool_conservador"], "Pool mudou: preservar execução anterior e iniciar outra pasta."
    grava("protocolo.json", protocolo)
    vectores = [
        HashingVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm=None, dtype=np.float32),
        HashingVectorizer(lowercase=False, analyzer="char", ngram_range=(2, 5), n_features=2**18, alternate_sign=False, norm=None, dtype=np.float32),
    ]
    original_counts, pool_counts = [], []
    for v in vectores:
        original_counts.append(contar_em_lotes(v, dev.text.tolist()))
        pool_counts.append(contar_em_lotes(v, amostra.text.tolist()))
        print(f"CONTAGENS {v.analyzer}: {time.monotonic()-inicio:.1f}s, rssmax={rss_mb():.0f}MB", flush=True)
    folds = list(GroupKFold(3).split(dev, dev.model, dev.prompt_id))
    resultados = []
    selecoes = {}
    for n in TAMANHOS:
        por = min(n // len(classes), max_por_classe)
        pos = np.concatenate([np.arange(i * max_por_classe, i * max_por_classe + por) for i in range(len(classes))]) if por else np.empty(0, dtype=int)
        selecoes[n] = pos
        arq = SAIDA / f"dev_{n}.json"
        if arq.exists():
            resultados.append(json.loads(arq.read_text()))
            continue
        t0 = time.monotonic()
        oof = np.empty((len(dev), len(classes)))
        notas = []
        for k, (tr, va) in enumerate(folds):
            treino_counts = [sp.vstack([a[tr], b[pos]], format="csr") for a, b in zip(original_counts, pool_counts)]
            valid_counts = [a[va] for a in original_counts]
            X, V = features(treino_counts, valid_counts)
            y = np.r_[dev.model.iloc[tr].to_numpy(), amostra.model.iloc[pos].to_numpy()]
            m = classificador().fit(X, y, sample_weight=pesos_dominios(len(tr), len(pos)))
            assert np.array_equal(m.classes_, classes)
            oof[va] = m.predict_proba(V)
            assert np.isfinite(oof[va]).all()
            notas.append(float(balanced_accuracy_score(dev.model.iloc[va], classes[oof[va].argmax(1)])))
            print(f"POOL={len(pos)} fold{k+1}: {notas[-1]:.4f}; {time.monotonic()-t0:.1f}s; rssmax={rss_mb():.0f}MB", flush=True)
            del X, V, treino_counts, valid_counts, m
        r = {"pool_pedido": n, "pool_efetivo": len(pos), "balanced_accuracy": float(balanced_accuracy_score(dev.model, classes[oof.argmax(1)])), "dobras": notas, "segundos": time.monotonic()-t0, "rss_max_mb": rss_mb()}
        grava(f"dev_{n}.json", r)
        np.save(SAIDA / f"oof_{n}.npy", oof)
        resultados.append(r)
        grava("curva_desenvolvimento.json", resultados)
    melhor = max(resultados, key=lambda r: r["balanced_accuracy"])
    grava("selecao_congelada.json", {"selecionado": melhor, "criterio": "Maior OOF dev, sem consultar novamente holdout até este ponto", "congelado_em": datetime.now(ZoneInfo("America/Fortaleza")).isoformat()})
    pos = selecoes[melhor["pool_pedido"]]
    treino_counts = [sp.vstack([a, b[pos]], format="csr") for a, b in zip(original_counts, pool_counts)]
    # Primeira transformação/avaliação do holdout nesta curva, após a seleção.
    hold_counts = [v.transform(df.text.iloc[hold_idx].tolist()) for v in vectores]
    X, H = features(treino_counts, hold_counts)
    y = np.r_[dev.model.to_numpy(), amostra.model.iloc[pos].to_numpy()]
    m = classificador().fit(X, y, sample_weight=pesos_dominios(len(dev), len(pos)))
    probas = m.predict_proba(H)
    previsto = classes[probas.argmax(1)]
    assert np.isfinite(probas).all()
    np.save(SAIDA / "probas_holdout_selecionado.npy", probas)
    verdade = df.model.iloc[hold_idx].to_numpy()
    grupos = df.prompt_id.iloc[hold_idx].to_numpy()
    g = np.unique(grupos)
    rng = np.random.default_rng(SEMENTE)
    boot = []
    for _ in range(1000):
        idx = np.concatenate([np.flatnonzero(grupos == z) for z in rng.choice(g, len(g), replace=True)])
        boot.append(balanced_accuracy_score(verdade[idx], previsto[idx]))
    r = {"pool_selecionado": melhor["pool_efetivo"], "balanced_accuracy": float(balanced_accuracy_score(verdade, previsto)),
         "log_loss": float(log_loss(verdade, probas, labels=classes)), "ic95_por_prompt": np.percentile(boot, [2.5, 97.5]).tolist(),
         "linhas": len(hold_idx), "prompts": len(g), "segundos_totais": time.monotonic()-inicio, "rss_max_mb": rss_mb(),
         "nota": "Única avaliação final desta curva; holdout já exposto historicamente e seleção múltipla entre campanhas deve ser declarada."}
    grava("holdout.json", r)
    print("FINAL", json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
