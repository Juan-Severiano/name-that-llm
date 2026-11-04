"""Auditoria local de dados, validação e caches; não usa rede nem altera artefatos.

Uso: .venv/bin/python scripts/auditar_dados.py
Os únicos fits são BGE pequeno, seu controle permutado e um controle ID-only.
"""

import os

for variavel in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variavel] = "1"

import ast
import hashlib
import importlib.util
import io
import json
import re
import time
import unicodedata
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


RAIZ = Path(__file__).resolve().parents[1]


def sha256(arquivo):
    h = hashlib.sha256()
    with arquivo.open("rb") as entrada:
        for bloco in iter(lambda: entrada.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def resumir(valores):
    a = np.asarray(valores)
    return {"min": float(a.min()), "p25": float(np.quantile(a, .25)), "mediana": float(np.median(a)),
            "p75": float(np.quantile(a, .75)), "max": float(a.max()), "media": float(a.mean())}


def oof(X, y, folds, padronizar=True):
    classes = np.unique(y)
    p = np.zeros((len(y), len(classes)))
    for treino, valid in folds:
        clf = LogisticRegression(C=.05, max_iter=1000)
        m = make_pipeline(StandardScaler(), clf) if padronizar else clf
        m.fit(X[treino], y[treino])
        assert np.array_equal(m.classes_, classes)
        p[valid] = m.predict_proba(X[valid])
    return p


def bootstrap_prompts(acertos, y, grupos, classes, semente=20261103):
    prompts, indice = np.unique(grupos, return_inverse=True)
    acertos_prompt = np.zeros((len(prompts), len(classes)))
    totais_prompt = np.zeros_like(acertos_prompt)
    mapa = {c: i for i, c in enumerate(classes)}
    colunas = np.array([mapa[c] for c in y])
    np.add.at(acertos_prompt, (indice, colunas), acertos)
    np.add.at(totais_prompt, (indice, colunas), 1)
    pesos = np.random.default_rng(semente).multinomial(len(prompts), np.full(len(prompts), 1 / len(prompts)), size=5000)
    notas = np.mean((pesos @ acertos_prompt) / (pesos @ totais_prompt), axis=1)
    return {"metodo": "bootstrap de 150 prompts, 5000 replicas, condicionado às previsões OOF existentes",
            "intervalo_95": np.quantile(notas, [.025, .975]).tolist(), "desvio": float(notas.std())}


def main():
    inicio = time.monotonic()
    dados = RAIZ / "data"
    runs = RAIZ / "runs"
    df = pd.read_csv(dados / "train.csv", keep_default_na=False)
    teste = pd.read_csv(dados / "test.csv", keep_default_na=False)
    y, grupos = df.model.to_numpy(), df.prompt_id.to_numpy()
    classes = np.unique(y)
    folds = list(GroupKFold(5).split(df, y, grupos))
    classes_col = {c: i for i, c in enumerate(classes)}
    verdadeiro = np.array([classes_col[c] for c in y])
    resultado = {"verificado_em": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(timespec="seconds"),
                 "sklearn": sklearn.__version__, "threads": 1, "rede": False,
                 "fontes": {"train_sha256": sha256(dados / "train.csv"),
                            "test_sha256": sha256(dados / "test.csv"),
                            "ordem_id_model_sha256": hashlib.sha256(df[["id", "model"]].to_csv(index=False).encode()).hexdigest()}}
    resultado["dados"] = {
        "linhas": len(df), "prompts": int(df.prompt_id.nunique()), "classes": classes.tolist(),
        "por_classe": df.model.value_counts().sort_index().to_dict(),
        "tamanhos_de_prompt": {str(k): int(v) for k, v in df.groupby("prompt_id").size().value_counts().items()},
        "ids_duplicados": int(df.id.duplicated().sum()),
        "pares_prompt_modelo_duplicados": int(df.duplicated(["prompt_id", "model"]).sum()),
        "campos_vazios": {c: int(df[c].eq("").sum()) for c in df},
        "ids_intersecao_treino_teste": len(set(df.id) & set(teste.id)),
        "textos_intersecao_treino_teste": len(set(df.text) & set(teste.text)),
        "test_publico_linhas": len(teste), "test_publico_tamanho_insuficiente_para_inferir_shift": True,
        "caracteres_treino": resumir(df.text.str.len()), "palavras_treino": resumir(df.text.str.split().str.len()),
    }
    familias = np.array([str(c).split("-")[0].split("3")[0] for c in y])
    # Normalização em três níveis, sem gravar qualquer texto no relatório.
    normal = df.text.map(lambda s: " ".join(unicodedata.normalize("NFKC", s).casefold().split()))
    sem_pontuacao = normal.map(lambda s: " ".join(re.findall(r"\w+", s)))
    resultado["duplicatas"] = {
        "exatas_linhas_excedentes": int(df.text.duplicated().sum()),
        "unicode_casefold_espacos_linhas_excedentes": int(normal.duplicated().sum()),
        "sem_pontuacao_linhas_excedentes": int(sem_pontuacao.duplicated().sum()),
    }
    # Jaccard exato de conjuntos de bigramas de palavras em todos os pares com interseção.
    X = CountVectorizer(ngram_range=(2, 2), lowercase=True, token_pattern=r"(?u)\b\w+\b", binary=True, dtype=np.int32).fit_transform(df.text)
    tamanhos = np.asarray(X.sum(axis=1)).ravel()
    inter = (X @ X.T).tocoo()
    pares = inter.row < inter.col
    i, j, comum = inter.row[pares], inter.col[pares], inter.data[pares]
    jac = comum / (tamanhos[i] + tamanhos[j] - comum)
    cross = grupos[i] != grupos[j]
    top = np.argsort(jac)[-10:][::-1]
    resultado["duplicatas"]["near_duplicates"] = {
        "metrica": "Jaccard exato dos conjuntos de bigramas de palavras, todos os pares",
        "pares_jaccard_ge_08": int((jac >= .8).sum()),
        "pares_cross_prompt_jaccard_ge_08": int(((jac >= .8) & cross).sum()),
        "maior_jaccard": float(jac.max()),
        "maior_jaccard_cross_prompt": float(jac[cross].max()),
        "top_pares_sem_textos": [{"linhas": [int(i[k]), int(j[k])], "jaccard": float(jac[k]),
                                   "mesmo_prompt": bool(not cross[k]), "mesma_classe": bool(y[i[k]] == y[j[k]])} for k in top],
        "limite": "não detecta paráfrases sem sobreposição lexical; isso exige auditoria semântica separada",
    }
    resultado["folds"] = [{"fold": n, "treino": len(t), "validacao": len(v),
                           "prompts_treino": len(np.unique(grupos[t])), "prompts_validacao": len(np.unique(grupos[v])),
                           "intersecao_prompts": len(set(grupos[t]) & set(grupos[v])),
                           "classes_treino": len(np.unique(y[t])), "classes_validacao": len(np.unique(y[v])),
                           "duplicatas_exatas_entre_partes": len(set(df.text.iloc[t]) & set(df.text.iloc[v]))}
                          for n, (t, v) in enumerate(folds)]
    mod = ast.parse((RAIZ / "src/name_that_llm/pipeline.py").read_text())
    camadas = next(ast.literal_eval(n.value) for n in mod.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CAMADAS" for t in n.targets))
    rotulos = np.array(json.loads((dados / "artefatos/rotulos.json").read_text()))
    resultado["artefatos"] = {"rotulos_mesma_ordem_do_train": bool(np.array_equal(rotulos, y)),
                              "rotulos_invalidos": sorted(set(rotulos) - set(classes)),
                              "embeddings": {}, "oof_sem_manifesto_de_ids_ou_classes": True}
    bge_X, bge_camada4 = None, None
    for modelo, cs in camadas.items():
        nome = modelo.split("/")[-1]
        arquivo = dados / "emb" / f"{modelo.replace('/', '__')}.treino.npy"
        E = np.load(arquivo, allow_pickle=False).astype(np.float32)
        selecionado = np.concatenate([E[:, c] for c in cs], axis=1)
        A = np.load(dados / "artefatos" / f"emb_{nome}.npy", allow_pickle=False).astype(np.float32)
        delta = A - selecionado
        den = np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(selecionado, axis=1), 1e-12)
        cos = np.sum(A * selecionado, axis=1) / den
        trocado = np.roll(selecionado, 1, axis=0)
        cos_trocado = np.sum(A * trocado, axis=1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(trocado, axis=1), 1e-12)
        resultado["artefatos"]["embeddings"][nome] = {
            "cache_shape": list(E.shape), "artefato_shape": list(A.shape), "camadas": cs,
            "finitos": bool(np.isfinite(E).all() and np.isfinite(A).all()),
            "diferenca_maxima": float(np.abs(delta).max()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "coseno_mesma_linha_min": float(cos.min()), "coseno_mesma_linha_mediana": float(np.median(cos)),
            "coseno_linha_deslocada_mediana": float(np.median(cos_trocado)), "cache_sha256": sha256(arquivo)}
        if nome == "bge-small-en-v1.5":
            bge_X, bge_camada4 = selecionado, E[:, 4].copy()
        del E, selecionado, A, delta
    vocab = json.loads((dados / "artefatos/tfidf_vocab.json").read_text())
    with np.load(dados / "artefatos/tfidf.npz", allow_pickle=False) as pesos:
        resultado["artefatos"]["tfidf"] = {
            "vocabularios_tamanho": {k: len(v) for k, v in vocab.items()},
            "vocabularios_sem_duplicatas": all(len(set(v)) == len(v) for v in vocab.values()),
            "coef_shape": list(pesos["coef"].shape), "finitos": all(np.isfinite(pesos[k]).all() for k in pesos.files),
            "colunas_coerentes": pesos["coef"].shape[1] == sum(len(v) for v in vocab.values()),
            "ordem_classes_nao_persistida_no_npz": True,
        }
    ps = {p.stem.removeprefix("oof_"): np.load(p, allow_pickle=False) for p in sorted(runs.glob("oof_*.npy"))}
    resultado["oof"] = {}
    for nome, p in ps.items():
        resultado["oof"][nome] = {"shape": list(p.shape), "finitos": bool(np.isfinite(p).all()),
                                  "min": float(p.min()), "max_erro_soma": float(np.max(np.abs(p.sum(axis=1) - 1))),
                                  "balanced_accuracy": float(balanced_accuracy_score(y, classes[p.argmax(axis=1)])),
                                  "sha256": sha256(runs / f"oof_{nome}.npy")}
    probas = np.mean(list(ps.values()), axis=0)
    sem_tfidf = np.mean([p for n, p in ps.items() if n != "tfidf"], axis=0)
    acertou = probas.argmax(axis=1) == verdadeiro
    oracle = np.any(np.column_stack([p.argmax(axis=1) == verdadeiro for p in ps.values()]), axis=1)
    familia_classes = np.array([str(c).split("-")[0].split("3")[0] for c in classes])
    recall = {c: float(acertou[y == c].mean()) for c in classes}
    prompt_acc = pd.Series(acertou).groupby(grupos).mean().to_numpy()
    resultado["ensemble"] = {
        "cinco_balanced_accuracy": float(balanced_accuracy_score(y, classes[probas.argmax(axis=1)])),
        "quatro_sem_tfidf_balanced_accuracy": float(balanced_accuracy_score(y, classes[sem_tfidf.argmax(axis=1)])),
        "public_score_v2": .393, "gap_cv_quatro_para_publico": float(balanced_accuracy_score(y, classes[sem_tfidf.argmax(axis=1)]) - .393),
        "ci_cinco": bootstrap_prompts(acertou, y, grupos, classes),
        "ci_quatro": bootstrap_prompts(sem_tfidf.argmax(axis=1) == verdadeiro, y, grupos, classes),
        "por_dobra_cinco": [float(balanced_accuracy_score(y[v], classes[probas[v].argmax(axis=1)])) for _, v in folds],
        "recall_por_classe": recall, "accuracy_por_prompt": resumir(prompt_acc),
        "acerto_familia": float(np.mean(familia_classes[probas.argmax(axis=1)] == familias)),
        "oracle_escolher_top1_de_um_componente_balanced_accuracy": float(np.mean([oracle[y == c].mean() for c in classes])),
        "limite_oracle": "limite para selecionar uma das top1 atuais; fusão pode acertar classes que nenhum componente colocou em primeiro",
        "topk": {str(k): float(np.mean([np.mean(np.any(np.argsort(probas[y == c], axis=1)[:, -k:] == classes_col[c], axis=1)) for c in classes])) for k in (1, 2, 3, 5)},
    }
    previsto = classes[probas.argmax(axis=1)]
    erros = pd.DataFrame({"real": y[~acertou], "previsto": previsto[~acertou]}).value_counts().head(15)
    resultado["ensemble"]["maiores_confusoes"] = [{"real": a, "previsto": b, "contagem": int(n)} for (a, b), n in erros.items()]
    resultado["ensemble"]["erros_dentro_da_mesma_familia_fracao"] = float(np.mean(familia_classes[probas[~acertou].argmax(axis=1)] == familias[~acertou]))
    # Exercita o caminho de inferência real sem carregar nenhum encoder ou alterar arquivos do projeto.
    spec = importlib.util.spec_from_file_location("pipeline_auditoria", RAIZ / "src/name_that_llm/pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    entradas_borda = ["", " \t\n", "🦙 中文 Ω Ж"]
    borda = pipeline.proba_tfidf(entradas_borda, dados / "artefatos")
    originais = pipeline.CAMADAS, pipeline.proba_tfidf, pipeline._proba_encoder
    try:
        pipeline.CAMADAS = {"auditoria/encoder": [0]}
        pipeline.proba_tfidf = lambda textos, artefatos: np.full((len(textos), len(classes)), np.nan)
        pipeline._proba_encoder = lambda modelo, cs, textos, pasta, artefatos, y: np.full((len(textos), len(classes)), 1 / len(classes))
        with redirect_stdout(io.StringIO()):
            contaminado = pipeline.prever(pd.DataFrame({"id": ["audit-1", "audit-2"], "text": ["a", "b"]}), dados / "modelos", dados / "artefatos")
    finally:
        pipeline.CAMADAS, pipeline.proba_tfidf, pipeline._proba_encoder = originais
    resultado["cenarios_de_falha"] = {
        "textos_vazios_ou_desconhecidos": {"finitos": bool(np.isfinite(borda).all()), "max_erro_soma": float(np.abs(borda.sum(axis=1) - 1).max())},
        "injecao_probabilidades_nan": {"metodo": "um componente devolve NaN e outro devolve probabilidades uniformes; mesmo prever() de produção",
                                       "falha_levantada": False, "saida_constante_primeira_classe": bool(contaminado.model.eq(classes[0]).all()),
                                       "interpretacao": "risco confirmado no código, não observado nos artefatos atuais, que são finitos"},
        "todos_componentes_falham": "prever levanta RuntimeError; célula do notebook captura e mantém fallback constante, então COMPLETE não prova componentes funcionais",
        "rotulos_ausentes": "célula captura falha ao localizar rotulos.json e mantém fallback constante",
        "fallback_gpt55_valido": "gpt-5.5" in classes,
        "manifestos_ausentes": "troca silenciosa da ordem dos labels/classes não é detectada automaticamente; nesta auditoria a ordem bate exatamente",
    }
    with threadpool_limits(limits=1):
        print("Auditoria: reproduzindo BGE OOF...", flush=True)
        refeito = oof(bge_X, y, folds)
        resultado["reproducao_bge_oof"] = {
            "diferenca_maxima": float(np.abs(refeito - ps["bge-small-en-v1.5"]).max()),
            "argmax_iguais": bool(np.array_equal(refeito.argmax(axis=1), ps["bge-small-en-v1.5"].argmax(axis=1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, classes[refeito.argmax(axis=1)])),
            "mesma_ordem_classes_verificada_em_todos_fits": True,
        }
        rng = np.random.default_rng(20261103)
        permutado = y.copy()
        for g in np.unique(grupos):
            idx = np.flatnonzero(grupos == g)
            permutado[idx] = rng.permutation(y[idx])
        print("Auditoria: controle labels permutados BGE camada4...", flush=True)
        p_perm = oof(bge_camada4, permutado, folds)
        id_features = np.array([[int(c, 16) for c in s] for s in df.id])
        id_onehot = np.eye(16)[id_features].reshape(len(df), -1)
        print("Auditoria: controle ID-only...", flush=True)
        p_id = oof(id_onehot, y, folds, padronizar=False)
    resultado["controles"] = {
        "acaso_balanced_accuracy": 1 / len(classes),
        "labels_embaralhados": {"metodo": "embaralhar labels dentro de cada prompt; retreinar BGE camada4 com 5 folds disjuntos",
                                  "semente": 20261103, "balanced_accuracy": float(balanced_accuracy_score(permutado, classes[p_perm.argmax(axis=1)]))},
        "id_only": {"representacao": "12 posições hexadecimais one-hot; 192 dimensões; LR C=.05; mesmos GroupKFold",
                    "balanced_accuracy": float(balanced_accuracy_score(y, classes[p_id.argmax(axis=1)])),
                    "ids_sao_hex_12": bool(df.id.str.fullmatch(r"[0-9a-f]{12}").all()),
                    "pipeline_producao_usa_somente_coluna_text": True},
    }
    resultado["fatos_interpretativos"] = [
        "As camadas e o subconjunto foram comparados nas mesmas 5 dobras; o score não é uma avaliação externa independente da seleção.",
        "A simulação de 30 prompts do treino verifica o caminho de execução, mas foi feita depois de escolhas guiadas pelos 150 prompts.",
        "Arrays de cache não guardam IDs/classes/configuração; a reprodução BGE e a comparação linha a linha reduzem o risco, sem garantir a proveniência histórica de todos os caches.",
        "A nota 0.393 de v2 não identifica quais componentes rodaram no oculto. O notebook captura falhas e pode entregar uma previsão parcial ou constante.",
        "Não existe evidência aqui de que 0.75 seja atingível com os dados e representações atuais.",
    ]
    resultado["hipoteses_gap_cv_lb"] = [
        "Mudança de distribuição de prompts/estilos ou prevalência intra-classe entre treino e teste público.",
        "Otimismo de seleção de camadas/encoders/ensemble na própria validação.",
        "Variação amostral do subconjunto público; tamanho efetivo por classe desconhecido.",
        "Algum encoder pode ter falhado na reexecução oculta de v2; não há log para confirmar ou excluir.",
    ]
    resultado["recomendacoes"] = [
        "Manter um teste por prompts intocado e escolher camadas/hiperparâmetros somente em folds internos; se todos os prompts já orientaram decisões, usar nested CV.",
        "Exigir concordância entre cache e manifesto de IDs/classes/hash/configuração antes de reutilizar OOF e embeddings.",
        "Medir variantes por família e pelo ranking verdadeiro; usar ensemble apenas se melhorar em validação externa, não apenas nas dobras usadas na seleção.",
        "Investigar novas representações ou ajuste supervisionado se a oracle top1 dos componentes ficar muito abaixo de 0.75; trocar média por pesos não garante esse salto.",
        "Avaliar v3 somente após ok de nova submissão; seu TF-IDF compatível resolve um componente, mas seu ganho CV mede poucos milésimos.",
    ]
    resultado["segundos"] = time.monotonic() - inicio
    destino = runs / "auditoria_pesada.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"arquivo": str(destino), "segundos": resultado["segundos"], "ensemble": resultado["ensemble"],
                      "reproducao_bge_oof": resultado["reproducao_bge_oof"], "controles": resultado["controles"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
