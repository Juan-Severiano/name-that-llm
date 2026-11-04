"""Baixa o pool público com versão fixada e audita sem ler labels do holdout.

Só consulta Kaggle e baixa dados públicos. Não treina, envia ou submete nada.
O peso match_weight do publicador usa os 150 prompts oficiais; não deve ser
usado como peso de treino na avaliação dos 30 prompts reservados.
"""

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data/open_pool"
FONTE = "crtproduction/name-that-llm-open-pool"
LIMITE = 300_000_000
ARQUIVOS = (
    "ntl_open_pool.parquet", "prompts.parquet", "sources.csv", "per_model.csv",
    "removal_summary.csv", "dataq_results.csv", "dataq4_paired.csv", "typography_match.csv",
)


def salvar(nome, dados):
    (DESTINO / nome).write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n")


def normalizar(texto):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(texto)).casefold()).strip()


def sha256(caminho):
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def api_autenticada():
    for linha in (RAIZ / ".env").read_text().splitlines():
        if linha.strip() and not linha.lstrip().startswith("#"):
            k, v = linha.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def baixar():
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    api = api_autenticada()
    req = ApiGetDatasetRequest()
    req.owner_slug, req.dataset_slug = FONTE.split("/", 1)
    with api.build_kaggle_client() as k:
        meta = k.datasets.dataset_api_client.get_dataset(req).to_dict()
    versao = meta["currentVersionNumber"]
    fonte = f"{FONTE}/{versao}"
    files = api.dataset_list_files(fonte, page_size=100).to_dict()["datasetFiles"]
    tamanhos = {f["name"]: f["totalBytes"] for f in files}
    total = sum(tamanhos[n] for n in ARQUIVOS)
    assert total <= LIMITE, (total, LIMITE)
    manifesto = {
        "consultado_em": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(),
        "fonte": FONTE, "url": f"https://www.kaggle.com/datasets/{FONTE}",
        "versao_kaggle": versao, "versao_editorial": "v12", "titulo": meta["title"],
        "licenca_dataset": meta["licenseName"], "atualizado_em": meta["lastUpdated"],
        "total_dataset_bytes": meta["totalBytes"], "bytes_planejados": total,
        "limite_download_bytes": LIMITE, "arquivos": [],
        "alertas": [
            "Dataset comunitário, não é material criado/garantido pelos organizadores.",
            "match_weight usa todos os prompts oficiais: não usar esse peso pronto em CV/holdout.",
            "Filtragem de overlaps não prova ausência de toda paráfrase ou reconstrução do prompt.",
        ],
    }
    salvar("manifesto.json", manifesto)
    for nome in ARQUIVOS:
        caminho = DESTINO / nome
        if not caminho.exists():
            api.dataset_download_file(fonte, nome, path=str(DESTINO), quiet=True, unzip=True)
        assert caminho.stat().st_size == tamanhos[nome], (nome, caminho.stat().st_size, tamanhos[nome])
        manifesto["arquivos"].append({"nome": nome, "bytes": caminho.stat().st_size, "sha256": sha256(caminho)})
        salvar("manifesto.json", manifesto)
        print(f"BAIXADO {nome}: {caminho.stat().st_size} bytes", flush=True)
    return manifesto


def auditar():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    schema = pq.read_schema(DESTINO / "ntl_open_pool.parquet")
    colunas = schema.names
    essenciais = [c for c in ("id", "text", "model", "prompt_canon", "prompt_group", "source_dataset", "source_license", "source_url", "match_weight") if c in colunas]
    essenciais += [c for c in colunas if c.startswith("flag_")]
    pool = pd.read_parquet(DESTINO / "ntl_open_pool.parquet", columns=essenciais)
    # Não ler a coluna model do treino oficial nesta auditoria.
    original = pd.read_csv(RAIZ / "data/train.csv", usecols=["id", "prompt_id", "text"], keep_default_na=False)
    split = np.load(RAIZ / "runs/experimentos_estilo/split.npz", allow_pickle=False)
    holdout = original.iloc[split["holdout"]]
    classes_permitidas = set(split["classes"].tolist())
    textos_orig = set(original.text)
    norm_orig = {normalizar(t) for t in original.text}
    norm_hold = {normalizar(t) for t in holdout.text}
    exato = pool.text.isin(textos_orig).to_numpy()
    normalizado = np.fromiter((normalizar(t) in norm_orig for t in pool.text), dtype=bool, count=len(pool))
    hold_norm = np.fromiter((normalizar(t) in norm_hold for t in pool.text), dtype=bool, count=len(pool))
    invalidos = ~pool.model.isin(classes_permitidas).to_numpy()
    suspeitos = exato | normalizado | invalidos
    flags = {}
    for c in essenciais:
        if c.startswith("flag_"):
            flags[c] = pool[c].value_counts(dropna=False).astype(int).to_dict()
            if "train_regen" in c or "train_prompt" in c:
                suspeitos |= pool[c].fillna(False).astype(bool).to_numpy()
    auditoria = {
        "linhas": len(pool), "schema": {f.name: str(f.type) for f in schema},
        "classes": sorted(pool.model.dropna().unique()), "classes_compativeis_18": set(pool.model.dropna().unique()) == classes_permitidas,
        "por_modelo": pool.model.value_counts().astype(int).to_dict(),
        "rotulos_invalidos": int(invalidos.sum()), "textos_nulos": int(pool.text.isna().sum()),
        "ids_duplicados": int(pool.id.duplicated().sum()) if "id" in pool else None,
        "overlap_texto_exato_treino_oficial": int(exato.sum()),
        "overlap_normalizado_treino_oficial": int(normalizado.sum()),
        "overlap_normalizado_holdout_540_textos": int(hold_norm.sum()),
        "holdout_prompts_reservados": sorted(holdout.prompt_id.unique()),
        "coluna_model_oficial_lida": False,
        "colunas_proveniencia": [c for c in colunas if any(k in c for k in ("source", "provenance", "route", "prompt", "license"))],
        "flags": flags, "excluidos_auditoria_inicial": int(suspeitos.sum()),
        "limite": "Ainda falta comparar os textos dos prompts recuperados; ausência de cópias não prova prompts diferentes.",
        "match_weight_utilizado": False,
    }
    if "source_license" in pool:
        auditoria["licencas_fontes"] = pool.source_license.fillna("<ausente>").value_counts().astype(int).to_dict()
    if "source_dataset" in pool:
        auditoria["n_fontes"] = int(pool.source_dataset.nunique())
    salvar("auditoria_inicial.json", auditoria)
    np.save(DESTINO / "excluir_auditoria_inicial.npy", suspeitos)
    print(json.dumps({k: auditoria[k] for k in ("linhas", "classes_compativeis_18", "rotulos_invalidos", "overlap_normalizado_treino_oficial", "overlap_normalizado_holdout_540_textos", "excluidos_auditoria_inicial")}, ensure_ascii=False), flush=True)


def preparar_conservador():
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    api = api_autenticada()
    fonte_reconstruida = "pjmathematician/ntl-train-prompts-reconstructed"
    pasta = DESTINO / "reconstructed"
    pasta.mkdir(exist_ok=True)
    req = ApiGetDatasetRequest()
    req.owner_slug, req.dataset_slug = fonte_reconstruida.split("/", 1)
    with api.build_kaggle_client() as k:
        meta = k.datasets.dataset_api_client.get_dataset(req).to_dict()
    arq = pasta / "train_prompts_reconstructed.csv"
    if not arq.exists():
        api.dataset_download_file(f"{fonte_reconstruida}/{meta['currentVersionNumber']}", arq.name, path=str(pasta), quiet=True, unzip=True)
    (pasta / "source.json").write_text(json.dumps({k: meta.get(k) for k in ("ref", "title", "currentVersionNumber", "licenseName", "lastUpdated")}, indent=2) + "\n")
    recon = pd.read_csv(arq, keep_default_na=False)
    prompts = pd.read_parquet(DESTINO / "prompts.parquet")
    # Prompts são inferências públicas, e não as instruções originais do host.
    texto_recon = recon.inferred_prompt.astype(str).tolist()
    texto_pool = prompts.prompt_text.fillna("").astype(str).tolist()
    vet = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000, sublinear_tf=True, dtype=np.float32)
    mat = vet.fit_transform(texto_recon + texto_pool)
    sims = linear_kernel(mat[len(recon):], mat[:len(recon)])
    top = sims.argmax(1)
    melhor = sims[np.arange(len(prompts)), top]
    norm_oficiais = {normalizar(t) for t in texto_recon}
    exato_prompt = np.array([normalizar(t) in norm_oficiais for t in texto_pool])
    # Limiar declarado antes do treinamento; filtro conservador de contaminação,
    # sem ajustar por label/score dos 30 prompts reservados.
    limiar = .35
    suspeita_prompt = exato_prompt | (melhor >= limiar)
    split = np.load(RAIZ / "runs/experimentos_estilo/split.npz", allow_pickle=False)
    oficial = pd.read_csv(RAIZ / "data/train.csv", usecols=["prompt_id"], keep_default_na=False)
    holdout_prompts = set(oficial.prompt_id.iloc[split["holdout"]])
    evidencia = pd.DataFrame({"prompt_canon": prompts.prompt_canon,
                              "prompt_id_reconstruido": recon.prompt_id.to_numpy()[top],
                              "cosseno_tfidf": melhor,
                              "prompt_exato_normalizado": exato_prompt,
                              "suspeita_por_limiar": suspeita_prompt})
    evidencia["prompt_reservado_holdout"] = evidencia.prompt_id_reconstruido.isin(holdout_prompts)
    evidencia.to_csv(DESTINO / "comparacao_prompts.csv", index=False)
    prompts_suspeitos = set(prompts.prompt_canon[suspeita_prompt])
    colunas = ["id", "text", "model", "prompt_canon", "source_dataset", "source_file", "source_license", "origin_hint", "prompt_evidence",
               "flag_kb_proxy", "flag_gemini37_routing_risk", "flag_near_train_prompt", "flag_train_prompt_regen"]
    pool = pd.read_parquet(DESTINO / "ntl_open_pool.parquet", columns=colunas)
    motivos = {
        "auditoria_inicial": np.load(DESTINO / "excluir_auditoria_inicial.npy"),
        "kaggle_benchmarks_proxy": pool.flag_kb_proxy.to_numpy(),
        "risco_roteamento": pool.flag_gemini37_routing_risk.to_numpy(),
        "prompt_treino_sinalizado_publicador": (pool.flag_near_train_prompt | pool.flag_train_prompt_regen).to_numpy(),
        "prompt_sem_identificacao": ~pool.prompt_canon.isin(set(prompts.prompt_canon)).to_numpy(),
        "prompt_similar_ao_oficial_reconstruido": pool.prompt_canon.isin(prompts_suspeitos).to_numpy(),
        "texto_ausente": (pool.text.isna() | pool.text.str.strip().eq("")).to_numpy(),
    }
    texto_hash = pd.Series((hashlib.sha256(normalizar(t).encode()).hexdigest() for t in pool.text), index=pool.index)
    motivos["texto_normalizado_duplicado"] = texto_hash.duplicated(keep="first").to_numpy()
    excluir = np.any(np.stack(list(motivos.values())), axis=0)
    indices = np.flatnonzero(~excluir)
    np.save(DESTINO / "indices_conservadores.npy", indices)
    treino = pool.iloc[indices][[c for c in colunas if not c.startswith("flag_")]]
    treino.to_parquet(DESTINO / "pool_conservador.parquet", index=False)
    treino.groupby(["source_dataset", "model"], dropna=False).size().rename("passagens").to_csv(DESTINO / "conservador_fonte_classe.csv")
    auditoria = {
        "linhas_originais": len(pool), "linhas_conservadoras": len(treino),
        "por_modelo": treino.model.value_counts().astype(int).to_dict(),
        "classes_18_preservadas": set(treino.model) == set(split["classes"]),
        "fontes": int(treino.source_dataset.nunique()),
        "licencas_fontes": treino.source_license.value_counts().astype(int).to_dict(),
        "motivos_podem_sobrepor": {k: int(v.sum()) for k, v in motivos.items()},
        "prompt_limiar_coseno_tfidf": limiar,
        "prompts_pool_comparados": len(prompts), "prompts_reconstruidos": len(recon),
        "prompts_suspeitos_todos_oficiais": int(suspeita_prompt.sum()),
        "prompts_suspeitos_holdout": int((suspeita_prompt & evidencia.prompt_reservado_holdout.to_numpy()).sum()),
        "prompt_similaridade_quantiles": {str(q): float(np.quantile(melhor, q)) for q in (0, .5, .9, .99, 1)},
        "rotulos_oficiais_do_holdout_usados": False,
        "metadados_ou_match_weight_como_features": False,
        "arquivo_treino": "pool_conservador.parquet", "sha256_arquivo_treino": sha256(DESTINO / "pool_conservador.parquet"),
        "fonte_reconstrucoes": fonte_reconstruida, "reconstrucoes_sha256": sha256(arq),
        "limites": ["Prompts reconstruídos são hipóteses por LLM; matching lexical não exclui toda paráfrase.",
                    "Exclusão de proxy é uma escolha de proveniência, não evidência de melhor desempenho.",
                    "Labels e proveniência vêm dos publicadores; não há validação independente de todas as rotas.",
                    "match_weight e scores do publicador não estão no arquivo de treino conservador."],
    }
    salvar("auditoria_conservadora.json", auditoria)
    print(json.dumps({k: auditoria[k] for k in ("linhas_conservadoras", "classes_18_preservadas", "fontes", "motivos_podem_sobrepor", "prompts_suspeitos_holdout")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    baixar()
    auditar()
    preparar_conservador()
