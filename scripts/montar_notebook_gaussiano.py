"""Monta o candidato robusto offline, separado de notebook/; não publica nada.

O notebook inclui as duas fontes Gaussian e todos os hashes esperados. Não
precisa de pacote local, train.csv, sample_submission ou classificadores pickle.
"""
import ast
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "notebook_gaussiano"
NOME = "name-that-llm-submissao"
DATASETS = ["juanseveriano/name-that-llm-encoders", "juanseveriano/name-that-llm-gaussiano"]
ESPECIFICACOES = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", [2, 4, 6]),
    "roberta-base": ("FacebookAI/roberta-base", [2, 4, 6, 8, 10, 12]),
    "deberta-v3-base": ("microsoft/deberta-v3-base", [4, 8, 10, 11, 12]),
    "Qwen3-0.6B": ("Qwen/Qwen3-0.6B", [0, 4, 6, 18, 20]),
    "StyleDistance_L12": ("StyleDistance/styledistance", [12]),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def fontes():
    puro = (RAIZ / "src/name_that_llm/gaussiano.py").read_text()
    robusto = (RAIZ / "src/name_that_llm/gaussiano_robusto.py").read_text()
    linhas = robusto.splitlines(keepends=True)
    importacoes = [n for n in ast.parse(robusto).body
                   if isinstance(n, ast.ImportFrom) and n.module == "name_that_llm"]
    assert len(importacoes) == 1
    n = importacoes[0]
    robusto = "".join(linhas[:n.lineno - 1] + linhas[n.end_lineno:])
    assert "from name_that_llm" not in robusto
    return puro, robusto


def manifesto():
    modelos = {}
    nomes = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.json", "merges.txt", "vocab.txt", "spm.model", "model.safetensors.index.json"}
    for nome, (backbone, camadas) in ESPECIFICACOES.items():
        pasta = RAIZ / "data/modelos" / ("styledistance" if nome == "StyleDistance_L12" else nome)
        arquivos = sorted(p for p in pasta.iterdir() if p.is_file()
                          and (p.name in nomes or p.name.endswith(".safetensors")))
        assert any(p.name.endswith(".safetensors") for p in arquivos)
        assert (pasta / "tokenizer_config.json").exists()
        config = json.loads((pasta / "config.json").read_text())
        modelos[nome] = {"backbone": backbone, "camadas": camadas,
                         "hidden_size": config["hidden_size"],
                         "arquivos": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in arquivos}}
    robusto = RAIZ / "runs/gaussiano_candidato/robusto"
    arquivos = sorted(p for p in robusto.rglob("*") if p.is_file())
    manifesto_dataset = RAIZ / "data/kaggle_gaussiano/manifesto.json"
    origem = json.loads(manifesto_dataset.read_text())
    assert origem["format_version"] == 1 and origem["candidate"] == "gaussian5_robust"
    for p in arquivos:
        assert origem["files"]["robusto/" + p.relative_to(robusto).as_posix()] == {"sha256": sha(p), "bytes": p.stat().st_size}
    for nome, spec in modelos.items():
        if nome == "StyleDistance_L12":
            assert all(origem["files"]["styledistance/" + n] == s for n, s in spec["arquivos"].items())
        else:
            assert origem["encoders"][nome]["files"] == spec["arquivos"]
    return {"versao": 1, "modelos": modelos,
            "artefatos": {p.relative_to(robusto).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in arquivos},
            "dataset_manifesto": {"sha256": sha(manifesto_dataset), "bytes": manifesto_dataset.stat().st_size,
                                   "candidate": "gaussian5_robust", "format_version": 1},
            "representacao": {"max_length": 512, "dtype_modelo": "float32", "dtype_features": "float16",
                              "pooling": "hidden_states masked mean; todos tokens válidos incluindo especiais; sem L2",
                              "semantica": "bge-small-en-v1.5 hidden_states[12]", "tokens_especiais": "padrão do tokenizer",
                              "eos_manual": False, "camadas_truncadas": False, "autocast": False},
            "fontes_sha256": {nome: sha(RAIZ / "src/name_that_llm" / nome)
                               for nome in ("gaussiano.py", "gaussiano_robusto.py")}}


INTRO = """# Name That LLM — Gaussian robusto, cinco vistas

Inferência offline com BGE-small, RoBERTa-base, DeBERTa-v3-base, Qwen3-0.6B e
StyleDistance. Os parâmetros ajustados e o TF-IDF portátil são arrays NumPy/JSON.
Grupos inferidos com menos de quatro textos recebem a média TF-IDF + quatro LR;
os demais recebem o Gaussian de cinco vistas. Não há quotas de classes.

O agrupamento e a inferência Gaussian recebem o conjunto de teste completo.
Somente os forwards dos encoders são divididos em lotes. Os campos utilizados
do teste são `id` e `text`. A execução valida hashes e falha claramente se um
componente estiver ausente ou inválido.

Dados privados: `juanseveriano/name-that-llm-encoders` e
`juanseveriano/name-that-llm-gaussiano`. Sem internet e sem instalação de pacotes.
"""

RUNTIME = r'''import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def sha_arquivo(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def salvar_json(path, valor):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(valor, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def conferir_arquivos(pasta, arquivos):
    for nome, esperado in arquivos.items():
        p = pasta / nome
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo obrigatório ausente: {p}")
        if p.stat().st_size != esperado["bytes"] or sha_arquivo(p) != esperado["sha256"]:
            raise ValueError(f"SHA256/tamanho inválido: {p}")


def achar_teste(entrada):
    candidatos = []
    for p in sorted(entrada.rglob("test.csv")):
        colunas = pd.read_csv(p, nrows=0).columns
        if {"id", "text"} <= set(colunas):
            candidatos.append(p)
    if len(candidatos) != 1:
        raise ValueError(f"Esperado um test.csv com id/text; encontrados {len(candidatos)}: {candidatos}")
    return candidatos[0]


def achar_artefatos(entrada):
    esperado = PACOTE["artefatos"]["metadados.json"]["sha256"]
    candidatos = [p.parent for p in sorted(entrada.rglob("metadados.json"))
                  if p.parent.joinpath("fallback.npz").is_file() and sha_arquivo(p) == esperado]
    if len(candidatos) != 1:
        raise ValueError(f"Esperado um pacote robusto compatível; encontrados {len(candidatos)}")
    conferir_arquivos(candidatos[0], PACOTE["artefatos"])
    raiz_dataset = candidatos[0].parent
    conferir_arquivos(raiz_dataset, {"manifesto.json": PACOTE["dataset_manifesto"]})
    manifesto = json.loads((raiz_dataset / "manifesto.json").read_text())
    if manifesto.get("format_version") != 1 or manifesto.get("candidate") != "gaussian5_robust":
        raise ValueError("Schema/candidato do manifesto do dataset incompatível")
    for rel in manifesto["files"]:
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("Path relativo inválido no manifesto do dataset")
    conferir_arquivos(raiz_dataset, manifesto["files"])
    return candidatos[0]


def achar_modelo(entrada, nome):
    spec = PACOTE["modelos"][nome]
    esperado = spec["arquivos"]["config.json"]["sha256"]
    candidatos = []
    for p in sorted(entrada.rglob("config.json")):
        if p.stat().st_size == spec["arquivos"]["config.json"]["bytes"] and sha_arquivo(p) == esperado:
            # RoBERTa e StyleDistance podem compartilhar a arquitetura, mas não os pesos.
            if all((p.parent / n).is_file() and (p.parent / n).stat().st_size == s["bytes"]
                   for n, s in spec["arquivos"].items()):
                candidatos.append(p.parent)
    validos = []
    for pasta in candidatos:
        try:
            conferir_arquivos(pasta, spec["arquivos"])
        except ValueError:
            continue
        validos.append(pasta)
    if len(validos) != 1:
        raise ValueError(f"Esperado um encoder {nome} com SHA256 compatível; encontrados {len(validos)}")
    return validos[0]


def extrair_vista(caminho, textos, camadas, *, dispositivo, lote=16, semantica=False):
    inicio = time.monotonic()
    tok = AutoTokenizer.from_pretrained(caminho, local_files_only=True, trust_remote_code=False)
    enc = AutoModel.from_pretrained(caminho, local_files_only=True, trust_remote_code=False,
                                    torch_dtype=torch.float32).float().to(dispositivo).eval()
    if any(p.dtype != torch.float32 for p in enc.parameters() if p.is_floating_point()):
        raise ValueError("Encoder precisa estar em float32")
    ordem = np.argsort([len(t) for t in textos])
    camadas_necessarias = sorted(set(camadas) | ({12} if semantica else set()))
    dim = enc.config.hidden_size
    features = np.empty((len(textos), len(camadas) * dim), dtype=np.float16)
    sem = np.empty((len(textos), dim), dtype=np.float16) if semantica else None
    truncados = 0
    comprimentos = tok(list(textos), add_special_tokens=True, truncation=False)["input_ids"]
    truncados = sum(len(x) > 512 for x in comprimentos)
    del comprimentos
    with torch.inference_mode():
        for inicio_lote in range(0, len(textos), lote):
            idx = ordem[inicio_lote:inicio_lote + lote]
            b = tok([textos[j] for j in idx], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(dispositivo)
            states = enc(**b, output_hidden_states=True).hidden_states
            # Mesma média mascarada dos caches: inclui todos tokens especiais reais.
            hidden = torch.stack([states[c] for c in camadas_necessarias], dim=1)
            mask = b["attention_mask"][:, None, :, None].to(hidden.dtype)
            means = ((hidden * mask).sum(2) / mask.sum(2)).float().cpu().numpy().astype(np.float16)
            features[idx] = np.concatenate([means[:, camadas_necessarias.index(c)] for c in camadas], axis=1)
            if semantica:
                sem[idx] = means[:, camadas_necessarias.index(12)]
            del b, states, hidden, mask, means
    if not np.isfinite(features).all() or (sem is not None and not np.isfinite(sem).all()):
        raise ValueError(f"Features não finitas: {caminho}")
    diagnostico = {"caminho": str(caminho), "camadas": list(camadas), "shape": list(features.shape),
                   "dtype_modelo": "float32", "dtype_features": str(features.dtype), "max_length": 512,
                   "textos_truncados": truncados, "lote": lote, "segundos": time.monotonic() - inicio,
                   "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                   "semantica_sha256": hashlib.sha256(sem.tobytes()).hexdigest() if sem is not None else None}
    del enc, tok
    gc.collect()
    if dispositivo == "cuda":
        torch.cuda.empty_cache()
    elif dispositivo == "mps":
        torch.mps.empty_cache()
    return features, sem, diagnostico


def proibido_ajustar(*args, **kwargs):
    raise RuntimeError("Ajuste de classificadores proibido na inferência")


def executar(entrada=Path("/kaggle/input"), saida=Path("/kaggle/working"), dispositivo=None, lote=16):
    entrada, saida = Path(entrada), Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    if lote < 1:
        raise ValueError("Lote precisa ser positivo")
    if dispositivo is None:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if dispositivo == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    inicio = time.monotonic()
    versoes = {n: importlib.metadata.version(n) for n in
               ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "pyarrow")}
    try:
        versoes["sentencepiece"] = importlib.metadata.version("sentencepiece")
    except importlib.metadata.PackageNotFoundError:
        # Os tokenizers rápidos exportados usam tokenizer.json sem essa dependência.
        versoes["sentencepiece"] = "ausente (tokenizer.json rápido)"
    log = {"versoes": versoes, "python": platform.python_version(), "dispositivo": dispositivo,
           "pacote": PACOTE, "componentes": {}, "offline": True, "refit_classificadores": False,
           "batching": "somente encoders; Gaussian e agrupamento recebem pool inteiro"}
    salvar_json(saida / "diagnostico.json", log)
    print("Ambiente:", json.dumps({"versoes": versoes, "dispositivo": dispositivo}), flush=True)
    teste_path = achar_teste(entrada)
    teste = pd.read_csv(teste_path, usecols=["id", "text"], dtype={"id": str, "text": str}, keep_default_na=False)
    ids, textos = g._ids_textos(teste)
    if not len(ids):
        raise ValueError("test.csv sem linhas")
    log.update({"test_path": str(teste_path), "test_sha256": sha_arquivo(teste_path), "linhas": len(ids),
                "ids_sha256": g.checksum_ids(ids), "ids_completos": ids.tolist()})
    pasta_artefatos = achar_artefatos(entrada)
    modelo = robusto.carregar(pasta_artefatos)
    if len(modelo.gaussiano.classes) != 18:
        raise ValueError("Esperadas as 18 classes do candidato")
    # A presença das funções de treino na fonte embutida não permite usá-las aqui.
    g.fit = g._ajustar_vista = robusto.fit = proibido_ajustar
    vistas, semantica = {}, None
    for nome, spec in g.VISTAS.items():
        print("Encoder:", nome, "camadas", spec["camadas"], flush=True)
        caminho = achar_modelo(entrada, nome)
        X, sem, info = extrair_vista(caminho, textos, spec["camadas"], dispositivo=dispositivo,
                                     lote=lote, semantica=(nome == "bge-small-en-v1.5"))
        esperado = len(modelo.gaussiano.parametros[nome]["media"])
        if X.shape != (len(ids), esperado) or X.dtype != np.float16 or not np.isfinite(X).all():
            raise ValueError(f"Contrato de features inválido: {nome}")
        vistas[nome] = X
        if sem is not None:
            semantica = sem
        log["componentes"][nome] = info
        salvar_json(saida / "diagnostico.json", log)
        print("Encoder pronto:", nome, json.dumps(info), flush=True)
    if semantica is None or semantica.shape != (len(ids), modelo.gaussiano.metadados["dimensao_semantica"]):
        raise ValueError("Semântica BGE12 ausente/incompatível")
    matrizes = g.Matrizes(ids, vistas, semantica)
    # Uma chamada, sobre TODAS as linhas; nenhuma partição Gaussian em minibatches.
    previsao = robusto.prever(modelo, teste, matrizes)
    p = previsao.probabilidades
    sub = previsao.submissao
    if sub.columns.tolist() != ["id", "model"] or sub.id.tolist() != ids.tolist() or not sub.id.is_unique:
        raise ValueError("IDs/ordem/colunas da submissão inconsistentes")
    if len(sub) != len(ids) or not set(sub.model) <= set(modelo.gaussiano.classes):
        raise ValueError("Submissão incompleta ou classes inesperadas")
    if p.shape != (len(ids), 18) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError("Probabilidades inválidas")
    tmp = saida / "submission.tmp.parquet"
    sub.to_parquet(tmp, index=False)
    tmp.replace(saida / "submission.parquet")
    reaberto = pd.read_parquet(saida / "submission.parquet")
    pd.testing.assert_frame_equal(sub, reaberto)
    np.savez_compressed(saida / "probabilidades.npz", ids=ids, classes=modelo.gaussiano.classes,
                        probabilidades=p, grupos=previsao.grupos)
    _, inv, counts = np.unique(previsao.grupos, return_inverse=True, return_counts=True)
    pd.DataFrame({"id": ids, "grupo_inferido": previsao.grupos, "tamanho_grupo": counts[inv],
                  "componente": np.where(counts[inv] >= 4, "gaussiano5", "mean5")}).to_csv(saida / "roteamento.csv", index=False)
    log.update({"concluido": True, "segundos_total": time.monotonic() - inicio,
                "classes": modelo.gaussiano.classes.tolist(), "roteamento": previsao.diagnostico,
                "grupos_por_tamanho": {str(k): int((counts == k).sum()) for k in np.unique(counts)},
                "distribuicao_predita": sub.model.value_counts().to_dict(),
                "probas_finitas": True, "ids_completos_preservados": True,
                "submission_sha256": sha_arquivo(saida / "submission.parquet")})
    salvar_json(saida / "diagnostico.json", log)
    print("Final:", json.dumps({k: log[k] for k in ("linhas", "ids_sha256", "roteamento", "distribuicao_predita", "segundos_total")}), flush=True)
    return previsao
'''

EXECUCAO = '''try:
    resultado = executar()
except Exception as exc:
    print("FALHA EXPLÍCITA:", type(exc).__name__, str(exc), flush=True)
    traceback.print_exc()
    raise
'''


def celula(tipo, fonte):
    base = {"cell_type": tipo, "metadata": {}, "source": fonte.splitlines(keepends=True)}
    return base | ({"execution_count": None, "outputs": []} if tipo == "code" else {})


def notebook():
    puro, robusto = fontes()
    pacote = manifesto()
    codigo = "import sys\nimport types\n\n"
    codigo += "g = types.ModuleType('gaussiano_embutido')\nsys.modules[g.__name__] = g\n"
    codigo += f"exec({puro!r}, g.__dict__)\n\n"
    codigo += "robusto = types.ModuleType('gaussiano_robusto_embutido')\nsys.modules[robusto.__name__] = robusto\nrobusto.g = g\n"
    codigo += f"exec({robusto!r}, robusto.__dict__)\n\n"
    codigo += f"PACOTE = {pacote!r}\n"
    doc = {"cells": [celula("markdown", INTRO), celula("code", codigo), celula("code", RUNTIME), celula("code", EXECUCAO)],
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
           "nbformat": 4, "nbformat_minor": 4}
    return doc, pacote


def metadata():
    return {"id": f"juanseveriano/{NOME}", "title": NOME, "code_file": f"{NOME}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
            "enable_tpu": "false", "enable_internet": "false", "dataset_sources": DATASETS,
            "competition_sources": ["name-that-llm"], "kernel_sources": [], "model_sources": []}


def replay_cache():
    """Ensaio das células reais em 540 linhas, sem labels/novos forwards/refit.

    O extrator é substituído por features existentes; isto verifica o pacote e o
    ramo Gaussian em volume, e não mede acurácia nem valida novamente os encoders.
    """
    import os
    import sys
    import time
    for nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[nome] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import sklearn
    from threadpoolctl import threadpool_limits

    inicio = time.monotonic()
    destino = RAIZ / "runs/gaussiano_notebook_smoke" / ("cache540_sklearn_" + sklearn.__version__)
    entrada = destino / "input"
    saida = destino / "output"
    competencia = entrada / "competition/name-that-llm"
    competencia.mkdir(parents=True, exist_ok=True)
    nbpath = SAIDA / f"{NOME}.ipynb"
    doc = json.loads(nbpath.read_text())
    ns = {}
    for i in (1, 2):
        exec(compile("".join(doc["cells"][i]["source"]), f"notebook_gaussiano.cell{i}", "exec"), ns)
    pacote = ns["PACOTE"]
    split = np.load(RAIZ / "runs/experimentos_estilo/split.npz", allow_pickle=False)
    indices = split["holdout"]
    assert len(indices) == 540
    teste = pd.read_csv(RAIZ / "data/train.csv", usecols=["id", "text"], keep_default_na=False).iloc[indices].reset_index(drop=True)
    teste.to_csv(competencia / "test.csv", index=False)
    extra = entrada / "datasets/juanseveriano/name-that-llm-gaussiano"
    mpath = RAIZ / "data/kaggle_gaussiano/manifesto.json"
    manifest = json.loads(mpath.read_text())
    for rel in [*manifest["files"], "manifesto.json"]:
        p = extra / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.symlink_to(RAIZ / "data/kaggle_gaussiano" / rel)
    vistas = {}
    for nome, spec in pacote["modelos"].items():
        if nome == "StyleDistance_L12":
            style = np.load(RAIZ / "runs/experimentos_styledistance/train_embeddings.npy", mmap_mode="r", allow_pickle=False)
            vistas[nome] = np.asarray(style[indices, 12], dtype=np.float16)
            continue
        pasta = entrada / "datasets/juanseveriano/name-that-llm-encoders" / nome
        pasta.mkdir(parents=True, exist_ok=True)
        for rel in spec["arquivos"]:
            p = pasta / rel
            if not p.exists():
                p.symlink_to(RAIZ / "data/modelos" / nome / rel)
        vistas[nome] = np.asarray(np.load(RAIZ / "data/artefatos" / f"emb_{nome}.npy", mmap_mode="r", allow_pickle=False)[indices], dtype=np.float16)
    bge = np.load(RAIZ / "data/emb/BAAI__bge-small-en-v1.5.treino.npy", mmap_mode="r", allow_pickle=False)
    semantica = np.asarray(bge[indices, 12], dtype=np.float16)
    chamadas = []

    def extrator_cache(caminho, textos, camadas, *, dispositivo, lote=16, semantica=False):
        nome = "StyleDistance_L12" if caminho.name == "styledistance" else caminho.name
        assert textos == teste.text.tolist()
        assert list(camadas) == pacote["modelos"][nome]["camadas"]
        chamadas.append(nome)
        return vistas[nome], sem_bge if semantica else None, {"fonte": "cache replay", "novos_forwards": 0, "shape": list(vistas[nome].shape)}

    sem_bge = semantica
    ns["extrair_vista"] = extrator_cache
    # Mesmo o dispositivo CPU fica bloqueado contra carregamento acidental de pesos.
    ns["AutoModel"].from_pretrained = ns["AutoTokenizer"].from_pretrained = ns["proibido_ajustar"]
    sys.path.insert(0, str(RAIZ / "src"))
    from name_that_llm import gaussiano as api_g
    from name_that_llm import gaussiano_robusto as api_r
    api_modelo = api_r.carregar(RAIZ / "runs/gaussiano_candidato/robusto")
    with threadpool_limits(limits=1):
        real = ns["executar"](entrada, saida, dispositivo="cpu", lote=16)
        ref = api_r.prever(api_modelo, teste, api_g.Matrizes(teste.id.to_numpy(dtype=str), vistas, semantica))
    np.testing.assert_array_equal(real.probabilidades, ref.probabilidades)
    np.testing.assert_array_equal(real.grupos, ref.grupos)
    pd.testing.assert_frame_equal(real.submissao, ref.submissao)
    pd.testing.assert_frame_equal(pd.read_parquet(saida / "submission.parquet"), ref.submissao)
    assert real.diagnostico == ref.diagnostico
    assert real.diagnostico["linhas_gaussian"] > 0 and len(chamadas) == 5
    rota = pd.read_csv(saida / "roteamento.csv")
    _, inverso, tamanhos = np.unique(ref.grupos, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(rota.tamanho_grupo, tamanhos[inverso])
    np.testing.assert_array_equal(rota.componente, np.where(tamanhos[inverso] >= 4, "gaussiano5", "mean5"))
    resultado = {"tipo": "cache replay das células reais, não novo ensaio GPU/acurácia", "sklearn": sklearn.__version__,
                 "linhas": len(teste), "indices_origem": indices.tolist(), "nb_sha256": sha(nbpath),
                 "ids_sha256": api_g.checksum_ids(teste.id), "id_text_apenas": True, "labels_lidos": False,
                 "novos_forwards": 0, "gpu": False, "refit": False, "acuracia_medida": False,
                 "maxdiff_probabilidades_vs_api": float(np.max(np.abs(real.probabilidades - ref.probabilidades))),
                 "grupos_iguais": True, "rotas_iguais": True, "parquet_igual_api": True,
                 "diagnostico": real.diagnostico, "segundos": time.monotonic() - inicio}
    (destino / "verificacao.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in resultado.items() if k != "indices_origem"}, ensure_ascii=False), flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-cache", action="store_true", help="Testa células com 540 linhas/embeddings existentes, sem labels/GPU")
    args = p.parse_args()
    if args.replay_cache:
        replay_cache()
        return
    SAIDA.mkdir(exist_ok=True)
    doc, pacote = notebook()
    (SAIDA / f"{NOME}.ipynb").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    (SAIDA / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    (SAIDA / "manifesto_pacote.json").write_text(json.dumps(pacote, ensure_ascii=False, indent=2) + "\n")
    print(f"Escrito: {SAIDA}; nenhum upload ou submissão.")


if __name__ == "__main__":
    main()
