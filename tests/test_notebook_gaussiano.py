"""Packaged Gaussian cells: CPU fixtures, no network or real encoder weights."""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import montar_notebook_gaussiano as builder
from name_that_llm import artefatos, gaussiano as g, gaussiano_robusto as r
from name_that_llm.pipeline import montar_tfidf


def fingerprints(directory):
    return {p.relative_to(directory).as_posix(): {
        "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
    } for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def synthetic_artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gaussian_notebook_artifacts")
    rng = np.random.default_rng(71)
    groups, y = np.repeat(np.arange(8), 18), np.tile(np.arange(18), 8)
    topics = ["pizza tomato dough oven", "planet orbit telescope moon", "ocean whale salt waves",
              "garden leaves root soil", "violin concert melody piano", "football goal match team",
              "winter snow ice mountain", "computer memory chip software"]
    df = pd.DataFrame({
        "id": [f"training-{i:04d}" for i in range(len(y))],
        "text": [topics[p] + f" authorstyle{c:02d}" for p, c in zip(groups, y)],
        "prompt_id": [f"p{p}" for p in groups], "model": [f"m{c:02d}" for c in y],
    })
    views = {}
    for name, spec in g.VISTAS.items():
        dimension = 8 * len(spec["camadas"])
        author, prompt = rng.normal(size=(18, dimension)) * 3, rng.normal(size=(8, dimension)) * 2
        views[name] = (author[y] + prompt[groups] + rng.normal(size=(len(y), dimension)) * .03).astype(np.float16).astype(np.float32)
    semantic = (np.eye(8)[groups] + rng.normal(size=(len(y), 8)) * .0001).astype(np.float16).astype(np.float32)
    matrices = g.Matrizes(df.id.to_numpy(dtype=str), views, semantic)
    gaussian = g.fit(df, matrices)
    tfidf = directory / "tfidf"
    tfidf.mkdir()
    pipe = montar_tfidf(LogisticRegression(C=20, max_iter=2000)).fit(df, df.model)
    artefatos.exportar_tfidf(pipe, tfidf)
    (tfidf / "rotulos.json").write_text(json.dumps(df.model.tolist()))
    robust = r.fit(df, matrices, gaussian, tfidf)
    r.salvar(robust, directory / "robusto")
    return SimpleNamespace(df=df, matrices=matrices, robust=robust, directory=directory)


@pytest.fixture
def package(tmp_path, synthetic_artifacts, monkeypatch):
    source = synthetic_artifacts
    entrada = tmp_path / "input"
    dataset = entrada / "datasets/fixture-user/name-that-llm-gaussiano/version-2"
    encoders = entrada / "datasets/fixture-user/name-that-llm-encoders/version-4"
    shutil.copytree(source.directory / "robusto", dataset / "robusto")
    models, names = {}, {}
    for name, spec in g.VISTAS.items():
        directory = dataset / "styledistance" if name == "StyleDistance_L12" else encoders / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps({"hidden_size": 8, "fixture": name}))
        (directory / "tokenizer_config.json").write_text(json.dumps({"fixture": name}))
        (directory / "tokenizer.json").write_text(json.dumps({"fixture": name}))
        (directory / "model.safetensors").write_bytes(name.encode())
        models[name] = {"backbone": spec["backbone"], "camadas": spec["camadas"],
                        "hidden_size": 8, "arquivos": fingerprints(directory)}
        names[str(directory)] = name
    files = {"robusto/" + name: value for name, value in fingerprints(dataset / "robusto").items()}
    files.update({"styledistance/" + name: value for name, value in fingerprints(dataset / "styledistance").items()})
    manifest_file = dataset / "manifesto.json"
    manifest_file.write_text(json.dumps({"format_version": 1, "candidate": "gaussian5_robust", "files": files}))
    pacote = {"versao": 1, "modelos": models, "artefatos": fingerprints(dataset / "robusto"),
              "dataset_manifesto": {"sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
                                    "bytes": manifest_file.stat().st_size, "candidate": "gaussian5_robust", "format_version": 1},
              "representacao": {"max_length": 512, "dtype_modelo": "float32", "dtype_features": "float16"},
              "fontes_sha256": {}}
    monkeypatch.setattr(builder, "manifesto", lambda: copy.deepcopy(pacote))
    doc, _ = builder.notebook()
    namespace = {}
    for cell in doc["cells"]:
        if cell["cell_type"] == "code" and "".join(cell["source"]) != builder.EXECUCAO:
            exec("".join(cell["source"]), namespace)
    lookup = {text: i for i, text in enumerate(source.df.text)}
    calls = []

    def fake_extract(path, texts, layers, *, dispositivo, lote=16, semantica=False):
        assert dispositivo == "cpu"
        name = names[str(path)]
        assert list(layers) == g.VISTAS[name]["camadas"]
        indices = [lookup[text] for text in texts]
        calls.append((name, len(texts)))
        return (source.matrices.vistas[name][indices].astype(np.float16),
                source.matrices.semantica[indices].astype(np.float16) if semantica else None,
                {"fixture": True, "name": name})

    namespace["extrair_vista"] = fake_extract
    return SimpleNamespace(entrada=entrada, saida=tmp_path / "working", dataset=dataset,
                           names=names, namespace=namespace, doc=doc, source=source, calls=calls)


def input_frame(package, k):
    df = package.source.df
    indices = np.concatenate([np.flatnonzero(df.prompt_id.to_numpy() == p)[:k] for p in df.prompt_id.unique()])
    frame = df.iloc[indices].copy()
    frame["id"] = "test-" + frame["id"]
    directory = package.entrada / "competitions/name-that-llm"
    directory.mkdir(parents=True)
    frame.to_csv(directory / "test.csv", index=False)
    return frame, indices, directory / "test.csv"


def run_cells(package):
    execute = package.namespace["executar"]
    package.namespace["executar"] = lambda: execute(package.entrada, package.saida, dispositivo="cpu", lote=2)
    exec(builder.EXECUCAO, package.namespace)
    return package.namespace["resultado"]


@pytest.mark.parametrize("k", [1, 3, 4])
def test_packaged_cells_without_train_whole_pool_and_route(package, k, monkeypatch):
    test, indices, _ = input_frame(package, k)
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **kw: pytest.fail("refit at inference"))
    calls = []
    original = package.namespace["robusto"].prever

    def count_pool(model, frame, matrices):
        calls.append(len(frame))
        assert list(frame.columns) == ["id", "text"]
        return original(model, frame, matrices)

    package.namespace["robusto"].prever = count_pool
    result = run_cells(package)
    expected_matrices = g.Matrizes(test.id.to_numpy(dtype=str),
        {name: values[indices] for name, values in package.source.matrices.vistas.items()},
        package.source.matrices.semantica[indices])
    if k < 4:
        expected = r.prever_baseline(package.source.robust, test[["id", "text"]], expected_matrices)
    else:
        expected = g.prever(package.source.robust.gaussiano, test[["id", "text"]], expected_matrices).probabilidades
    np.testing.assert_array_equal(result.probabilidades, expected)
    assert calls == [len(test)] and len(package.calls) == 5
    saved = pd.read_parquet(package.saida / "submission.parquet")
    assert saved.columns.tolist() == ["id", "model"] and saved.id.tolist() == test.id.tolist()
    assert set(saved.model) <= set(package.source.robust.gaussiano.classes)
    assert not list(package.entrada.rglob("train.csv")) and not list(package.entrada.rglob("sample_submission*"))
    diagnostic = json.loads((package.saida / "diagnostico.json").read_text())
    assert diagnostic["concluido"] and diagnostic["refit_classificadores"] is False
    routing = pd.read_csv(package.saida / "roteamento.csv")
    assert set(routing.componente) == ({"gaussiano5"} if k >= 4 else {"mean5"})


def test_missing_sentencepiece_metadata_is_optional(package, monkeypatch):
    input_frame(package, 1)
    original = importlib.metadata.version

    def version(name):
        if name == "sentencepiece":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    run_cells(package)
    diagnostic = json.loads((package.saida / "diagnostico.json").read_text())
    assert "ausente" in diagnostic["versoes"]["sentencepiece"]


@pytest.mark.parametrize("failure", ["encoder_hash", "artifact_hash", "duplicate_ids", "encoder_failure", "features_dtype"])
def test_failures_propagate_without_constant_submission(package, failure):
    test, _, path = input_frame(package, 1)
    if failure == "encoder_hash":
        folder = next(Path(folder) for folder, name in package.names.items() if name == "bge-small-en-v1.5")
        with (folder / "model.safetensors").open("ab") as stream:
            stream.write(b"bad")
        expected = "encoder bge-small"
    elif failure == "artifact_hash":
        with (package.dataset / "robusto/fallback.npz").open("ab") as stream:
            stream.write(b"bad")
        expected = "SHA256/tamanho"
    elif failure == "duplicate_ids":
        test.loc[test.index[1], "id"] = test.iloc[0].id
        test.to_csv(path, index=False)
        expected = "ids precisam"
    elif failure == "encoder_failure":
        def fail(*a, **kw):
            raise RuntimeError("fixture_encoder_failure")
        package.namespace["extrair_vista"] = fail
        expected = "fixture_encoder_failure"
    else:
        original = package.namespace["extrair_vista"]
        def wrong_dtype(*a, **kw):
            x, sem, info = original(*a, **kw)
            return x.astype(np.float32), sem, info
        package.namespace["extrair_vista"] = wrong_dtype
        expected = "Contrato de features"
    with pytest.raises((ValueError, RuntimeError), match=expected):
        run_cells(package)
    assert not (package.saida / "submission.parquet").exists()


def test_encoder_contract_cpu_inclusive_mean_then_float16(package, monkeypatch):
    namespace = package.namespace
    batches = []

    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            sequences = [[101, 8, 102] if text == "short" else [101, 7, 9, 102] for text in texts]
            assert kwargs.get("add_special_tokens", True) is True
            if kwargs.get("return_tensors") != "pt":
                assert kwargs["truncation"] is False
                return {"input_ids": sequences}
            batches.append(kwargs)
            size = max(map(len, sequences))
            return Batch(input_ids=torch.tensor([s + [0] * (size - len(s)) for s in sequences]),
                         attention_mask=torch.tensor([[1] * len(s) + [0] * (size - len(s)) for s in sequences]))

    class Encoder(nn.Module):
        config = SimpleNamespace(hidden_size=3)
        def forward(self, input_ids, attention_mask, output_hidden_states):
            assert output_hidden_states
            values = input_ids.float()
            states = tuple(torch.stack([values + layer, 2 * values + layer, values * 0 + layer], dim=-1)
                           for layer in range(13))
            return SimpleNamespace(hidden_states=states)

    def load_model(*a, **kwargs):
        assert kwargs["local_files_only"] and kwargs["trust_remote_code"] is False
        assert kwargs["torch_dtype"] == torch.float32
        return Encoder()

    monkeypatch.setattr(namespace["AutoModel"], "from_pretrained", load_model)
    monkeypatch.setattr(namespace["AutoTokenizer"], "from_pretrained", lambda *a, **kw: Tokenizer())
    separate = {"g": namespace["g"], "robusto": namespace["robusto"], "PACOTE": namespace["PACOTE"]}
    exec(builder.RUNTIME, separate)
    view, semantic, diagnostic = separate["extrair_vista"]("fixture", ["longer text", "short"], [2, 4], dispositivo="cpu", semantica=True)
    expected, expected_semantic = [], []
    for sequence in [[101, 7, 9, 102], [101, 8, 102]]:
        mean = np.mean(sequence, dtype=np.float32)
        expected.append([mean + 2, 2 * mean + 2, 2, mean + 4, 2 * mean + 4, 4])
        expected_semantic.append([mean + 12, 2 * mean + 12, 12])
    np.testing.assert_array_equal(view, np.asarray(expected, dtype=np.float16))
    np.testing.assert_array_equal(semantic, np.asarray(expected_semantic, dtype=np.float16))
    assert semantic.dtype == np.float16 and diagnostic["dtype_modelo"] == "float32"
    assert batches and all(b["max_length"] == 512 and b["padding"] and b["truncation"] for b in batches)

