import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from name_that_llm import artefatos, pipeline

CLASSES = ["m-a", "m-b", "m-c"]
FALSOS = {"x/enc-a": [0, 1], "x/enc-b": [1]}


def textos(n_por_classe: int, semente: int) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    linhas = [
        (f"{semente}-{c}-{i}", f"p{i % 4}", f"{c} palavra{rng.integers(6)} estilo{c} comum{rng.integers(3)} fim", c)
        for c in CLASSES
        for i in range(n_por_classe)
    ]
    return pd.DataFrame(linhas, columns=["id", "prompt_id", "text", "model"])


def extrair_falso(modelo, textos_, lote=16, max_len=512):
    """[n, 2 camadas, 5]: o rótulo aparece como um deslocamento, com ruído."""
    classe = np.array([CLASSES.index(t.split()[0].split("-")[0] + "-" + t.split()[0].split("-")[1]) for t in textos_])
    rng = np.random.default_rng(len(textos_))
    return (np.eye(5)[classe % 5][:, None, :] * 3 + rng.normal(size=(len(textos_), 2, 5))).astype(np.float16)


@pytest.fixture
def ambiente(monkeypatch, tmp_path):
    for modulo in (pipeline, artefatos):
        monkeypatch.setattr(modulo, "CAMADAS", FALSOS)
        monkeypatch.setattr(modulo, "extrair", extrair_falso)
    treino = textos(30, semente=1)
    artefatos.construir(treino, tmp_path / "modelos", tmp_path / "art")
    return treino, tmp_path


def test_prever_completo_devolve_formato_valido_e_salva_a_cada_componente(ambiente):
    _, tmp = ambiente
    teste = textos(8, semente=2)[["id", "text"]]
    salvos = []
    sub = pipeline.prever(teste, tmp / "modelos", tmp / "art", salvar=salvos.append)
    assert list(sub.columns) == ["id", "model"] and list(sub["id"]) == list(teste["id"])
    assert set(sub["model"]) <= set(CLASSES) and sub["model"].notna().all()
    assert len(salvos) == 3  # tfidf + dois encoders


def test_componente_que_levanta_excecao_e_pulado(ambiente, monkeypatch):
    _, tmp = ambiente

    def quebra_no_b(modelo, textos_, **kw):
        if "enc-b" in str(modelo):
            raise RuntimeError("sem GPU")
        return extrair_falso(modelo, textos_)

    monkeypatch.setattr(pipeline, "extrair", quebra_no_b)
    salvos = []
    sub = pipeline.prever(textos(8, 3)[["id", "text"]], tmp / "modelos", tmp / "art", salvar=salvos.append)
    assert len(salvos) == 2 and set(sub["model"]) <= set(CLASSES)


def test_tfidf_corrompido_e_pulado_sem_impedir_os_encoders(ambiente):
    _, tmp = ambiente
    (tmp / "art" / "tfidf.npz").write_bytes(b"lixo")
    salvos = []
    sub = pipeline.prever(textos(8, 4)[["id", "text"]], tmp / "modelos", tmp / "art", salvar=salvos.append)
    assert len(sub) == 24 and len(salvos) == 2
    assert set(sub["model"]) <= set(CLASSES)


def test_sem_nenhum_componente_levanta_erro(ambiente, monkeypatch):
    _, tmp = ambiente
    (tmp / "art" / "tfidf.npz").write_bytes(b"lixo")
    monkeypatch.setattr(pipeline, "extrair", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError, match="nenhum componente"):
        pipeline.prever(textos(8, 5)[["id", "text"]], tmp / "modelos", tmp / "art")


@pytest.mark.parametrize("n_classes", [2, 3, 18])
def test_tfidf_portavel_equivale_ao_predict_proba(n_classes, tmp_path):
    treino = pd.DataFrame(
        [(f"classe{c} Café! estilo{c} comum{i % 3}\t  fim\n", f"m-{c:02}")
         for c in range(n_classes) for i in range(12)],
        columns=["text", "model"],
    )
    pipe = pipeline.montar_tfidf(LogisticRegression(C=20, max_iter=2000)).fit(treino, treino["model"])
    artefatos.exportar_tfidf(pipe, tmp_path)
    teste = pd.DataFrame({"text": [
        "classe0 Café! estilo0 comum1 fim", "classe1\tCafé!  estilo1 comum2\n fim",
        "CLASSE0 café? zyxwvu987654321", "", "\t \n", "🦙 中文", "!!!  ...",
    ]})
    esperado = pipe.predict_proba(teste)
    obtido = pipeline.proba_tfidf(teste["text"].tolist(), tmp_path)
    np.testing.assert_allclose(obtido, esperado, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("cenario", ["completo", "sem_tfidf", "so_tfidf"])
def test_notebook_executa_sem_train_nem_sample_e_isola_componentes(ambiente, monkeypatch, cenario):
    import montar_notebook

    _, tmp = ambiente
    teste = textos(8, 6)[["id", "text"]]
    teste.to_csv(tmp / "test.csv", index=False)
    if cenario == "sem_tfidf":
        (tmp / "art" / "tfidf.npz").unlink()
    if cenario != "so_tfidf":
        for modelo in FALSOS:
            pasta = tmp / "modelos" / modelo.split("/")[-1]
            pasta.mkdir(parents=True)
            (pasta / "config.json").write_text("{}")
    monkeypatch.chdir(tmp)
    ns = {}
    exec(montar_notebook.codigo_do_pipeline(), ns)
    ns["CAMADAS"] = FALSOS

    def extrair_local(modelo, textos_, **kw):
        assert (modelo / "config.json").exists()
        return extrair_falso(modelo, textos_)

    ns["extrair"] = extrair_local
    exec(montar_notebook.EXECUCAO.replace('Path("/kaggle/input")', f"Path({str(tmp)!r})"), ns)
    sub = pd.read_parquet(tmp / "submission.parquet")
    assert list(sub.columns) == ["id", "model"]
    assert sub["id"].tolist() == teste["id"].tolist()
    assert set(sub["model"]) == set(CLASSES)  # um fallback constante não satisfaz o teste
    assert not (tmp / "train.csv").exists() and not (tmp / "sample_submission.parquet").exists()
