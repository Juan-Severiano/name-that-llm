import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import montar_notebook  # noqa: E402


def test_celulas_de_codigo_compilam_e_nao_dependem_do_pacote():
    codigo = [c for c in montar_notebook.notebook()["cells"] if c["cell_type"] == "code"]
    assert len(codigo) == 2
    for c in codigo:
        fonte = "".join(c["source"])
        compile(fonte, "<celula>", "exec")
        assert "name_that_llm" not in fonte


def test_celula_do_pipeline_define_o_que_a_execucao_usa():
    pipeline, execucao = ("".join(c["source"]) for c in montar_notebook.notebook()["cells"][1:])
    assert "def prever(" in pipeline and "prever(" in execucao
    assert not pipeline.lstrip().startswith('"""')


def test_metadata_sem_internet_com_gpu_e_a_competicao():
    meta = montar_notebook.metadata()
    assert meta["enable_internet"] == "false" and meta["enable_gpu"] == "true"
    assert meta["competition_sources"] == ["name-that-llm"]
    assert meta["dataset_sources"] == [montar_notebook.DATASET, montar_notebook.DATASET_ARTEFATOS]
    assert meta["id"].split("/")[1] == meta["title"]
    json.dumps(meta)


def test_execucao_nao_depende_do_train_csv_nem_do_sample_submission():
    execucao = "".join(montar_notebook.notebook()["cells"][2]["source"])
    assert "sample_submission" not in execucao
    assert "arq_treino" not in execucao and "treino=" not in execucao
    assert "except Exception" in execucao  # o notebook não pode terminar com erro
