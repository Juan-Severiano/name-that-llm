import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from name_that_llm import gaussiano as g


@pytest.fixture
def ambiente():
    rng=np.random.default_rng(71)
    groups=np.repeat(np.arange(8),6)
    y=np.tile([0,0,1,1,2,2],8)  # cada grupo contém classes repetidas
    topics=["pizza tomato dough oven", "planet orbit telescope moon", "ocean whale salt waves", "garden leaves root soil",
            "violin concert melody piano", "football goal match team", "winter snow ice mountain", "computer memory chip software"]
    df=pd.DataFrame({"id":[f"s{i:03}" for i in range(len(y))],"text":[topics[p]+f" style{c}" for p,c in zip(groups,y)],
                     "prompt_id":[f"p{p}" for p in groups],"model":[f"m{c}" for c in y]})
    views={}
    for k in g.VISTAS:
        class_effect=rng.normal(size=(3,8))*3
        prompt_effect=rng.normal(size=(8,8))*2
        views[k]=(class_effect[y]+prompt_effect[groups]+rng.normal(size=(len(y),8))*.3).astype(np.float32)
    semantic=(np.eye(8)[groups]+rng.normal(size=(len(y),8))*.005).astype(np.float32)
    matrices=g.Matrizes(df.id.to_numpy(dtype=str),views,semantic)
    model=g.fit(df,matrices)
    return df,matrices,model


def test_export_reload_parity_and_no_refit(ambiente,tmp_path,monkeypatch):
    df,m,model=ambiente
    before=g.prever(model,df[["id","text"]],m)
    g.salvar(model,tmp_path)
    monkeypatch.setattr(g,"_ajustar_vista",lambda *a,**k:pytest.fail("reload não pode refazer treino"))
    loaded=g.carregar(tmp_path)
    after=g.prever(loaded,df[["id","text"]],m)
    np.testing.assert_array_equal(before.probabilidades,after.probabilidades)
    np.testing.assert_array_equal(before.grupos,after.grupos)
    pd.testing.assert_frame_equal(before.submissao,after.submissao)
    with np.load(tmp_path/"parametros.npz",allow_pickle=False) as data:
        assert all(data[k].dtype.kind!="O" for k in data.files)
    assert loaded.metadados["treino_ids_sha256"]==g.checksum_ids(df.id)


def test_prediction_ignores_prompt_and_label_columns(ambiente):
    df,m,model=ambiente
    first=g.prever(model,df[["id","text"]],m)
    malicious=df.copy()
    malicious["prompt_id"]=[f"unrelated{i}" for i in range(len(df))]
    malicious["model"]="not-a-valid-class"
    second=g.prever(model,malicious,m)
    np.testing.assert_array_equal(first.probabilidades,second.probabilidades)
    np.testing.assert_array_equal(first.grupos,second.grupos)
    assert not second.diagnostico["prompt_id_da_chamada_usado"]
    assert not second.diagnostico["classes_da_chamada_usadas"]


def test_reordered_feature_ids_are_rejected(ambiente):
    df,m,model=ambiente
    wrong=g.Matrizes(m.ids[::-1],m.vistas,m.semantica)
    with pytest.raises(ValueError,match="ordem/identidade"):
        g.prever(model,df[["id","text"]],wrong)


@pytest.mark.parametrize("failure",["nan","overflow_float32","wrong_shape","missing_view","semantic_shape"])
def test_invalid_features_fail_without_constant_fallback(ambiente,failure):
    df,m,model=ambiente
    wrong=copy.deepcopy(m)
    first=next(iter(g.VISTAS))
    if failure=="nan":wrong.vistas[first][0,0]=np.nan
    elif failure=="overflow_float32":
        wrong.vistas[first]=wrong.vistas[first].astype(np.float64)
        wrong.vistas[first][0,0]=1e300  # finito em float64, inválido depois da conversão
    elif failure=="wrong_shape":wrong.vistas[first]=wrong.vistas[first][:-1]
    elif failure=="missing_view":del wrong.vistas[first]
    else:wrong.semantica=wrong.semantica[:,:-1]
    with pytest.raises(ValueError):g.prever(model,df[["id","text"]],wrong)


def test_model_nonfinite_is_rejected_before_prediction(ambiente):
    df,m,model=ambiente
    model.parametros[next(iter(g.VISTAS))]["prototipos"][0,0]=np.nan
    with pytest.raises(ValueError,match="não finitos"):
        g.prever(model,df[["id","text"]],m)


def test_checksum_and_class_order_are_guarded(ambiente,tmp_path):
    _,_,model=ambiente
    g.salvar(model,tmp_path)
    meta=json.loads((tmp_path/"metadados.json").read_text())
    meta["classes"]=meta["classes"][::-1]
    (tmp_path/"metadados.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="ordem das classes"):g.carregar(tmp_path)
    g.salvar(model,tmp_path)
    with (tmp_path/"parametros.npz").open("ab") as f:f.write(b"corruption")
    with pytest.raises(ValueError,match="checksum"):g.carregar(tmp_path)


def test_repeated_classes_and_groups_larger_than_class_count():
    p={"media":np.zeros(2),"escala":np.ones(2),"V":np.eye(2),"A":np.eye(2),
       "prototipos":np.array([[0.,0.],[8.,0.]]),"lambda":np.array([.1,.1])}
    X=np.array([[.1,0.],[.2,.1],[.3,-.1],[.2,.2]],dtype=np.float32)
    probabilities=g._probabilidades(X,p,np.zeros(4,dtype=int))
    np.testing.assert_array_equal(probabilities.argmax(axis=1),[0,0,0,0])
    np.testing.assert_allclose(probabilities.sum(axis=1),1)
    singleton=g._probabilidades(X,p,np.arange(4))
    assert np.isfinite(singleton).all()


def test_empty_text_singleton_and_unique_complete_ids(ambiente):
    df,m,model=ambiente
    df=df.iloc[:1][["id","text"]].copy();df["text"]=""
    subset=g.Matrizes(m.ids[:1],{k:x[:1] for k,x in m.vistas.items()},m.semantica[:1])
    r=g.prever(model,df,subset)
    assert r.grupos.tolist()==[0] and r.submissao.id.tolist()==df.id.tolist()
    assert set(r.submissao.model)<=set(model.classes) and np.isfinite(r.probabilidades).all()
    bad=pd.concat([df,df],ignore_index=True)
    with pytest.raises(ValueError,match="ids"):
        g.prever(model,bad,g.Matrizes(np.repeat(m.ids[:1],2),{k:np.repeat(x[:1],2,axis=0) for k,x in m.vistas.items()},np.repeat(m.semantica[:1],2,axis=0)))


def test_math_module_import_does_not_load_hf_or_torch():
    env=dict(os.environ,PYTHONPATH=str(Path(g.__file__).resolve().parents[1]))
    result=subprocess.run([sys.executable,"-c","import sys; import name_that_llm.gaussiano; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                          env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
