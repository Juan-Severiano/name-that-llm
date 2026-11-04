import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_gaussiano import ambiente
from name_that_llm import gaussiano as g
from name_that_llm import gaussiano_robusto as r


@pytest.fixture
def preparado(ambiente,tmp_path):
    from name_that_llm.artefatos import exportar_tfidf
    from name_that_llm.pipeline import montar_tfidf
    from sklearn.linear_model import LogisticRegression
    df,matrices,gaussian=ambiente
    pasta=tmp_path/"tfidf";pasta.mkdir()
    pipe=montar_tfidf(LogisticRegression(C=20,max_iter=2000)).fit(df,df.model.to_numpy(dtype=str))
    exportar_tfidf(pipe,pasta)
    (pasta/"rotulos.json").write_text(json.dumps(df.model.tolist()))
    robust=r.fit(df,matrices,gaussian,pasta)
    return df,matrices,gaussian,robust,pipe,pasta


def subset(df,m,indices):
    return df.iloc[indices][["id","text"]],g.Matrizes(m.ids[indices],{k:x[indices] for k,x in m.vistas.items()},m.semantica[indices])


def test_portable_mean5_matches_native_pipelines(preparado):
    from name_that_llm.pipeline import lr_embeddings
    df,m,_,robust,tfidf,_=preparado
    native=[tfidf.predict_proba(df)]
    for name in r.BASES:
        pipe=lr_embeddings().fit(m.vistas[name],df.model.to_numpy(dtype=str))
        native.append(pipe.predict_proba(m.vistas[name]))
    expected=np.mean(native,axis=0);expected/=expected.sum(axis=1,keepdims=True)
    obtained=r.prever_baseline(robust,df[["id","text"]],m)
    np.testing.assert_allclose(obtained,expected,rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(obtained.argmax(axis=1),expected.argmax(axis=1))
    np.testing.assert_allclose(r._tfidf(df.text.tolist(),robust),native[0],rtol=1e-10,atol=1e-12)


def test_save_reload_has_no_fit_or_pickle_and_preserves_pure(preparado,tmp_path,monkeypatch):
    from sklearn.linear_model import LogisticRegression
    df,m,gaussian,robust,_,_=preparado
    pure=tmp_path/"pure";g.salvar(gaussian,pure)
    original=(pure/"parametros.npz").read_bytes()
    before=r.prever(robust,df[["id","text"]],m)
    dest=tmp_path/"robust";r.salvar(robust,dest)
    monkeypatch.setattr(LogisticRegression,"fit",lambda *a,**k:pytest.fail("inferência/reload não pode ajustar LR"))
    monkeypatch.setattr(g,"_ajustar_vista",lambda *a,**k:pytest.fail("Gaussian não pode ser reajustado"))
    loaded=r.carregar(dest)
    after=r.prever(loaded,df[["id","text"]],m)
    np.testing.assert_array_equal(before.probabilidades,after.probabilidades)
    np.testing.assert_array_equal(before.grupos,after.grupos)
    pd.testing.assert_frame_equal(before.submissao,after.submissao)
    assert (pure/"parametros.npz").read_bytes()==original
    with np.load(dest/"fallback.npz",allow_pickle=False) as f:
        assert all(f[k].dtype.kind!="O" for k in f.files)


@pytest.mark.parametrize("k",[1,2,3,4])
def test_route_uses_inferred_size_and_boundary_four(preparado,k):
    df,m,_,robust,_,_=preparado
    indices=np.concatenate([np.flatnonzero(df.prompt_id.to_numpy()==p)[:k] for p in df.prompt_id.unique()])
    test,features=subset(df,m,indices)
    result=r.prever(robust,test,features)
    counts=np.bincount(result.grupos)[result.grupos]
    assert np.all(counts==k)
    if k<4:
        np.testing.assert_array_equal(result.probabilidades,r.prever_baseline(robust,test,features))
        assert result.diagnostico["linhas_gaussian"]==0
    else:
        np.testing.assert_array_equal(result.probabilidades,g.prever(robust.gaussiano,test,features).probabilidades)
        assert result.diagnostico["linhas_mean5"]==0


def test_baseline_ignores_style_semantics_and_test_labels(preparado):
    df,m,_,robust,_,_=preparado
    baseline=r.prever_baseline(robust,df[["id","text"]],m)
    auxiliary=copy.deepcopy(m)
    auxiliary.vistas["StyleDistance_L12"][:]=np.nan
    auxiliary.semantica[:]=np.nan
    malicious=df.copy();malicious["model"]="invalid";malicious["prompt_id"]="all-the-same"
    np.testing.assert_array_equal(baseline,r.prever_baseline(robust,malicious,auxiliary))
    # Gaussian needs these matrices: the full predictor must reject corruption, not silently fall back.
    with pytest.raises(ValueError,match="não finitos"):r.prever(robust,malicious,auxiliary)
    original=r.prever(robust,df[["id","text"]],m)
    np.testing.assert_array_equal(original.probabilidades,r.prever(robust,malicious,m).probabilidades)


@pytest.mark.parametrize("kind",["ids","overflow","classes","checksum_train","nan_coef"])
def test_invalid_inputs_or_models_raise_instead_of_fallback(preparado,kind):
    df,m,_,model,_,_=preparado
    m=copy.deepcopy(m);model=copy.deepcopy(model)
    first=r.BASES[0]
    if kind=="ids":m.ids=m.ids[::-1]
    elif kind=="overflow":
        m.vistas[first]=m.vistas[first].astype(np.float64);m.vistas[first][0,0]=1e300
    elif kind=="classes":model.lineares[first]["classes"]=model.lineares[first]["classes"][::-1]
    elif kind=="checksum_train":model.metadados["treino_ids_sha256"]="wrong"
    else:model.lineares[first]["coef"][0,0]=np.nan
    with pytest.raises(ValueError):r.prever(model,df[["id","text"]],m)


def test_empty_pool_and_empty_text_are_valid(preparado):
    df,m,_,model,_,_=preparado
    empty,features=subset(df,m,np.array([],dtype=int))
    result=r.prever(model,empty,features)
    assert result.submissao.empty and result.probabilidades.shape==(0,3)
    singleton,features=subset(df,m,np.array([0]));singleton=singleton.copy();singleton["text"]=""
    result=r.prever(model,singleton,features)
    assert result.submissao.id.tolist()==singleton.id.tolist() and np.isfinite(result.probabilidades).all()
    np.testing.assert_array_equal(result.probabilidades,r.prever_baseline(model,singleton,features))


def test_fallback_train_must_match_pure_and_tfidf_order(preparado,monkeypatch):
    df,m,pure,_,_,folder=preparado
    monkeypatch.setattr(g,"_ajustar_vista",lambda *a,**k:pytest.fail("não reajustar Gaussian"))
    r.fit(df,m,pure,folder)
    with pytest.raises(ValueError,match="ordem"):
        r.fit(df.iloc[::-1],m,pure,folder)
    (folder/"rotulos.json").write_text(json.dumps(df.model.tolist()[::-1]))
    with pytest.raises(ValueError,match="TF-IDF"):
        r.fit(df,m,pure,folder)


def test_corrupt_artifact_checksum_is_not_suppressed(preparado,tmp_path):
    _,_,_,model,_,_=preparado
    folder=tmp_path/"bad";r.salvar(model,folder)
    with (folder/"fallback.npz").open("ab") as f:f.write(b"bad")
    with pytest.raises(ValueError,match="checksum"):r.carregar(folder)


def test_robust_math_import_stays_light():
    env=dict(os.environ,PYTHONPATH=str(Path(r.__file__).resolve().parents[1]))
    proc=subprocess.run([sys.executable,"-c","import sys; import name_that_llm.gaussiano_robusto; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                        env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
