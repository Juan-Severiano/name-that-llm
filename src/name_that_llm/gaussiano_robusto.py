"""Política conservadora local, separada do Gaussian puro.

Grupos INFERIDOS menores que quatro recebem mean5 (TF-IDF + quatro LR).
Os demais recebem o Gaussian puro. Nenhuma exceção vira previsão constante.
Scalers/LR/TF-IDF são arrays portáteis: inferência não ajusta classificadores,
não usa pickle, labels de teste, torch, transformers ou quotas por classe.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit, softmax
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from name_that_llm import gaussiano as g

BASES=tuple(k for k in g.VISTAS if k!="StyleDistance_L12")
CONFIG={"versao":1,"menor_grupo_gaussiano":4,"alpha_gaussiano_grupo_elegivel":1.,
        "baseline":"mean5_tfidf_quatro_lr","C_embeddings":.05,"max_iter_embeddings":1000,
        "sinkhorn":False,"quota_classes":False}
TOKENS=r"(?u)\b\w+\b|[^\w\s]"
TFIDF_PARAMS={"palavras":{"token_pattern":TOKENS,"ngram_range":(1,2)},
              "caracteres":{"analyzer":"char","ngram_range":(1,5)}}


@dataclass
class Modelo:
    gaussiano:g.Modelo
    lineares:dict[str,dict[str,np.ndarray]]
    vocabulario:dict[str,list[str]]
    pesos_tfidf:dict[str,np.ndarray]
    metadados:dict


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def _linear(X,p):
    X=np.asarray(X,dtype=np.float32).copy()
    X-=p["media"].astype(np.float32)
    X/=p["escala"].astype(np.float32)
    scores=X@p["coef"].T+p["intercepto"]
    if scores.shape[1]==1:
        positive=expit(scores[:,0])
        return np.column_stack((1-positive,positive))
    return softmax(scores,axis=1)


def _tfidf(textos,modelo):
    blocks=[]
    for name in ("palavras","caracteres"):
        counts=CountVectorizer(vocabulary=modelo.vocabulario[name],lowercase=False,**TFIDF_PARAMS[name]).transform(textos).astype(np.float64)
        counts.data=1+np.log(counts.data)
        blocks.append(normalize(counts@sp.diags(modelo.pesos_tfidf[f"idf_{name}"])))
    X=sp.hstack(blocks).tocsr()
    scores=X@modelo.pesos_tfidf["coef"].T+modelo.pesos_tfidf["intercepto"]
    if scores.shape[1]==1:
        positive=expit(scores[:,0])
        return np.column_stack((1-positive,positive))
    return softmax(scores,axis=1)


def fit(treino,matrizes,gaussiano,artefatos_tfidf):
    """Ajusta somente os quatro classificadores fallback; Gaussian permanece fixo."""
    g._validar_modelo(gaussiano)
    ids,_=g._ids_textos(treino)
    if not np.array_equal(ids,gaussiano.treino_ids):
        raise ValueError("treino do fallback difere da ordem do Gaussian")
    if not {"model"}<=set(treino.columns):raise ValueError("treino sem labels")
    labels=treino.model.to_numpy(dtype=str)
    if not np.array_equal(np.unique(labels),gaussiano.classes):raise ValueError("classes fallback/Gaussian diferentes")
    vistas,_=g._validar_matrizes(ids,matrizes,gaussiano)
    pasta=Path(artefatos_tfidf)
    if json.loads((pasta/"rotulos.json").read_text())!=labels.tolist():
        raise ValueError("ordem/labels de treino do TF-IDF incompatíveis")
    vocabulario=json.loads((pasta/"tfidf_vocab.json").read_text())
    with np.load(pasta/"tfidf.npz",allow_pickle=False) as f:
        pesos={k:f[k].copy(order="K") for k in f.files}
    lineares={}
    for name in BASES:
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=1000)).fit(vistas[name],labels)
        scaler,clf=pipe.steps[0][1],pipe.steps[1][1]
        lineares[name]={"media":scaler.mean_.copy(),"escala":scaler.scale_.copy(),
                        "coef":clf.coef_.copy(order="K"),"intercepto":clf.intercept_.copy(),"classes":clf.classes_.astype(str)}
        if not np.array_equal(lineares[name]["classes"],gaussiano.classes):raise ValueError("classificador fora da ordem canônica")
    metadata={"config":dict(CONFIG),"classes":gaussiano.classes.tolist(),"treino_ids_sha256":g.checksum_ids(ids),
              "treino_linhas":len(ids),"vistas_baseline":json.loads(json.dumps({k:g.VISTAS[k] for k in BASES})),
              "tfidf_origem_sha256":_sha(pasta/"tfidf.npz"),"tfidf_vocab_origem_sha256":_sha(pasta/"tfidf_vocab.json"),
              "regra_origem":"hipótese pós-diagnóstico testada no split20261105, sem retuning nesse split",
              "proibido":"labels/prompt_id de teste, quotas de classe, fallback por exceção"}
    modelo=Modelo(gaussiano,lineares,vocabulario,pesos,metadata)
    _validar(modelo)
    return modelo


def _validar(modelo):
    g._validar_modelo(modelo.gaussiano)
    meta=modelo.metadados
    classes=modelo.gaussiano.classes
    if meta.get("config")!=CONFIG or meta.get("classes")!=classes.tolist():raise ValueError("config/classes robustas incompatíveis")
    if meta.get("treino_ids_sha256")!=g.checksum_ids(modelo.gaussiano.treino_ids) or meta.get("treino_linhas")!=len(modelo.gaussiano.treino_ids):
        raise ValueError("checksum/ordem de treino fallback incompatível")
    if set(modelo.lineares)!=set(BASES) or meta.get("vistas_baseline")!={k:g.VISTAS[k] for k in BASES}:raise ValueError("vistas fallback incompatíveis")
    C=len(classes)
    for name,p in modelo.lineares.items():
        if set(p)!={"media","escala","coef","intercepto","classes"}:raise ValueError("campos LR inválidos")
        if not np.array_equal(p["classes"],classes):raise ValueError("ordem das classes LR incompatível")
        for k in ("media","escala","coef","intercepto"):
            if np.asarray(p[k]).dtype.kind not in "fiu" or not np.isfinite(p[k]).all():raise ValueError("parâmetros LR não finitos")
        D=len(p["media"]);rows=1 if C==2 else C
        if p["media"].shape!=(D,) or p["escala"].shape!=(D,) or p["coef"].shape!=(rows,D) or p["intercepto"].shape!=(rows,):raise ValueError("shape LR inválido")
        if D!=len(modelo.gaussiano.parametros[name]["media"]) or np.any(p["escala"]<=0):raise ValueError("dimensão/escala LR inválida")
    if set(modelo.vocabulario)!={"palavras","caracteres"}:raise ValueError("vocabulários TF-IDF incompatíveis")
    if set(modelo.pesos_tfidf)!={"idf_palavras","idf_caracteres","coef","intercepto"}:raise ValueError("pesos TF-IDF incompatíveis")
    for name,terms in modelo.vocabulario.items():
        if not isinstance(terms,list) or not terms or any(not isinstance(s,str) for s in terms) or len(set(terms))!=len(terms):raise ValueError("vocabulário TF-IDF inválido")
        idf=modelo.pesos_tfidf[f"idf_{name}"]
        if idf.shape!=(len(terms),) or np.any(idf<=0):raise ValueError("shape/idf TF-IDF inválido")
    if any(np.asarray(v).dtype.kind not in "fiu" or not np.isfinite(v).all() for v in modelo.pesos_tfidf.values()):raise ValueError("pesos TF-IDF não finitos")
    rows=1 if C==2 else C
    if modelo.pesos_tfidf["coef"].shape!=(rows,sum(map(len,modelo.vocabulario.values()))) or modelo.pesos_tfidf["intercepto"].shape!=(rows,):raise ValueError("classes/shape TF-IDF incompatíveis")


def prever_baseline(modelo,teste,matrizes):
    """Somente quatro vistas são lidas; StyleDistance/semântica não participam."""
    _validar(modelo)
    ids,textos=g._ids_textos(teste)
    if not isinstance(matrizes,g.Matrizes) or not np.array_equal(ids,np.asarray(matrizes.ids,dtype=str)):raise ValueError("IDs do baseline fora de ordem")
    if not set(BASES)<=set(matrizes.vistas):raise ValueError("vista baseline ausente")
    if not len(ids):return np.empty((0,len(modelo.gaussiano.classes)))
    parts=[_tfidf(textos,modelo)]
    for name in BASES:
        X=g._matriz(matrizes.vistas[name],len(ids),name)
        if X.shape[1]!=len(modelo.lineares[name]["media"]):raise ValueError("shape de vista baseline inválido")
        parts.append(_linear(X,modelo.lineares[name]))
    p=np.mean(parts,axis=0)
    if p.shape!=(len(ids),len(modelo.gaussiano.classes)) or not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(axis=1)<=0):raise ValueError("probabilidades baseline inválidas")
    return p/p.sum(axis=1,keepdims=True)


def prever(modelo,teste,matrizes):
    """Pool inteiro após extrair todas as features; batching somente nos encoders.

    Roteamento explícito por tamanho inferido, sem capturar erros de modelos.
    Particionar este método em chunks muda os grupos e as previsões.
    """
    _validar(modelo)
    gaussian=g.prever(modelo.gaussiano,teste,matrizes)
    baseline=prever_baseline(modelo,teste,matrizes)
    if not len(baseline):return gaussian
    _,inv,counts=np.unique(gaussian.grupos,return_inverse=True,return_counts=True)
    mask=counts[inv]>=CONFIG["menor_grupo_gaussiano"]
    p=baseline.copy();p[mask]=gaussian.probabilidades[mask]
    if not np.isfinite(p).all():raise ValueError("roteamento gerou probabilidades não finitas")
    ids,_=g._ids_textos(teste)
    submission=pd.DataFrame({"id":ids,"model":modelo.gaussiano.classes[p.argmax(axis=1)]})
    diagnostic=dict(gaussian.diagnostico)|{"politica":"grupo inferido <4 mean5; >=4 Gaussian", "linhas_gaussian":int(mask.sum()),
                                            "linhas_mean5":int((~mask).sum()),"falhas_convertidas_em_fallback":False}
    return g.Resultado(submission,p,gaussian.grupos,diagnostic)


def salvar(modelo,destino):
    _validar(modelo)
    destino=Path(destino);destino.mkdir(parents=True,exist_ok=True)
    g.salvar(modelo.gaussiano,destino/"gaussiano")
    arrays={"classes":modelo.gaussiano.classes}
    for i,name in enumerate(BASES):arrays.update({f"b{i}_{k}":v for k,v in modelo.lineares[name].items()})
    arrays.update({f"tfidf_{k}":v for k,v in modelo.pesos_tfidf.items()})
    tmp=destino/"fallback.npz.tmp"
    with tmp.open("wb") as f:np.savez_compressed(f,**arrays)
    tmp.replace(destino/"fallback.npz")
    tmp=destino/"tfidf_vocab.json.tmp";tmp.write_text(json.dumps(modelo.vocabulario,ensure_ascii=False));tmp.replace(destino/"tfidf_vocab.json")
    metadata=dict(modelo.metadados)|{"fallback_npz_sha256":_sha(destino/"fallback.npz"),"vocab_json_sha256":_sha(destino/"tfidf_vocab.json"),
                                     "gaussiano_npz_sha256":_sha(destino/"gaussiano/parametros.npz"),"formato":"allow_pickle=False"}
    tmp=destino/"metadados.json.tmp";tmp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n");tmp.replace(destino/"metadados.json")


def carregar(origem):
    origem=Path(origem);metadata=json.loads((origem/"metadados.json").read_text())
    for filename,key in (("fallback.npz","fallback_npz_sha256"),("tfidf_vocab.json","vocab_json_sha256"),("gaussiano/parametros.npz","gaussiano_npz_sha256")):
        if _sha(origem/filename)!=metadata.get(key):raise ValueError(f"checksum inválido: {filename}")
    gaussian=g.carregar(origem/"gaussiano")
    with np.load(origem/"fallback.npz",allow_pickle=False) as f:
        expected={"classes"}|{f"b{i}_{k}" for i in range(len(BASES)) for k in ("media","escala","coef","intercepto","classes")}|{f"tfidf_{k}" for k in ("idf_palavras","idf_caracteres","coef","intercepto")}
        if set(f.files)!=expected:raise ValueError("campos fallback NPZ inválidos")
        if not np.array_equal(f["classes"],gaussian.classes):raise ValueError("classes NPZ/Gaussian incompatíveis")
        lineares={name:{k:f[f"b{i}_{k}"].copy(order="K") for k in ("media","escala","coef","intercepto","classes")} for i,name in enumerate(BASES)}
        weights={k:f[f"tfidf_{k}"].copy(order="K") for k in ("idf_palavras","idf_caracteres","coef","intercepto")}
    model=Modelo(gaussian,lineares,json.loads((origem/"tfidf_vocab.json").read_text()),weights,metadata)
    _validar(model)
    return model
