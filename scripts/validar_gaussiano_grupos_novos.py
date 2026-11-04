"""Mesma auditoria OOF5 do Gaussian, mas com os grupos de teste inferidos por atribuicao.agrupar (corte 0,8,
teto 18) em vez do corte 0,60 sem teto. Uso: PYTHONPATH=src:scripts uv run python scripts/validar_gaussiano_grupos_novos.py"""

import numpy as np

import validar_gaussiano_completo as v
from name_that_llm import atribuicao as atr
from threadpoolctl import threadpool_limits

v.research.infer_groups = lambda textos, sem: atr.agrupar(atr.distancias(textos, np.asarray(sem, dtype=np.float32)))
v.OUT = v.ROOT / "runs/gaussiano_grupos_novos"
with threadpool_limits(limits=1):
    v.main()
