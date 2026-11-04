"""Ensaio das células reais da v5 em 540 linhas com embeddings em cache (sem GPU, sem rótulos, sem treino).

Reaproveita o replay da v4 trocando as comparações finais: aqui as probabilidades e os rótulos mudam de propósito.
Confere formato, que cada grupo inferido tem rótulos distintos e quantos rótulos a v5 muda em relação à v4.
Uso: PYTHONPATH=src:scripts uv run python scripts/replay_atribuicao.py"""

import inspect

import montar_notebook_gaussiano as m
import montar_notebook_atribuicao as v5

m.SAIDA = v5.SAIDA
fonte = inspect.getsource(m.replay_cache)
corte = fonte.index("    np.testing.assert_array_equal(real.probabilidades, ref.probabilidades)")
verificacao = '''    from name_that_llm import atribuicao as atr
    sub = real.submissao
    pd.testing.assert_frame_equal(pd.read_parquet(saida / "submission.parquet"), sub)
    assert sub.id.tolist() == teste.id.tolist() and sub.id.is_unique and sub.model.notna().all()
    grupos = atr.agrupar(atr.distancias(teste.text.tolist(), semantica.astype(np.float32)))
    tamanhos = np.bincount(grupos)
    repetidos = sum(int(sub.model[grupos == g].duplicated().sum()) for g in np.unique(grupos) if 1 < tamanhos[g] <= 18)
    mudou = float((sub.model.to_numpy() != ref.submissao.model.to_numpy()).mean())
    print(json.dumps({"linhas": len(sub), "grupos": int(len(tamanhos)), "tamanho_mediano": float(np.median(tamanhos)),
                      "maior_grupo": int(tamanhos.max()), "rotulos_repetidos_dentro_de_grupo": repetidos,
                      "fracao_de_rotulos_diferentes_da_v4": mudou, "segundos": time.monotonic() - inicio}), flush=True)
    assert repetidos == 0 and len(chamadas) == 5
'''
exec(compile(fonte[:corte].replace("def replay_cache", "def replay_v5") + verificacao, "replay_v5", "exec"), m.__dict__)
m.replay_v5()
