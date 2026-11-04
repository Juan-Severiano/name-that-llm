"""Notebook v5: o Gaussian robusto da v4 mais a atribuição húngara por grupo inferido (atribuicao.py).

Reaproveita o gerador da v4 (montar_notebook_gaussiano.py) e troca só o trecho que decide os rótulos.
Não publica nada. Uso: PYTHONPATH=src uv run python scripts/montar_notebook_atribuicao.py"""

import json

import montar_notebook_gaussiano as m

SAIDA = m.RAIZ / "notebook_atribuicao"

ANTES = "    previsao = robusto.prever(modelo, teste, matrizes)\n    p = previsao.probabilidades\n    sub = previsao.submissao\n"
DEPOIS = '''    # Os mesmos grupos servem ao Gaussian (remoção do tema) e à atribuição: corte 0,8 com teto de 18 textos.
    g._grupos = lambda txt, sem: (atribuicao.agrupar(atribuicao.distancias(txt, sem)), {"sem_vocabulario": False})
    previsao = robusto.prever(modelo, teste, matrizes)
    base = robusto.prever_baseline(modelo, teste, matrizes)
    gauss = g.prever(modelo.gaussiano, teste, matrizes)
    L, rotulos = atribuicao.refinar(base, gauss.probabilidades, textos, semantica, previsao.grupos,
                                    robusto.CONFIG["menor_grupo_gaussiano"])
    sub = pd.DataFrame({"id": ids, "model": modelo.gaussiano.classes[rotulos]})
    p = g.softmax(L, axis=1)
    previsao = g.Resultado(sub, p, previsao.grupos, previsao.diagnostico | {
        "atribuicao": f"hungaro por grupo inferido (corte {atribuicao.CORTE}, teto {atribuicao.TETO}), Gaussian 70% + base 30%"})
'''
INTRO = m.INTRO.replace("# Name That LLM — Gaussian robusto, cinco vistas", "# Name That LLM — Gaussian robusto + atribuição por prompt") + """
Novidade: cada modelo escreve no máximo um texto por prompt. Os textos de um mesmo prompt são agrupados pelo
conteúdo (TF-IDF + BGE) e, em cada grupo, os rótulos são distintos (algoritmo húngaro sobre a mistura de
70% Gaussian e 30% média dos classificadores).
"""


def notebook():
    doc, pacote = m.notebook()
    codigo = doc["cells"][1]["source"]
    fonte = (m.RAIZ / "src/name_that_llm/atribuicao.py").read_text()
    codigo = "".join(codigo) + (
        "\natribuicao = types.ModuleType('atribuicao_embutida')\nsys.modules[atribuicao.__name__] = atribuicao\n"
        f"exec({fonte!r}, atribuicao.__dict__)\n"
    )
    doc["cells"][0] = m.celula("markdown", INTRO)
    doc["cells"][1] = m.celula("code", codigo)
    runtime = "".join(doc["cells"][2]["source"])
    assert runtime.count(ANTES) == 1
    doc["cells"][2] = m.celula("code", runtime.replace(ANTES, DEPOIS))
    return doc, pacote


if __name__ == "__main__":
    SAIDA.mkdir(exist_ok=True)
    doc, pacote = notebook()
    (SAIDA / f"{m.NOME}.ipynb").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    (SAIDA / "kernel-metadata.json").write_text(json.dumps(m.metadata(), indent=2) + "\n")
    (SAIDA / "manifesto_pacote.json").write_text(json.dumps(pacote, ensure_ascii=False, indent=2) + "\n")
    print(f"Escrito: {SAIDA}; nenhum upload ou submissão.")
