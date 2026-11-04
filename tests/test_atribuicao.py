import numpy as np

from name_that_llm import atribuicao


def test_atribuir_forca_rotulos_distintos_no_grupo():
    logp = np.log(np.tile([0.7, 0.2, 0.1], (3, 1)))  # os três textos preferem a classe 0
    rotulos = atribuicao.atribuir(logp, np.zeros(3, dtype=int))
    assert sorted(rotulos) == [0, 1, 2]


def test_atribuir_escolhe_a_melhor_soma_e_nao_o_primeiro():
    logp = np.log(np.array([[0.5, 0.4, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]]))
    assert list(atribuicao.atribuir(logp, np.zeros(3, dtype=int))) == [1, 0, 2]


def test_texto_sozinho_e_grupo_maior_que_as_classes_ficam_no_argmax():
    logp = np.log(np.array([[0.1, 0.9], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]))
    grupos = np.array([0, 1, 1, 1])  # grupo 1 tem 3 textos e só 2 classes
    assert list(atribuicao.atribuir(logp, grupos)) == [1, 0, 0, 0]


def test_agrupar_separa_grupos_e_respeita_o_teto():
    rng = np.random.default_rng(0)
    centros = np.repeat(np.eye(3) * 5, 8, axis=0)  # 3 grupos de 8 pontos
    X = centros + rng.normal(scale=0.1, size=centros.shape)
    D = np.linalg.norm(X[:, None] - X[None], axis=2) / 10
    grupos = atribuicao.agrupar(D, corte=0.8, teto=10)
    assert len(np.unique(grupos)) >= 3 and np.bincount(grupos).max() <= 10
    assert len({tuple(sorted(set(grupos[i * 8:(i + 1) * 8]))) for i in range(3)}) >= 3


def test_refinar_devolve_um_rotulo_por_texto_e_usa_so_a_base_em_grupo_pequeno():
    rng = np.random.default_rng(1)
    n, k = 12, 4
    base = rng.dirichlet(np.ones(k), n)
    gauss = rng.dirichlet(np.ones(k), n)
    textos = [f"palavra{i % 3} texto sobre o assunto {i % 3} numero {i}" for i in range(n)]
    semantica = rng.normal(size=(n, 5))
    L, rotulos = atribuicao.refinar(base, gauss, textos, semantica, np.arange(n))  # grupos de 1: sem Gaussian
    assert L.shape == (n, k) and rotulos.shape == (n,) and set(rotulos) <= set(range(k))
    np.testing.assert_allclose(L, np.log(np.clip(base, 1e-9, None)))


def test_agrupar_aceita_um_texto_so():
    assert list(atribuicao.agrupar(np.zeros((1, 1)))) == [0]
