# name-that-llm

Classificador que diz qual de 18 LLMs escreveu um parágrafo, para a competição
[Name That LLM](https://www.kaggle.com/competitions/name-that-llm) do Kaggle (métrica: balanced accuracy,
envio por notebook sem internet). Trabalho em andamento durante a competição.

## Resultados até agora

Nota pública no Kaggle (amostra do teste oculto):

| Envio | O que é | Nota pública |
|---|---|---|
| v2 | média de 4 encoders congelados + TF-IDF (TF-IDF pulado no Kaggle por versão do sklearn) | 0,393 |
| v4 | Gaussian por prompt inferido, cinco vistas | 0,437 |
| v5 | v4 + atribuição húngara por grupo inferido | 0,412 |

A v5 rendia +0,10 na validação local e perdeu no público: a estrutura "um texto de cada modelo por prompt"
do treino não vale como esperado no teste oculto (causa não identificada, o Kaggle não mostra o log da
reexecução). Lição: truques que dependem da estrutura do treino não são confiáveis aqui.

Validação local (por prompt, 5 dobras, nos 2.699 textos oficiais): média de 5 componentes 0,447.
Treinando com 41 mil textos do pool público da comunidade (veja abaixo), a média dos 4 encoders chega a 0,487
nos textos oficiais e a 0,503 junto com os oficiais. Esses números são locais, não notas do Kaggle, e a
regularização foi escolhida olhando o conjunto oficial.

## Como funciona

- `src/name_that_llm/pipeline.py`: inferência do envio. Embeddings (média dos tokens em várias camadas) de
  bge-small, roberta-base, deberta-v3-base e Qwen3-0.6B, regressão logística em cada um, TF-IDF de palavras
  e caracteres, média das probabilidades. Cada componente falha sozinho sem derrubar o notebook.
- `src/name_that_llm/artefatos.py`: constrói os artefatos de treino (a reexecução do Kaggle troca o dataset
  da competição por uma versão oculta, então o notebook não pode depender do `train.csv`).
- `src/name_that_llm/gaussiano*.py`: Gaussian com remoção do efeito do tema compartilhado entre respostas.
- `src/name_that_llm/atribuicao.py`: grupos de textos do mesmo prompt e atribuição húngara.
- `scripts/treino_lora.py`: treino LoRA de um Qwen3 como classificador, em duas etapas (pool e oficiais),
  para rodar numa GPU T4 do Kaggle. `scripts/montar_notebook*.py` geram os notebooks.
- `RELATORIO_VALIDACAO.md` e `ANALISE_ARTIGOS.md`: campanha de validação e leitura de artigos.

## Dados e licenças

Os dados da competição e os embeddings não estão no repositório (regras da competição). O pool público usado
para treino é o dataset `crtproduction/name-that-llm-open-pool` do Kaggle, que reúne datasets da comunidade
com licenças CC-BY 4.0, CC-BY-SA 4.0 e CC0 (lista em `sources.csv` do próprio dataset).

## Rodar

```bash
uv sync
uv run pytest
```

Os scripts em `scripts/` esperam os dados em `data/` (baixados com a CLI do Kaggle) e a credencial do Kaggle
em `.env` (fora do repositório).
