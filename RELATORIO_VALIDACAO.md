# Validação local pesada — Name That LLM

Estado em 3 de novembro de 2026: **a meta acima de 0,75 ainda não foi demonstrada**.
A nota pública nova é **0,437**, contra **0,393** da versão anterior: ganho
observado de **4,4 pontos percentuais**, ainda distante da meta de 0,75.
Após a campanha local, o usuário autorizou publicar o candidato e submetê-lo:
**notebook v4, submissão 57958207, COMPLETE sem erro reportado**, nota observada
às 16:23:16 de Fortaleza. A v2/submissão57953429 é a referência anterior.
A campanha local foi concluída. A variante LSA melhorou a amostra pequena, mas foi
rejeitada pelo diagnóstico de escala. A última comparação fixa, TF-IDF de
conteúdo sem compressão, não melhorou o desenvolvimento e parou sem novo holdout.
O candidato local mantido é o Gaussian de cinco vistas, com a opção conservadora
de fallback para grupos pequenos. Esse candidato está agora na versão 4 remota;
o notebook anterior foi preservado localmente em `notebook/`.

## O que melhorou

O ganho consistente veio de estimar e remover o efeito do tema compartilhado
entre respostas. Os grupos de teste são inferidos pelo texto e por embeddings;
`prompt_id` e `model` da validação não entram na inferência.

Na auditoria completa de cinco dobras por prompt, refazendo os ajustes em cada
dobra, o baseline histórico foi reproduzido em **0,4467**; Gaussian fez
**0,5568** e a política conservadora **0,5516**. Ambos ganharam nas cinco dobras.
Gaussian ganhou em 114 prompts, empatou em 18 e perdeu em 18; o intervalo
condicional de 95% do ganho foi [0,0919; 0,1294]. Esta CV usa a configuração
escolhida na exploração anterior da mesma base, portanto não é nested CV
independente da seleção. Evidência: `runs/gaussiano_validacao_completa/results.json`.

| Separação por prompt | Média original | Gaussian com tema inferido | Diferença |
| --- | ---: | ---: | ---: |
| Seed 20261103 | 0,4352 | 0,5611 | +0,1259 |
| Seed 20261104, configuração fixa | 0,4481 | 0,5352 | +0,0870 |
| Seed 20261105, configuração fixa | 0,4148 | 0,5704 | +0,1556 |

Métrica: balanced accuracy. Nas separações acima, cada classe tem 30 exemplos,
então accuracy e balanced accuracy coincidem. Os intervalos de 95% do ganho,
por bootstrap pareado de prompts, foram [0,0741; 0,1796] e [0,0537; 0,1222]
nas duas primeiras separações. São intervalos condicionais a este experimento;
não corrigem a seleção exploratória de muitos métodos e não preveem a nota Kaggle.

O método pode regredir com grupos muito pequenos. Após esse diagnóstico, fixamos
uma política: grupos inferidos com menos de quatro textos usam a média original;
os demais usam Gaussian. Na terceira separação, essa política fez **0,5593**
contra 0,4148 da média original, com intervalo do ganho [0,1056; 0,1870].
Nos cenários com grupos de tamanho 1–3 testados, preservou exatamente as
probabilidades do modelo anterior. É uma alternativa conservadora, não uma
garantia contra toda mudança de distribuição.

Evidências: `runs/experimentos_prompt_removal/paired_original.json`,
`paired_second_split.json`, `policy_seed20261105_*` e
`runs/validacao_pesada/resumo.json`.

## Como a avaliação foi separada

O manifesto `runs/validacao_pesada/split.json` fixa o hash dos dados, IDs/classes
e índices. São 2.159 textos de 120 prompts para desenvolvimento e 540 textos
de 30 prompts reservados. A escolha inicial de configurações usa apenas três
dobras agrupadas dentro do desenvolvimento. Os OOFs foram recalculados sem
usar modelos treinados no holdout externo.

Há uma limitação relevante: estes dados já foram usados em validações históricas
do projeto. As partições posteriores também vêm da mesma base; a segunda
compartilha 126 linhas reservadas com a primeira. A política de grupos pequenos
foi proposta depois dos diagnósticos anteriores e avaliada sem retune na terceira
partição. Isso não substitui um conjunto externo realmente novo.

## O que a auditoria descartou

- Não foram encontradas duplicatas exatas/normalizadas ou similares com Jaccard
  >= 0,8 cruzando os prompts; folds, IDs, classes e ordem dos rótulos conferem.
- Os quatro caches de embeddings são idênticos às fatias usadas pelos artefatos.
  O OOF BGE foi reproduzido exatamente.
- Uma reextração offline em CPU de 14 textos (incluindo extremos de comprimento)
  encontrou a linha correta entre os 2.699 registros em todos os cinco encoders.
  As diferenças de precisão entre dispositivos não mudaram nenhuma das 14
  previsões; a maior variação de probabilidade foi 0,0026 no Gaussian.
  É uma conferência amostral, sem medir acurácia. Evidência:
  `runs/validacao_pesada/embeddings_reextraidos_cpu.json`.
- Rótulos embaralhados e IDs isolados ficaram em aproximadamente 0,0556, o acaso
  com 18 classes. Não surgiu evidência de que um simples erro de ordem explique a nota.
- Restaurar o TF-IDF resolve a incompatibilidade, mas a diferença local do
  ensemble com/sem ele é pequena. Isso não sustenta esperar um salto para 0,75.

Evidência: `runs/auditoria_pesada.json`.

## Alternativas medidas

| Alternativa | Resultado observado | Decisão |
| --- | --- | --- |
| Fusão geométrica dos cinco componentes | Holdout 0,4389; baseline 0,4352 | Ganho não demonstrado |
| TF-IDF com estilometria | Holdout 0,3907 | Inferior |
| MLP sobre embeddings | CV 0,3964; regressão linear 0,4042 | Descartada antes do holdout |
| Fine-tuning RoBERTa, 10 épocas | Melhor inner fold 0,3056; sobreajuste | Descartado antes do holdout |
| Estatísticas de log-probabilidade Qwen + embeddings | Holdout 0,4185 | Inferior |
| StyleDistance misturado ao ensemble | Holdout 0,4296 | Inferior como mistura linear |
| 27 mil passagens públicas, hashing/SGD | Holdout 0,3981 | Não justifica misturar pool indiscriminadamente |
| LoRA Qwen 0,6B só com treino oficial | Inner fold 0,3014; twin blend fixo 0,3278 | Parada no critério prévio; sem holdout |
| LoRA Qwen 0,6B: pool público e depois oficial | Inner fold 0,3167; twin fixo 0,3444 | Ganho pequeno; centered isolado piorou; sem holdout |
| Gaussian sobre embeddings desse LoRA | Inner fold 0,4542; Gaussian dos cinco encoders 0,5083 | Blends .25/.5 não melhoraram; diagnóstico de uma dobra |
| Gaussian sobre LoRA de duas etapas | Inner fold 0,4694; blend 50% 0,5111 versus 0,5083 | Delta +0,0028, IC95 [−0,0250; +0,0306]; sem ganho demonstrado |
| Gaussian adicional Qwen3-4B-Base | Dev 0,5386; holdout 0,5500 versus 0,5611 | Seleção camada20/peso25%; delta IC95 [−0,0296; +0,0074], sem ganho demonstrado |
| LR Qwen3-4B-Base | Dev 0,3774; holdout 0,4056 | Inferior à média original |
| 25% LR4B + 75% média original | Dev 0,4028; holdout 0,4407 versus 0,4352 | Delta IC95 [−0,0111; +0,0241], inconclusivo |
| Gaussian com grupos LSA128 sem limite | Dev 0,5418; holdout 0,6019 | Rejeitado por fragilidade de agrupamento em conjunto maior |
| Gaussian com grupos TF-IDF direto, sem SVD | Dev 0,5270 versus 0,5284 | Encerrado no critério prévio; sem novo holdout |
| Seis regras de fusão das probabilidades Gaussian | Melhor dev 0,5321; holdout 0,5556 versus 0,5611 | Ganho não confirmado; média original preservada |

As linhas com inner fold, CV e holdout têm protocolos distintos e não devem ser
classificadas como se fossem a mesma medição. Resultados negativos do proxy
linear ou do modelo 0,6B não descartam ganhos com modelos maiores.

O treino em duas etapas selecionou 2.001 textos públicos de 230 grupos/18 classes;
1.770 foram efetivamente visitados pelos triplets em uma época. Depois ajustou
somente 1.439 textos oficiais por duas épocas e avaliou os 720 do inner fold0.
Os hashes confirmam início do stage1 no modelo original e início do stage2 no
checkpoint público final. Foram cerca de 20 minutos de treino/extração em MPS,
com memória estável após reduzir o lote físico. A escala é muito menor que os
8B/14B e ~35 mil textos da receita pública; não é uma reprodução daquela receita.

O Qwen3-4B-Base foi fixado na revisão
`906bfd4b4dc7f14ee4320094d8b41684abff8539`, com os três arquivos de pesos
verificados por SHA256. A extração frozen usa 28 camadas, seis profundidades,
EOS explícito, pooling sem o primeiro token e remoção da normalização final.
Os 2.699 textos terminaram sem NaN/OOM em cerca de 7,3 minutos, com memória MPS
observada de 7,26 GB. O tokenizer do mirror Kaggle declarava EOS da variante
chat; `tokenizer_config.json` foi substituído pela versão Base oficial pinada
antes da extração, e ambos os hashes foram registrados.

## Comparação com notebooks públicos

Foram lidos e baixados, sem executar seu código:

- [Teach the Model, Remove the Prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt)
- [Two-stage LoRA on the public pool](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)
- [The Prompt Is Not the Author](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author)
- [The Prompt-Recovery Trap](https://www.kaggle.com/code/yangkuangou/name-that-llm-the-prompt-recovery-trap/log)

A receita combina Qwen3 de 8B/14B, LoRA em duas etapas (pool público e depois
somente dados oficiais), treino contrastivo com respostas do mesmo prompt,
classificadores de texto e redução do efeito do tema. O título `0.600` não é
uma nota confirmada por nós: o próprio texto da versão baixada afirma que a
combinação atual ainda não tem resultado oficial. Notas históricas citadas
pelos autores permanecem relatos, não reproduções desta campanha.

A [avaliação oficial](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation)
informa que as quantidades por modelo são desiguais no teste e que a composição
pública/privada difere. Por isso o candidato não exige quotas uniformes, classes
distintas em cada grupo ou uma resposta de cada autor.

O quarto notebook motivou uma hipótese LSA para agrupar temas. Seu resultado
publicado usa também limite de 18 textos por grupo e assignment de autores;
esses componentes foram excluídos da variante local. Seu ARI publicado não
comprova o desempenho da adaptação sem limite.

A variante local LSA128/limiar de similaridade 0,18 chegou a 0,6019 nos 540
textos reservados, contra 0,5611 do candidato atual. O ganho teve intervalo
condicional [0,0074; 0,0796]. Contudo, nos 2.699 textos/150 prompts a precisão
dos pares agrupados caiu para **0,4227**, com grupos de até 144 textos e
ARI 0,5902. O agrupamento atual, no mesmo conjunto, manteve precisão **0,9997**,
ARI 0,7896 e grupo máximo 18, sem impor limite. A variante LSA foi rejeitada:
o resultado de 0,6019 não é apresentado como candidato pronto para teste maior.
Paridade 1.9/1.6 e permutação passaram; o problema observado é de qualidade
do agrupamento em escala, não de serialização. Na comparação sem compressão,
o TF-IDF direto teve precisão de pares 1,0 nas três dobras, porém recall
0,576–0,604 e balanced accuracy 0,5270, abaixo do candidato atual (0,5284).
O teste foi encerrado sem avaliar outro holdout. Isso não estabelece a
compressão como causa única: representações e limiares mudam o compromisso
entre juntar temas diferentes e separar respostas do mesmo tema.

O [pool público versão 14](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/14)
foi fixado por hashes. O filtro conservador deixou 115.553 passagens de 18 classes,
excluindo fontes proxy, prompts desconhecidos e similaridades suspeitas. Pesos
de qualidade do publicador não foram usados, pois tinham calibração nos prompts
oficiais. Matching lexical não prova ausência de toda paráfrase. Proveniência,
licenças e exclusões estão em `data/open_pool/manifesto.json`.

## Robustez operacional do candidato local

- Artefatos JSON/NPZ sem pickle; exportação em sklearn 1.9.1 e inferência em
  1.6.1 com previsões e clusters idênticos, sem novo ajuste.
- 43 testes passaram nas duas versões, incluindo a política de fallback,
  dados vazios, texto vazio, corrupção, ordem de IDs/classes e ausência de refit.
- O fallback portátil reproduziu a média dos Pipelines originais com diferença
  máxima 4,16e-16. Na troca de versão, suas probabilidades ficaram idênticas.
- Amostra real offline completou os cinco encoders e gerou três linhas válidas.
- Estresse com 5.000 linhas/features: 1,79 s na etapa matemática, pico de 1.261 MB,
  IDs completos, probabilidades finitas e grupos maiores que o número de classes.
  Isso não inclui o custo dos encoders e não mede nova acurácia.
- Cinco permutações de 540 linhas preservaram grupos e previsões; diferença
  numérica máxima 2,05e-15.
- Partir a classificação em blocos de 1/16/64 linhas muda 18,3%/17,4%/13,5% das
  previsões. **Os encoders podem usar batches; agrupamento e classificação
  precisam receber todas as features do teste juntas.**
- A revisão independente encontrou e corrigiu dois controles do cache 4B:
  aceitação de uma representação incompatível e retomada interrompida após
  a última linha. Testes CPU rejeitam parâmetros/dtype/shape/NaN incorretos e
  comprovam finalização de checkpoint completo sem recarregar o modelo.

O fechamento técnico em `runs/validacao_pesada/verificacao_final.json` registra
53/53 testes em cada sklearn, incluindo dez testes novos do notebook empacotado,
e compilação dos arquivos Python locais. O ambiente
sklearn 1.6.1/SciPy 1.18.1 emitiu avisos de opção `iprint` desconhecida durante
fits dos testes, sem falhas; não houve execução de notebooks de terceiros.

Código: `src/name_that_llm/gaussiano.py`, `gaussiano_robusto.py` e
`scripts/preparar_gaussiano.py`.
Artefatos/evidências: `runs/gaussiano_candidato/`.
O subdiretório `robusto/` reúne o Gaussian puro e os pesos portáteis do fallback.
O notebook anterior foi preservado em `notebook/`; o novo pacote fica em
`notebook_gaussiano/`, com dataset Gaussian/StyleDistance separado.

## Publicação e verificação no Kaggle

O pedido “sobe la entao” autorizou esta publicação e uma nova submissão.
Foi criado o dataset **privado**, versão 1,
[name-that-llm-gaussiano](https://www.kaggle.com/datasets/juanseveriano/name-that-llm-gaussiano),
com 18 arquivos montados e 285.242.423 bytes de pesos portáteis e StyleDistance.
Os quatro encoders anteriores são reutilizados. Qwen4B, LoRA e LSA rejeitados
não entraram no pacote. Manifestos verificam nomes, tamanhos e SHA256.

O [notebook v4](https://www.kaggle.com/code/juanseveriano/name-that-llm-submissao)
terminou **COMPLETE**, offline, em GPU T4, com **5/5 encoders**. O pipeline
da amostra de três textos levou **42,09 segundos**, com IDs, rótulos e manifesto
iguais aos locais. Isso mede a execução preparatória, não o tempo ou a acurácia
da reexecução oculta. Na amostra, os três grupos eram unitários e usaram mean5.
O caminho Gaussian foi verificado separadamente no replay de 540 features reais:
395 linhas Gaussian e 145 mean5, com probabilidades idênticas entre notebook e
API local e entre sklearn 1.9.1 e 1.6.1, sem refit.

O notebook usa FP32 na extração, pooling e camadas equivalentes aos caches,
depois features FP16; classificação e agrupamento recebem o teste inteiro.
Falhas ou artefatos incompatíveis encerram a execução; não viram previsões falsas.
O SHA256 do notebook publicado é
`8e8b9b2faad1e30236b54fa357fa47229d03a205be90eb8381527f94531ab92b`.

**Submissão 57958207:** criada em 2026-11-03 às 19:14:17 UTC, versão 4,
arquivo `submission.parquet`; observada COMPLETE às 19:23:16 UTC, com
**nota pública 0,437** e sem erro reportado. Os aproximadamente nove minutos
entre envio e detecção da conclusão incluem fila e intervalo de consulta;
não são uma medição do tempo de inferência oculto.
O acompanhamento está em `runs/kaggle_gaussiano_acompanhamento.jsonl` e o
resultado em `runs/kaggle_gaussiano_resultado.json`. Não houve segundo envio
deste candidato. Restaram duas submissões no dia após esse envio.
Evidências: `runs/kaggle_gaussiano_v4_verificacao.json`,
`runs/kaggle_gaussiano_dataset_verificacao.json`,
`runs/kaggle_gaussiano_submissao_resposta.json` e
`runs/gaussiano_notebook_smoke/cache540_paridade.json`.

| Medição | Método anterior | Novo robusto |
| --- | ---: | ---: |
| Kaggle público | 0,393 (v2, TF-IDF pulado) | **0,437 (v4)** |
| OOF local, cinco dobras por prompt | 0,4467 (mean5 completo) | **0,5516** |

O ganho externo é menor que o local. O novo score fica 11,46 pontos percentuais
abaixo de seu OOF, mas essa diferença não identifica a causa. As linhas da
tabela não são o mesmo conjunto: a v2 remota também não executou o TF-IDF.
Houve mudanças de componentes e de inferência entre versões, portanto não é
possível atribuir todo o +4,4 pp ao Gaussian nem medir sua contribuição isolada
com essas duas submissões. Não temos predições, grupos ou comprimentos ocultos
para decompor os erros ou calcular um intervalo pareado para a nota pública.

O resultado demonstra melhora observada no leaderboard e execução técnica
completa, mas não confirma que a vantagem local de cerca de 10,5 pp se
generalize. A próxima prioridade recomendada é avaliação por fonte/prompts
novos, comprimentos e grupos incompletos, com decisões congeladas. Os ensaios
locais de modelos maiores não demonstraram ganho adicional. Não há evidência
atual para prometer 0,75.

A análise de performance, confusões e seis artigos primários está em
[ANALISE_ARTIGOS.md](ANALISE_ARTIGOS.md). Os próximos experimentos ali propostos
não foram executados nem fazem parte desta submissão.

## Reprodução local

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py preparar
PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py paridade
PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py preparar --com-fallback
PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py paridade --com-fallback
PYTHONPATH=src .venv/bin/python scripts/preparar_gaussiano.py stress --linhas 5000
PYTHONPATH=src .venv/bin/python scripts/estressar_gaussiano_transdutivo.py
```

Os experimentos preservam seus argumentos e resultados em `runs/`. Não há
chamadas a APIs pagas. Os treinos locais usam MPS com lotes físicos pequenos,
padding em múltiplos de 64, limites de memória e checkpoints; isso corrigiu
um travamento inicial causado pelo crescimento do cache Metal.

`runs/validacao_pesada/codigo_sha256.json` conserva o snapshot anterior à
publicação. Ele não cobre as alterações posteriores deste relatório e do pacote
novo. O manifesto em `notebook_gaussiano/manifesto_pacote.json` e as verificações
remotas acima identificam o candidato efetivamente publicado.
