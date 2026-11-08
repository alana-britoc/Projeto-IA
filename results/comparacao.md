# Comparação final

Gerado por `python -m src.evaluation.compare`. Não edite à mão.

## Média dos 6 projetos

| Modelo | AUC WP | AUC CP | PR-AUC WP | PR-AUC CP | F1 WP | F1 CP | MCC WP | MCC CP | Popt WP | Popt CP | Recall@20% WP | Recall@20% CP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lapredict | 0.725 | 0.725 | 0.436 | 0.436 | 0.412 | 0.403 | 0.242 | 0.232 | 0.872 | 0.873 | 0.691 | 0.685 |
| lightgbm | 0.760 | 0.740 | 0.492 | 0.469 | 0.507 | 0.472 | 0.310 | 0.283 | 0.883 | 0.881 | 0.707 | 0.706 |
| xgboost | 0.757 | 0.739 | 0.489 | 0.468 | 0.503 | 0.471 | 0.304 | 0.285 | 0.883 | 0.881 | 0.706 | 0.705 |
| random_forest | 0.758 | 0.737 | 0.488 | 0.460 | 0.504 | 0.471 | 0.308 | 0.283 | 0.882 | 0.879 | 0.705 | 0.702 |
| hist_gradient_boosting | 0.754 | 0.738 | 0.486 | 0.466 | 0.506 | 0.468 | 0.305 | 0.279 | 0.882 | 0.881 | 0.705 | 0.704 |
| mlp | 0.732 | 0.711 | 0.471 | 0.447 | 0.484 | 0.448 | 0.268 | 0.259 | 0.876 | 0.869 | 0.696 | 0.689 |
| cc2vec | 0.706 | 0.675 | 0.431 | 0.414 | — | — | — | — | — | — | — | — |
| deepjit | 0.706 | 0.670 | 0.433 | 0.413 | — | — | — | — | — | — | — | — |
| svm | 0.693 | 0.669 | 0.435 | 0.406 | 0.460 | 0.437 | 0.218 | 0.201 | 0.874 | 0.872 | 0.689 | 0.689 |
| lr_jit | 0.689 | 0.667 | 0.433 | 0.406 | 0.466 | 0.432 | 0.227 | 0.201 | 0.867 | 0.872 | 0.682 | 0.690 |
| dbn_jit | 0.671 | 0.663 | — | — | — | — | — | — | — | — | — | — |

cc2vec, deepjit: AUC e PR-AUC calculados a partir das predições publicadas pelos autores. F1 e MCC ficam de fora porque não há dados de validação para escolher o limiar, e Popt e Recall@20% porque as predições não foram cruzadas com as linhas de cada commit.

dbn_jit: só a AUC publicada no artigo (Tabela 8).

**Referência sem modelo para Popt e Recall@20%:** revisar do menor commit para o maior, sem usar nenhum modelo, já dá WP: Popt 0.872, Recall@20% 0.683; CP: Popt 0.872, Recall@20% 0.683. Como a ordem de revisão é a densidade de risco (probabilidade ÷ linhas), o tamanho do commit pesa muito nessas métricas, e o mérito de um modelo é só o quanto ele passa dessa referência (ver Yang et al., 2016).

## Algum modelo supera o lapredict?

**Sim:** lightgbm em AUC, xgboost em AUC, random_forest em AUC, hist_gradient_boosting em AUC, lightgbm em PR-AUC, xgboost em PR-AUC, hist_gradient_boosting em PR-AUC, lightgbm em F1, xgboost em F1, random_forest em F1, hist_gradient_boosting em F1, lightgbm em MCC, xgboost em MCC, random_forest em MCC, hist_gradient_boosting em MCC, lightgbm em Popt, xgboost em Popt, random_forest em Popt, hist_gradient_boosting em Popt, lightgbm em Recall@20%, xgboost em Recall@20%, random_forest em Recall@20%, hist_gradient_boosting em Recall@20%.

| Modelo | Métrica | Pares | Vitórias/derrotas | Mediana da diferença | p (Holm) | Cliff's delta | Resultado |
|---|---|---|---|---|---|---|---|
| lightgbm | AUC | 12 | 11/1 | +0.0186 | 0.0088 | +0.556 (grande) | supera |
| xgboost | AUC | 12 | 10/2 | +0.0199 | 0.0195 | +0.417 (médio) | supera |
| random_forest | AUC | 12 | 9/3 | +0.0233 | 0.0371 | +0.306 (pequeno) | supera |
| hist_gradient_boosting | AUC | 12 | 10/2 | +0.0157 | 0.0195 | +0.444 (médio) | supera |
| mlp | AUC | 12 | 5/7 | -0.0100 | 0.6772 | -0.194 (pequeno) | empate |
| cc2vec | AUC | 12 | 3/9 | -0.0352 | 0.0483 | -0.444 (médio) | perde |
| deepjit | AUC | 12 | 3/9 | -0.0380 | 0.0483 | -0.500 (grande) | perde |
| svm | AUC | 12 | 2/10 | -0.0342 | 0.0195 | -0.611 (grande) | perde |
| lr_jit | AUC | 12 | 1/11 | -0.0369 | 0.0195 | -0.667 (grande) | perde |
| dbn_jit | AUC | 12 | 0/12 | -0.0571 | 0.0049 | -0.778 (grande) | perde |
| lightgbm | PR-AUC | 12 | 11/1 | +0.0412 | 0.0088 | +0.222 (pequeno) | supera |
| xgboost | PR-AUC | 12 | 10/2 | +0.0444 | 0.0239 | +0.194 (pequeno) | supera |
| random_forest | PR-AUC | 12 | 9/3 | +0.0450 | 0.0967 | +0.167 (pequeno) | empate |
| hist_gradient_boosting | PR-AUC | 12 | 10/2 | +0.0399 | 0.0195 | +0.194 (pequeno) | supera |
| mlp | PR-AUC | 12 | 8/4 | +0.0262 | 0.4614 | +0.139 (negligível) | empate |
| cc2vec | PR-AUC | 12 | 4/8 | -0.0220 | 1.0000 | -0.028 (negligível) | empate |
| deepjit | PR-AUC | 12 | 4/8 | -0.0139 | 1.0000 | -0.028 (negligível) | empate |
| svm | PR-AUC | 12 | 4/8 | -0.0082 | 1.0000 | +0.000 (negligível) | empate |
| lr_jit | PR-AUC | 12 | 5/7 | -0.0122 | 1.0000 | +0.028 (negligível) | empate |
| lightgbm | F1 | 12 | 11/1 | +0.0270 | 0.0068 | +0.319 (pequeno) | supera |
| xgboost | F1 | 12 | 11/1 | +0.0272 | 0.0088 | +0.319 (pequeno) | supera |
| random_forest | F1 | 12 | 11/1 | +0.0295 | 0.0098 | +0.319 (pequeno) | supera |
| hist_gradient_boosting | F1 | 12 | 11/1 | +0.0290 | 0.0088 | +0.306 (pequeno) | supera |
| mlp | F1 | 12 | 8/4 | +0.0165 | 0.4541 | +0.208 (pequeno) | empate |
| svm | F1 | 12 | 6/6 | +0.0024 | 1.0000 | +0.125 (negligível) | empate |
| lr_jit | F1 | 12 | 6/6 | -0.0017 | 1.0000 | +0.153 (pequeno) | empate |
| lightgbm | MCC | 12 | 12/0 | +0.0333 | 0.0034 | +0.611 (grande) | supera |
| xgboost | MCC | 12 | 12/0 | +0.0322 | 0.0034 | +0.597 (grande) | supera |
| random_forest | MCC | 12 | 11/1 | +0.0299 | 0.0039 | +0.569 (grande) | supera |
| hist_gradient_boosting | MCC | 12 | 12/0 | +0.0333 | 0.0034 | +0.583 (grande) | supera |
| mlp | MCC | 12 | 8/4 | +0.0158 | 0.5288 | +0.306 (pequeno) | empate |
| svm | MCC | 12 | 3/9 | -0.0313 | 0.5288 | -0.278 (pequeno) | empate |
| lr_jit | MCC | 12 | 3/9 | -0.0296 | 0.5288 | -0.250 (pequeno) | empate |
| lightgbm | Popt | 12 | 12/0 | +0.0069 | 0.0034 | +0.167 (pequeno) | supera |
| xgboost | Popt | 12 | 12/0 | +0.0075 | 0.0034 | +0.167 (pequeno) | supera |
| random_forest | Popt | 12 | 12/0 | +0.0065 | 0.0034 | +0.167 (pequeno) | supera |
| hist_gradient_boosting | Popt | 12 | 12/0 | +0.0069 | 0.0034 | +0.167 (pequeno) | supera |
| mlp | Popt | 12 | 5/7 | -0.0008 | 1.0000 | -0.028 (negligível) | empate |
| svm | Popt | 12 | 7/5 | +0.0016 | 1.0000 | +0.028 (negligível) | empate |
| lr_jit | Popt | 12 | 6/6 | +0.0004 | 1.0000 | +0.000 (negligível) | empate |
| lightgbm | Recall@20% | 12 | 12/0 | +0.0117 | 0.0034 | +0.222 (pequeno) | supera |
| xgboost | Recall@20% | 12 | 12/0 | +0.0118 | 0.0034 | +0.222 (pequeno) | supera |
| random_forest | Recall@20% | 12 | 12/0 | +0.0097 | 0.0034 | +0.222 (pequeno) | supera |
| hist_gradient_boosting | Recall@20% | 12 | 12/0 | +0.0101 | 0.0034 | +0.222 (pequeno) | supera |
| mlp | Recall@20% | 12 | 6/6 | +0.0004 | 1.0000 | +0.056 (negligível) | empate |
| svm | Recall@20% | 12 | 7/5 | +0.0014 | 1.0000 | +0.056 (negligível) | empate |
| lr_jit | Recall@20% | 12 | 6/5 | +0.0006 | 1.0000 | +0.035 (negligível) | empate |

Pares = projeto × cenário, com a média das seeds de cada modelo. O p-valor é do Wilcoxon pareado bicaudal, corrigido por Holm entre os modelos de cada métrica. Magnitude do Cliff's delta pelos limiares de Romano et al. (2006).

**Limitação do CP:** o treino usa os outros cinco projetos inteiros, então parte dos commits de treino é mais recente que o período de teste. O artigo faz assim e mantivemos para comparar nas mesmas condições.

Figuras em `results/figures/`.
