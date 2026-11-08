# Comparação final

Gerado por `python -m src.evaluation.compare`. Não edite à mão.

## Média dos 6 projetos

| Modelo | AUC WP | AUC CP | PR-AUC WP | PR-AUC CP | F1 WP | F1 CP | MCC WP | MCC CP |
|---|---|---|---|---|---|---|---|---|
| lapredict | 0.725 | 0.725 | 0.436 | 0.436 | 0.412 | 0.403 | 0.242 | 0.232 |
| random_forest | 0.758 | 0.737 | 0.488 | 0.460 | 0.504 | 0.471 | 0.308 | 0.283 |
| hist_gradient_boosting | 0.754 | 0.738 | 0.486 | 0.466 | 0.506 | 0.468 | 0.305 | 0.279 |
| mlp | 0.732 | 0.711 | 0.471 | 0.447 | 0.484 | 0.448 | 0.268 | 0.259 |
| cc2vec | 0.706 | 0.675 | 0.431 | 0.414 | — | — | — | — |
| deepjit | 0.706 | 0.670 | 0.433 | 0.413 | — | — | — | — |
| svm | 0.693 | 0.669 | 0.435 | 0.406 | 0.460 | 0.437 | 0.218 | 0.201 |
| lr_jit | 0.689 | 0.667 | 0.433 | 0.406 | 0.466 | 0.432 | 0.227 | 0.201 |
| dbn_jit | 0.671 | 0.663 | — | — | — | — | — | — |

cc2vec, deepjit: AUC e PR-AUC calculados a partir das predições publicadas pelos autores. F1 e MCC ficam de fora porque não há dados de validação para escolher o limiar.

dbn_jit: só a AUC publicada no artigo (Tabela 8).

## Algum modelo supera o lapredict?

**Sim:** random_forest em AUC, hist_gradient_boosting em AUC, hist_gradient_boosting em PR-AUC, random_forest em F1, hist_gradient_boosting em F1, random_forest em MCC, hist_gradient_boosting em MCC.

| Modelo | Métrica | Pares | Vitórias/derrotas | Mediana da diferença | p (Holm) | Cliff's delta | Resultado |
|---|---|---|---|---|---|---|---|
| random_forest | AUC | 12 | 9/3 | +0.0233 | 0.0371 | +0.306 (pequeno) | supera |
| hist_gradient_boosting | AUC | 12 | 10/2 | +0.0157 | 0.0171 | +0.444 (médio) | supera |
| mlp | AUC | 12 | 5/7 | -0.0100 | 0.6772 | -0.194 (pequeno) | empate |
| cc2vec | AUC | 12 | 3/9 | -0.0352 | 0.0483 | -0.444 (médio) | perde |
| deepjit | AUC | 12 | 3/9 | -0.0380 | 0.0483 | -0.500 (grande) | perde |
| svm | AUC | 12 | 2/10 | -0.0342 | 0.0171 | -0.611 (grande) | perde |
| lr_jit | AUC | 12 | 1/11 | -0.0369 | 0.0171 | -0.667 (grande) | perde |
| dbn_jit | AUC | 12 | 0/12 | -0.0571 | 0.0039 | -0.778 (grande) | perde |
| random_forest | PR-AUC | 12 | 9/3 | +0.0450 | 0.0967 | +0.167 (pequeno) | empate |
| hist_gradient_boosting | PR-AUC | 12 | 10/2 | +0.0399 | 0.0171 | +0.194 (pequeno) | supera |
| mlp | PR-AUC | 12 | 8/4 | +0.0262 | 0.4614 | +0.139 (negligível) | empate |
| cc2vec | PR-AUC | 12 | 4/8 | -0.0220 | 1.0000 | -0.028 (negligível) | empate |
| deepjit | PR-AUC | 12 | 4/8 | -0.0139 | 1.0000 | -0.028 (negligível) | empate |
| svm | PR-AUC | 12 | 4/8 | -0.0082 | 1.0000 | +0.000 (negligível) | empate |
| lr_jit | PR-AUC | 12 | 5/7 | -0.0122 | 1.0000 | +0.028 (negligível) | empate |
| random_forest | F1 | 12 | 11/1 | +0.0295 | 0.0098 | +0.319 (pequeno) | supera |
| hist_gradient_boosting | F1 | 12 | 11/1 | +0.0290 | 0.0073 | +0.306 (pequeno) | supera |
| mlp | F1 | 12 | 8/4 | +0.0165 | 0.4541 | +0.208 (pequeno) | empate |
| svm | F1 | 12 | 6/6 | +0.0024 | 1.0000 | +0.125 (negligível) | empate |
| lr_jit | F1 | 12 | 6/6 | -0.0017 | 1.0000 | +0.153 (pequeno) | empate |
| random_forest | MCC | 12 | 11/1 | +0.0299 | 0.0039 | +0.569 (grande) | supera |
| hist_gradient_boosting | MCC | 12 | 12/0 | +0.0333 | 0.0024 | +0.583 (grande) | supera |
| mlp | MCC | 12 | 8/4 | +0.0158 | 0.5288 | +0.306 (pequeno) | empate |
| svm | MCC | 12 | 3/9 | -0.0313 | 0.5288 | -0.278 (pequeno) | empate |
| lr_jit | MCC | 12 | 3/9 | -0.0296 | 0.5288 | -0.250 (pequeno) | empate |

Pares = projeto × cenário, com a média das seeds de cada modelo. O p-valor é do Wilcoxon pareado bicaudal, corrigido por Holm entre os modelos de cada métrica. Magnitude do Cliff's delta pelos limiares de Romano et al. (2006).

**Limitação do CP:** o treino usa os outros cinco projetos inteiros, então parte dos commits de treino é mais recente que o período de teste. O artigo faz assim e mantivemos para comparar nas mesmas condições.

Figuras em `results/figures/`.
