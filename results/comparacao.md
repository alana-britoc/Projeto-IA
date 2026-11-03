# Comparação final

Gerado por `python -m src.evaluation.compare`. Não edite à mão.

## Média dos 6 projetos

| Modelo | AUC WP | AUC CP | PR-AUC WP | PR-AUC CP | F1 WP | F1 CP | MCC WP | MCC CP |
|---|---|---|---|---|---|---|---|---|
| lapredict | 0.725 | 0.725 | 0.436 | 0.436 | 0.412 | 0.403 | 0.242 | 0.232 |
| cc2vec | 0.706 | 0.675 | — | — | — | — | — | — |
| deepjit | 0.706 | 0.670 | — | — | — | — | — | — |
| lr_jit | 0.689 | 0.667 | 0.433 | 0.406 | 0.466 | 0.432 | 0.227 | 0.201 |
| dbn_jit | 0.671 | 0.663 | — | — | — | — | — | — |

cc2vec, dbn_jit, deepjit: só a AUC publicada no artigo (Tabela 8); as outras métricas dependem da #6.

## Algum modelo supera o lapredict?

**Não.** Nenhum modelo foi significativamente melhor que o lapredict em nenhuma métrica (Wilcoxon pareado com Holm, alfa = 0.05, e Cliff's delta não negligível).

| Modelo | Métrica | Pares | Vitórias/derrotas | Mediana da diferença | p (Holm) | Cliff's delta | Resultado |
|---|---|---|---|---|---|---|---|
| cc2vec | AUC | 12 | 3/9 | -0.0352 | 0.0322 | -0.444 (médio) | perde |
| deepjit | AUC | 12 | 3/9 | -0.0380 | 0.0322 | -0.500 (grande) | perde |
| lr_jit | AUC | 12 | 1/11 | -0.0369 | 0.0073 | -0.667 (grande) | perde |
| dbn_jit | AUC | 12 | 0/12 | -0.0571 | 0.0020 | -0.778 (grande) | perde |
| lr_jit | PR-AUC | 12 | 5/7 | -0.0122 | 0.3804 | +0.028 (negligível) | empate |
| lr_jit | F1 | 12 | 6/6 | -0.0017 | 0.9097 | +0.153 (pequeno) | empate |
| lr_jit | MCC | 12 | 3/9 | -0.0296 | 0.2334 | -0.250 (pequeno) | empate |

Pares = projeto × cenário, com a média das seeds de cada modelo. O p-valor é do Wilcoxon pareado bicaudal, corrigido por Holm entre os modelos de cada métrica. Magnitude do Cliff's delta pelos limiares de Romano et al. (2006).

Figuras em `results/figures/`.
