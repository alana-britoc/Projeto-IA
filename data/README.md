# Dados do projeto

Este diretório organiza os dados utilizados no projeto.

- `raw/`: armazene aqui os datasets originais, sem alterações.
- `processed/`: armazene aqui os dados resultantes de limpeza, transformação ou preparação.
- `reference/`: resultados publicados no artigo-base, usados para comparação.

Os arquivos de `raw/` e `processed/` são ignorados pelo Git por padrão. Apenas os arquivos `.gitkeep` são versionados para preservar a estrutura dos diretórios.

## Origem

Os dados vêm do artigo *Deep Just-in-Time Defect Prediction: How Far Are We?* (Zeng et al., ISSTA 2021):

- Artigo: https://dl.acm.org/doi/10.1145/3460319.3464819
- Repositório dos autores (licença MIT): https://github.com/ZZR0/ISSTA21-JIT-DP
- Pacote arquivado: https://zenodo.org/records/4727756

## Como obter

1. Baixe o `datasets.tar.gz` pelo link de Google Drive indicado no README do repositório dos autores (seção *Retraining Evaluation*). São cerca de 15 GB depois de extraído.
2. Extraia e copie as pastas dos seis projetos para `data/raw/`:

```text
data/raw/
├── qt/
├── openstack/
├── jdt/
├── platform/
├── gerrit/
└── go/
```

## Conteúdo usado até agora

Cada commit é uma linha, com rótulo `bug` (1 = commit que introduziu defeito, identificado por SZZ) e 14 features de Kamei: `ns, nd, nf, entrophy, la, ld, lt, fix, ndev, age, nuc, exp, rexp, sexp`.

| Arquivo | Uso |
| --- | --- |
| `raw/<projeto>/<projeto>_k_feature.csv` | Within-project (WP): primeiros 80% dos commits para treino, últimos 20% para teste |
| `raw/<projeto>/cross/k_train.csv` | Cross-project (CP): commits dos outros cinco projetos |
| `raw/<projeto>/cross/k_test.csv` | Cross-project (CP): teste, os mesmos 20% finais do WP |

Os arquivos `.pkl` em `raw/<projeto>/deepjit/` trazem `(ids, labels, msgs, codes)`, com a mensagem e o diff de cada commit já tokenizados.

| Projeto | Commits |
| --- | --- |
| qt | 23.912 |
| openstack | 22.757 |
| jdt | 3.279 |
| platform | 11.034 |
| gerrit | 14.927 |
| go | 19.009 |

## Referência

`reference/rq4_t8.csv` é a Tabela 8 do artigo (AUC por projeto, WP e CP), copiada de `ResultAnalysis/RQ4-T8/result.csv` no repositório dos autores.
