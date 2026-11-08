# Predição de Defeitos Just-in-Time com Machine Learning

[![Testes](https://github.com/alana-britoc/Projeto-IA/actions/workflows/testes.yml/badge.svg)](https://github.com/alana-britoc/Projeto-IA/actions/workflows/testes.yml)

Projeto da disciplina de Inteligência Artificial. A equipe replica e estende o artigo *Deep Just-in-Time Defect Prediction: How Far Are We?* (Zeng et al., ISSTA 2021).

## Equipe

**Nome da equipe:** Commit Limpo

| Integrante | Conta no GitHub | Responsabilidade |
| --- | --- | --- |
| Alana de Brito Cavalcanti | `@alana-britoc` | Carregamento dos dados, reprodução do LApredict e do LR-JIT, script de experimentos e README |
| João Araújo | `@joaogabrielarj` | Random Forest, HistGradientBoosting, SVM e MLP |
| Luiz Tenório | `@luiztcn78` | Métricas do DeepJIT e do CC2Vec a partir dos resultados dos autores |
| Ocimar Schroeder Junior | `@OcimarSchroederJR` | Comparação final: tabelas, gráficos e testes estatísticos |

## Descrição geral

Predição de defeitos *just-in-time* (JIT) tenta prever, no momento do commit, se ele vai introduzir um defeito. Com isso a revisão de código pode priorizar os commits mais arriscados enquanto a mudança ainda está fresca para quem a fez.

O artigo-base comparou modelos de deep learning (DeepJIT e CC2Vec) com modelos simples. A conclusão foi que o LApredict, uma regressão logística que usa só o número de linhas adicionadas (`la`), fica no mesmo nível ou acima dos modelos profundos.

**Objetivo:** reproduzir os baselines do artigo e testar se outros modelos clássicos de Machine Learning superam o LApredict nas mesmas condições. Os modelos testados são Random Forest, HistGradientBoosting, SVM e MLP, todos com as 14 features de Kamei.

**Dados:** commits de seis projetos open source (qt, openstack, jdt, platform, gerrit e go), disponibilizados pelos autores do artigo. Cada commit tem 14 features de Kamei e um rótulo indicando se introduziu defeito. A origem e o formato estão em [`data/README.md`](data/README.md).

**Cenários avaliados:**

- **WP (within-project):** treino nos primeiros 80% dos commits do projeto, em ordem cronológica, e teste nos 20% finais
- **CP (cross-project):** treino nos outros cinco projetos e teste nos mesmos 20% finais

**Métricas:** AUC, PR-AUC, F1 e MCC. O limiar de decisão do F1 e do MCC é escolhido nos 20% finais do treino, nunca no teste.

## Estrutura do repositório

```text
.
├── data/
│   ├── raw/                  # datasets dos autores (não versionados)
│   ├── processed/
│   ├── reference/            # Tabela 8 do artigo, para comparação
│   └── README.md
├── src/
│   ├── app.py                # painel (Streamlit), com as duas páginas abaixo
│   ├── resultados.py         # página de resultados
│   ├── dashboard.py          # dados e gráficos da página de resultados
│   ├── simulador.py          # página do simulador de commit
│   ├── data/loader.py        # leitura dos CSVs e divisão WP/CP
│   ├── preprocessing/        # padronização das features
│   ├── models/
│   │   ├── registry.py        # modelos e espaços de busca de hiperparâmetros
│   │   ├── train_simulator.py # treina o modelo do simulador
│   │   └── simulator.py       # carrega o modelo e calcula a probabilidade
│   └── evaluation/
│       ├── metrics.py        # AUC, PR-AUC, F1, MCC e escolha do limiar
│       ├── reproduce.py      # reprodução do LApredict e do LR-JIT
│       ├── run_experiment.py # grade modelo x projeto x cenário x seed
│       ├── deep_baselines.py # métricas do DeepJIT e do CC2Vec
│       ├── compare.py        # comparação final contra o LApredict
│       ├── feature_importance.py # importância das features nos modelos de árvore
│       ├── stats.py          # Wilcoxon, Holm e Cliff's delta
│       └── plots.py          # gráficos da comparação
├── models/
│   └── simulador.joblib      # Random Forest do simulador (versionado)
├── tests/
├── notebooks/
├── results/
│   ├── comparacao.md         # relatório da comparação final
│   ├── figures/              # gráficos da comparação
│   └── metrics/              # CSVs gerados pelos scripts
└── requirements.txt
```

## Configuração do ambiente

Use Python 3.10 ou mais recente.

```bash
git clone https://github.com/alana-britoc/Projeto-IA.git
cd Projeto-IA
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Linux ou macOS
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois baixe os dados seguindo [`data/README.md`](data/README.md). Os seis projetos devem ficar em `data/raw/`.

## Como rodar

Todos os comandos são executados a partir da raiz do repositório.

**Reproduzir os baselines do artigo** (LApredict e LR-JIT), comparando com a Tabela 8:

```bash
python -m src.evaluation.reproduce
```

O resultado é salvo em `results/metrics/reproducao.csv`.

**Rodar os experimentos.** Sem argumentos, o script roda todos os modelos, projetos e cenários com a seed 0:

```bash
python -m src.evaluation.run_experiment
```

Também é possível rodar só uma parte da grade:

```bash
python -m src.evaluation.run_experiment --models random_forest svm --projects qt go --settings WP --seeds 0 1 2 3 4
```

| Opção | Valores |
| --- | --- |
| `--models` | `lapredict`, `lr_jit`, `random_forest`, `hist_gradient_boosting`, `lightgbm`, `xgboost`, `svm`, `mlp` |
| `--projects` | `qt`, `openstack`, `jdt`, `platform`, `gerrit`, `go` |
| `--settings` | `WP`, `CP` |
| `--seeds` | uma ou mais seeds inteiras |
| `--n-iter` | combinações testadas na busca de hiperparâmetros (padrão 20, 0 desliga) |

Cada modelo é salvo em `results/metrics/experimentos_<modelo>.csv`. Rodar de novo a mesma combinação substitui só aquela linha. Os CSVs versionados foram gerados com as seeds 0 a 4.

**Calcular as métricas do DeepJIT e do CC2Vec** a partir das predições publicadas pelos autores. É preciso clonar o [repositório dos autores](https://github.com/ZZR0/ISSTA21-JIT-DP) ao lado deste projeto:

```bash
python -m src.evaluation.deep_baselines ../ISSTA21-JIT-DP/ResultAnalysis/RQ4-T8
```

O resultado é salvo em `results/metrics/deep_baselines.csv`.

**Gerar a comparação final** contra o LApredict:

```bash
python -m src.evaluation.compare
```

O script lê todos os `results/metrics/experimentos_*.csv` e o `deep_baselines.csv`, e usa a AUC publicada no artigo para o DBN-JIT. A média das seeds de cada modelo é comparada com o LApredict pelo teste de Wilcoxon pareado, com correção de Holm, e pelo Cliff's delta. Cada par é um projeto × cenário. O script gera:

- `results/metrics/comparacao_<metrica>.csv`: AUC, PR-AUC, F1 e MCC por modelo, projeto e cenário
- `results/metrics/comparacao_testes.csv`: resultado dos testes estatísticos
- `results/figures/*.png`: gráficos comparativos
- `results/comparacao.md`: relatório com a resposta à pergunta "algum modelo supera o LApredict?"

Um modelo novo entra na comparação sozinho: basta existir o `experimentos_<modelo>.csv` e rodar o comando de novo. A única coisa a fazer à mão é dar uma cor ao modelo em `MODEL_COLORS` (`src/evaluation/plots.py`), e um teste avisa se faltar. Os 8 modelos do grupo já ocupam as 8 cores da paleta, que foi checada para daltonismo, então um nono modelo precisa substituir algum ou ficar como referência em cinza, como o DeepJIT, o CC2Vec e o DBN-JIT.

**Medir a importância das features** no Random Forest e no HistGradientBoosting (precisa do dataset):

```bash
python -m src.evaluation.feature_importance
```

Os dois modelos são treinados com os hiperparâmetros da busca (seed 0), e a importância é a queda da AUC no teste quando cada feature é embaralhada. O script também compara o HistGradientBoosting com uma versão que não pode combinar features, para medir quanto as interações ajudam. Gera `results/metrics/importancia_features.csv`, `results/metrics/importancia_interacoes.csv` e `results/figures/importancia_features.png`.

**Abrir o painel** no navegador:

```bash
streamlit run src/app.py
```

O painel abre em http://localhost:8501 e tem duas páginas, **Resultados** e **Simulador de commit**, escolhidas no menu lateral. Nenhuma das duas precisa do dataset.

A página de resultados lê os CSVs versionados em `results/metrics/` e usa as mesmas funções do `compare.py`, então mostra os mesmos números de `results/comparacao.md`. Na barra lateral dá para escolher a métrica e filtrar por modelo, projeto e cenário. A página mostra:

- a resposta à pergunta "algum modelo supera o LApredict?";
- a média da métrica por modelo e o valor em cada projeto e cenário, em gráficos interativos;
- os testes de Wilcoxon e o Cliff's delta contra o LApredict;
- a tabela completa dos resultados.

Os testes estatísticos usam sempre os 12 pares projeto × cenário. Os filtros mudam os gráficos e as tabelas, mas não os testes.

No simulador, a pessoa preenche as 14 métricas de Kamei de um commit e vê a probabilidade de ele introduzir defeito, além de quanto ele é mais arriscado que os commits do treino. Os campos começam com a mediana dos commits do treino. É uma demonstração: o modelo só vale para commits parecidos com os dos 6 projetos.

O modelo do simulador é um Random Forest treinado com os 94.918 commits dos 6 projetos e salvo em `models/simulador.joblib`. Ele usa os hiperparâmetros que a busca escolheu mais vezes nos experimentos, com 100 árvores e profundidade máxima 12 para o arquivo ficar abaixo de 10 MB. Para treinar de novo, por exemplo depois de atualizar o scikit-learn, é preciso ter o dataset em `data/raw/`:

```bash
python -m src.models.train_simulator
```

**Testes:**

```bash
python -m pytest
```

Os mesmos testes rodam automaticamente no GitHub Actions em todo PR e em todo push na `main`.

## Resultados

| Modelo | AUC WP | AUC CP |
| --- | --- | --- |
| Random Forest | 0,758 | 0,737 |
| HistGradientBoosting | 0,754 | 0,738 |
| MLP | 0,738 | 0,709 |
| **LApredict** | **0,725** | **0,725** |
| DeepJIT | 0,706 | 0,670 |
| CC2Vec | 0,706 | 0,675 |
| SVM | 0,693 | 0,669 |
| LR-JIT | 0,689 | 0,667 |
| DBN-JIT | 0,671 | 0,663 |

Média dos 6 projetos. A reprodução do LApredict chegou à mesma AUC média publicada no artigo (0,725).

O Random Forest e o HistGradientBoosting superam o LApredict com diferença significativa: o HistGradientBoosting nas quatro métricas, e o Random Forest em AUC, F1 e MCC. O ganho é pequeno, de 0,02 a 0,03 de AUC. O MLP empata com o LApredict, e os demais modelos perdem em AUC, o que confirma a conclusão do artigo sobre os modelos de deep learning.

As tabelas completas, os testes estatísticos e as limitações estão em [`results/comparacao.md`](results/comparacao.md).

## Fluxo de desenvolvimento

```text
Issue → Branch → Commits → Pull Request → Code Review → Approval → Merge
```

1. Toda alteração de código deve estar associada a uma Issue.
2. Cada atividade é desenvolvida em uma branch própria, sem commits diretos na `main`.
3. Toda alteração entra na `main` por Pull Request, com `Closes #numero_da_issue` na descrição quando concluir a atividade.
4. Todo PR precisa de pelo menos uma aprovação de outro integrante antes do merge. O autor não aprova o próprio PR.
5. Cada integrante usa sua própria conta do GitHub e faz seus próprios commits.

## Referência

Zhengran Zeng, Yuqun Zhang, Haotian Zhang e Lingming Zhang. *Deep Just-in-Time Defect Prediction: How Far Are We?* ISSTA 2021. https://dl.acm.org/doi/10.1145/3460319.3464819