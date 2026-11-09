"""Pagina de resultados do painel (#19). Abre pelo src/app.py."""
import streamlit as st

from src import dashboard
from src.data.loader import PROJECTS
from src.evaluation.compare import ALPHA, BASELINE
from src.evaluation.metrics import METRIC_NAMES


@st.cache_data
def carregar():
    return dashboard.load_comparison()


comp, tests, order = carregar()

with st.sidebar:
    st.header("Filtros")
    metric = st.selectbox("Métrica", list(METRIC_NAMES), format_func=METRIC_NAMES.get)
    models = st.multiselect("Modelos", order, default=order, format_func=dashboard.model_name)
    projects = st.multiselect("Projetos", PROJECTS, default=PROJECTS)
    available = [s for s in dashboard.SETTINGS if s in set(comp["setting"])]
    settings = st.multiselect("Cenários", available, default=available,
                              format_func=dashboard.SETTING_NAMES.get,
                              help="WP: treino nos primeiros 80% dos commits do projeto e teste "
                                   "nos últimos 20%. CP: treino nos outros cinco projetos, como "
                                   "no artigo. CP cronológico: o mesmo teste do CP, mas só com "
                                   "os commits dos outros projetos anteriores ao teste.")
    comparisons = list(dict.fromkeys(tests["cenarios"]))
    comparison = st.radio("Comparação estatística", comparisons,
                          format_func=dashboard.COMPARISON_NAMES.get,
                          help="Cenários usados nos testes contra o LApredict. No CP do artigo, "
                               "parte do treino é mais recente que o teste.")

name = METRIC_NAMES[metric]
tests = tests[tests["cenarios"] == comparison]
realista = comparison == "WP+CP_crono"

st.title("Predição de commits que introduzem defeito")
st.caption("Resultados do grupo Commit Limpo em 6 projetos, comparados com o LApredict "
           "(ISSTA 2021). Os dados vêm de `results/metrics/`.")

st.subheader(f"Algum modelo supera o {BASELINE}"
             + (" no cenário realista (WP + CP cronológico)?" if realista else "?"))
st.markdown(dashboard.headline(tests))
st.caption(f"Wilcoxon pareado com correção de Holm (alfa = {ALPHA}) e Cliff's delta não "
           f"negligível. Os testes usam sempre os 12 pares projeto × cenário de "
           f"{dashboard.COMPARISON_NAMES[comparison]}, sem os filtros.")

filtered = dashboard.filter_comparison(comp, models, projects, settings)
with_metric = filtered.dropna(subset=[metric])
missing = [dashboard.model_name(m) for m in models if m not in set(with_metric["model"])]

ganho, calor, realista_tab, pontos, tabela = st.tabs(
    ["Ganho sobre o LApredict", "Mapa de calor", "Cenário realista", "Pontos", "Tabela"])

with ganho:
    st.subheader(f"Quanto cada modelo ganha do {BASELINE}: {name}")
    gains = dashboard.gain_table(comp, tests, metric, comparison, models)
    if gains.empty:
        st.info(f"Nenhum modelo escolhido tem {name} para comparar com o {BASELINE}.")
    else:
        st.altair_chart(dashboard.gain_chart(gains, metric), width="stretch")
        st.caption(f"Média da diferença para o {BASELINE} nos 12 pares projeto × cenário de "
                   f"{dashboard.COMPARISON_NAMES[comparison]}, os mesmos dos testes. Verde: "
                   "supera com diferença significativa; cinza: empate; laranja: perde.")
    st.subheader(f"Testes contra o {BASELINE}: {name}, "
                 f"{dashboard.COMPARISON_NAMES[comparison]}")
    table = dashboard.tests_table(tests, metric, models)
    if table.empty:
        st.info(f"Nenhum modelo escolhido tem teste de {name}.")
    else:
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Modelo": st.column_config.TextColumn(width="medium"),
            **{c: st.column_config.NumberColumn(format="%.4f")
               for c in ["Mediana da diferença", "p (Holm)", "Cliff's delta"]}})

with calor:
    escolhidos = [s for s in dashboard.SETTINGS if s in set(with_metric["setting"])]
    if not escolhidos:
        st.info(f"Nenhum resultado de {name} com os filtros escolhidos.")
    else:
        cenario = st.radio("Cenário", escolhidos, horizontal=True,
                           format_func=dashboard.SETTING_NAMES.get, key="calor_cenario")
        st.subheader(f"{name} por modelo e projeto: {dashboard.SETTING_NAMES[cenario]}")
        st.altair_chart(dashboard.heatmap_chart(with_metric, metric, order, cenario),
                        width="stretch")
        st.caption("Quanto mais escura a célula, maior o valor. Passe o mouse para ver o "
                   "número completo.")

with realista_tab:
    st.subheader(f"Tirar o futuro do treino muda o resultado? {name}")
    if not {"CP", "CP_crono"} <= set(with_metric["setting"]):
        st.info("Escolha os cenários CP e CP cronológico nos filtros para ver esta comparação.")
    else:
        st.altair_chart(dashboard.slope_chart(with_metric, metric, order), width="stretch")
        st.caption("Cada linha liga a média de um modelo no CP do artigo, em que parte do treino "
                   "é mais recente que o teste, à média no CP cronológico, que só usa commits "
                   "anteriores ao teste. Linha quase reta: o resultado não dependia do futuro. "
                   "Os modelos do artigo não aparecem porque não têm o cenário cronológico.")

with pontos:
    if with_metric.empty:
        st.info(f"Nenhum resultado de {name} com os filtros escolhidos.")
    else:
        st.subheader(f"{name} média por modelo")
        st.altair_chart(dashboard.mean_chart(with_metric, metric, order), width="stretch")
        st.subheader(f"{name} por projeto e cenário")
        st.altair_chart(dashboard.project_chart(with_metric, metric, order))

with tabela:
    st.subheader("Tabela completa")
    st.dataframe(
        filtered.assign(model=filtered["model"].map(dashboard.model_name)).rename(
            columns={"model": "Modelo", "project": "Projeto", "setting": "Cenário",
                     "source": "Origem", **METRIC_NAMES}),
        hide_index=True, width="stretch",
        column_config={n: st.column_config.NumberColumn(format="%.4f")
                       for n in METRIC_NAMES.values()})
    st.caption("Origem: **nosso** = média das 5 seeds rodadas pelo grupo; **autores** = "
               "calculado a partir das predições publicadas pelos autores; **artigo** = AUC da "
               "Tabela 8.")

if missing:
    st.caption(f"Sem {name}: {', '.join(missing)}. O DBN-JIT só tem a AUC publicada no "
               "artigo. O DeepJIT e o CC2Vec não têm F1 nem MCC, porque não há validação "
               "para escolher o limiar, nem Popt e Recall@20%, que precisam das linhas de "
               "cada commit cruzadas com as predições dos autores.")
