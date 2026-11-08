"""Dados e graficos do painel de resultados (#19). A pagina fica em src/app.py.

Reaproveita as funcoes da comparacao final (#7), entao o painel mostra os mesmos numeros
de results/comparacao.md. Le so os CSVs versionados, nao precisa do dataset.
"""
import altair as alt

import pandas as pd

from src.data.loader import PROJECTS, SETTINGS
from src.evaluation.compare import (METRICS_DIR, all_tests, answer, available_comparisons,
                                    build_comparison, load_results, model_order)
from src.evaluation.metrics import METRIC_NAMES
from src.evaluation.plots import MUTED, REFERENCE_MARKERS, SETTING_COLORS, model_colors

SETTING_NAMES = {"WP": "WP (mesmo projeto)", "CP": "CP (outros projetos)",
                 "CP_crono": "CP cronológico (outros projetos, só o passado)"}
# comparacoes estatisticas contra o LApredict (#24), pela coluna "cenarios" dos testes
COMPARISON_NAMES = {"WP+CP": "WP + CP (como no artigo)",
                    "WP+CP_crono": "WP + CP cronológico (realista)"}


def load_comparison(metrics_dir=METRICS_DIR):
    """Retorna (comp, tests, order): resultados por projeto x cenario, testes das comparacoes
    disponiveis e ordem dos modelos."""
    comp = build_comparison(load_results(metrics_dir))
    order = model_order(comp)
    # lapredict primeiro e depois por AUC, igual nas figuras; projetos na ordem do artigo
    rank = {"model": order, "project": PROJECTS, "setting": SETTINGS}
    comp = comp.sort_values(["model", "project", "setting"],
                            key=lambda s: s.map({v: i for i, v in enumerate(rank[s.name])}))
    tests = pd.concat([all_tests(comp, settings=s) for s in available_comparisons(comp)],
                      ignore_index=True)
    return comp.reset_index(drop=True), tests, order


def headline(tests):
    # a frase de resposta e a 3a linha do texto do relatorio (titulo, linha vazia, resposta)
    return answer(tests).splitlines()[2]


def filter_comparison(comp, models, projects, settings):
    keep = (comp["model"].isin(models) & comp["project"].isin(projects)
            & comp["setting"].isin(settings))
    return comp[keep]


def mean_by_model(comp, metric):
    """Media dos projetos por modelo e cenario. Modelo sem a metrica (ex.: dbn_jit no F1) sai."""
    out = comp.groupby(["model", "setting"], as_index=False)[metric].mean()
    return out.dropna(subset=[metric]).rename(columns={metric: "valor"})


def tests_table(tests, metric, models):
    out = tests[(tests["metric"] == metric) & tests["model"].isin(models)]
    out = out.assign(placar=out["wins"].astype(str) + "/" + out["losses"].astype(str))
    out = out[["model", "n", "placar", "median_diff", "p_holm", "delta", "magnitude", "verdict"]]
    return out.rename(columns={"model": "Modelo", "n": "Pares", "placar": "Vitórias/derrotas",
                               "median_diff": "Mediana da diferença", "p_holm": "p (Holm)",
                               "delta": "Cliff's delta", "magnitude": "Magnitude",
                               "verdict": "Resultado"}).reset_index(drop=True)


def _model_scale(order):
    colors = model_colors(order)
    return alt.Scale(domain=list(colors), range=list(colors.values()))


def mean_chart(comp, metric, order):
    """Um ponto por cenario em cada modelo, ligados por uma linha.

    Pontos em vez de barras: as medias ficam entre 0,6 e 0,8, e barra precisaria comecar no zero,
    o que esconde a diferenca entre os modelos.
    """
    name = METRIC_NAMES[metric]
    data = mean_by_model(comp, metric)
    present = [m for m in order if m in set(data["model"])]
    y = alt.Y("model:N", sort=present, title=None, axis=alt.Axis(labelLimit=220))
    base = alt.Chart(data)
    line = base.mark_rule(color=MUTED, strokeWidth=2).encode(
        y=y, x="min(valor):Q", x2="max(valor):Q")
    dots = base.mark_circle(size=140, opacity=1).encode(
        y=y,
        x=alt.X("valor:Q", title=f"{name} média dos projetos", scale=alt.Scale(zero=False)),
        color=alt.Color("setting:N", title="Cenário",
                        scale=alt.Scale(domain=SETTINGS,
                                        range=[SETTING_COLORS[s] for s in SETTINGS])),
        tooltip=[alt.Tooltip("model:N", title="Modelo"),
                 alt.Tooltip("setting:N", title="Cenário"),
                 alt.Tooltip("valor:Q", title=name, format=".4f")],
    )
    return alt.layer(line, dots).properties(height=alt.Step(30))


def project_chart(comp, metric, order):
    """Um ponto por modelo em cada projeto, com uma coluna por cenario."""
    name = METRIC_NAMES[metric]
    data = comp.dropna(subset=[metric])
    settings = [s for s in SETTINGS if s in set(data["setting"])]
    present = [m for m in order if m in set(data["model"])]
    # xOffset espalha os modelos dentro de cada projeto, senao os pontos ficam um em cima do outro
    # mesma legenda pra cor e forma (mesmo campo, titulo e ordem), entao o vega junta as duas
    legend = alt.Legend(orient="bottom", columns=5, labelLimit=200)
    return alt.Chart(data).mark_point(size=60, opacity=0.9, filled=True).encode(
        x=alt.X("project:N", sort=PROJECTS, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N", sort=present),
        y=alt.Y(f"{metric}:Q", title=name, scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title="Modelo", scale=_model_scale(order), sort=order,
                        legend=legend),
        shape=alt.Shape("model:N", title="Modelo", sort=order, legend=legend,
                        scale=alt.Scale(domain=order,
                                        range=[REFERENCE_MARKERS.get(m, "circle")
                                               for m in order])),
        column=alt.Column("setting:N", sort=settings, title=None,
                          header=alt.Header(labelExpr=f"{SETTING_NAMES}[datum.value]")),
        tooltip=[alt.Tooltip("model:N", title="Modelo"),
                 alt.Tooltip("project:N", title="Projeto"),
                 alt.Tooltip("setting:N", title="Cenário"),
                 alt.Tooltip(f"{metric}:Q", title=name, format=".4f")],
    ).properties(width=880 // max(len(settings), 2), height=320)
