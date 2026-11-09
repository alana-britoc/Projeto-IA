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
# nome que aparece na tela (#44); por dentro continua o id do modelo
MODEL_NAMES = {"lapredict": "LApredict", "lr_jit": "LR-JIT", "random_forest": "Random Forest",
               "hist_gradient_boosting": "HistGradientBoosting", "lightgbm": "LightGBM",
               "xgboost": "XGBoost", "svm": "SVM", "mlp": "MLP", "deepjit": "DeepJIT",
               "cc2vec": "CC2Vec", "dbn_jit": "DBN-JIT"}
VERDICT_COLORS = {"supera": "#1baf7a", "empate": "#b4b2a9", "perde": "#eb6834"}
_LABEL = f"({MODEL_NAMES}[datum.value] || datum.value)"


def model_name(model):
    return MODEL_NAMES.get(model, model)


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
    out = out.assign(placar=out["wins"].astype(str) + "/" + out["losses"].astype(str),
                     model=out["model"].map(model_name))
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
    y = alt.Y("model:N", sort=present, title=None,
              axis=alt.Axis(labelLimit=220, labelExpr=_LABEL))
    base = alt.Chart(data).transform_calculate(nome=f"{MODEL_NAMES}[datum.model]")
    line = base.mark_rule(color=MUTED, strokeWidth=2).encode(
        y=y, x="min(valor):Q", x2="max(valor):Q")
    dots = base.mark_circle(size=140, opacity=1).encode(
        y=y,
        # tickCount menor pra os numeros do eixo nao grudarem
        x=alt.X("valor:Q", title=f"{name} média dos projetos", scale=alt.Scale(zero=False),
                axis=alt.Axis(tickCount=6, format=".2f")),
        color=alt.Color("setting:N", title="Cenário",
                        scale=alt.Scale(domain=SETTINGS,
                                        range=[SETTING_COLORS[s] for s in SETTINGS])),
        tooltip=[alt.Tooltip("nome:N", title="Modelo"),
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
    legend = alt.Legend(orient="bottom", columns=5, labelLimit=200, labelExpr=_LABEL)
    chart = alt.Chart(data).transform_calculate(nome=f"{MODEL_NAMES}[datum.model]")
    return chart.mark_point(size=60, opacity=0.9, filled=True).encode(
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
        tooltip=[alt.Tooltip("nome:N", title="Modelo"),
                 alt.Tooltip("project:N", title="Projeto"),
                 alt.Tooltip("setting:N", title="Cenário"),
                 alt.Tooltip(f"{metric}:Q", title=name, format=".4f")],
    ).properties(width=880 // max(len(settings), 2), height=320)


def gain_table(comp, tests, metric, comparison, models, baseline="lapredict"):
    """Diferenca media para o baseline nos mesmos pares do teste, com o resultado do teste."""
    settings = comparison.split("+")
    data = comp[comp["setting"].isin(settings)]
    wide = data.pivot_table(index=["project", "setting"], columns="model", values=metric)
    if baseline not in wide.columns:
        return pd.DataFrame(columns=["model", "ganho", "verdict", "p_holm", "delta"])
    diff = wide.drop(columns=baseline).sub(wide[baseline], axis=0).mean()
    out = diff.dropna().rename("ganho").reset_index()
    t = tests[(tests["metric"] == metric) & (tests["cenarios"] == comparison)]
    # inner: modelo sem teste nessa comparacao (ex.: deepjit no CP_crono) fica de fora
    out = out.merge(t[["model", "verdict", "p_holm", "delta"]], on="model", how="inner")
    out = out[out["model"].isin(models)]
    return out.sort_values("ganho", ascending=False).reset_index(drop=True)


def gain_chart(gains, metric):
    """Uma barra por modelo com o ganho medio sobre o LApredict, colorida pelo teste."""
    name = METRIC_NAMES[metric]
    verdicts = list(VERDICT_COLORS)
    return alt.Chart(gains).transform_calculate(
        nome=f"{MODEL_NAMES}[datum.model]").mark_bar(cornerRadius=3).encode(
        y=alt.Y("model:N", sort=list(gains["model"]), title=None,
                axis=alt.Axis(labelLimit=220, labelExpr=_LABEL)),
        x=alt.X("ganho:Q", title=f"Diferença média de {name} para o LApredict",
                axis=alt.Axis(format="+.3f")),
        color=alt.Color("verdict:N", title="Resultado do teste", sort=verdicts,
                        scale=alt.Scale(domain=verdicts, range=list(VERDICT_COLORS.values()))),
        tooltip=[alt.Tooltip("nome:N", title="Modelo"),
                 alt.Tooltip("ganho:Q", title="Diferença média", format="+.4f"),
                 alt.Tooltip("verdict:N", title="Resultado"),
                 alt.Tooltip("p_holm:Q", title="p (Holm)", format=".4f"),
                 alt.Tooltip("delta:Q", title="Cliff's delta", format="+.2f")],
    ).properties(height=alt.Step(32))


def heatmap_chart(comp, metric, order, setting):
    """Grade modelo x projeto com a cor no valor da metrica, num cenario."""
    name = METRIC_NAMES[metric]
    data = comp[comp["setting"] == setting].dropna(subset=[metric])
    present = [m for m in order if m in set(data["model"])]
    projects = [p for p in PROJECTS if p in set(data["project"])]
    # texto branco nas celulas escuras
    meio = (data[metric].min() + data[metric].max()) / 2
    base = alt.Chart(data).transform_calculate(nome=f"{MODEL_NAMES}[datum.model]").encode(
        x=alt.X("project:N", sort=projects, title=None,
                axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("model:N", sort=present, title=None,
                axis=alt.Axis(labelLimit=220, labelExpr=_LABEL)))
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(f"{metric}:Q", title=name, scale=alt.Scale(scheme="tealblues")),
        tooltip=[alt.Tooltip("nome:N", title="Modelo"),
                 alt.Tooltip("project:N", title="Projeto"),
                 alt.Tooltip(f"{metric}:Q", title=name, format=".4f")])
    labels = base.mark_text(fontSize=13).encode(
        text=alt.Text(f"{metric}:Q", format=".3f"),
        color=alt.condition(f"datum['{metric}'] > {meio}", alt.value("white"),
                            alt.value("black")))
    return alt.layer(cells, labels).properties(height=alt.Step(36), width="container")


def slope_chart(comp, metric, order):
    """Uma linha por modelo, do CP para o CP cronologico (media dos projetos)."""
    name = METRIC_NAMES[metric]
    data = mean_by_model(comp[comp["setting"].isin(["CP", "CP_crono"])], metric)
    # so quem tem os dois cenarios (os modelos do artigo nao tem CP_crono)
    both = data.groupby("model")["setting"].nunique()
    data = data[data["model"].isin(both[both == 2].index)]
    present = [m for m in order if m in set(data["model"])]
    colors = model_colors(present)
    color = alt.Color("model:N", sort=present, title="Modelo",
                      legend=alt.Legend(labelExpr=_LABEL, labelLimit=200, symbolType="stroke",
                                        symbolStrokeWidth=4),
                      scale=alt.Scale(domain=present, range=[colors[m] for m in present]))
    base = alt.Chart(data).transform_calculate(nome=f"{MODEL_NAMES}[datum.model]").encode(
        x=alt.X("setting:N", sort=["CP", "CP_crono"], title=None,
                axis=alt.Axis(labelAngle=0, labelExpr=f"{SETTING_NAMES}[datum.value]",
                              labelLimit=300)),
        y=alt.Y("valor:Q", title=f"{name} média dos projetos", scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".3f")),
        color=color,
        tooltip=[alt.Tooltip("nome:N", title="Modelo"),
                 alt.Tooltip("setting:N", title="Cenário"),
                 alt.Tooltip("valor:Q", title=name, format=".4f")])
    # legenda pq os nomes na ponta da linha se sobrepunham
    lines = base.mark_line(strokeWidth=3)
    dots = base.mark_circle(size=90, opacity=1)
    return alt.layer(lines, dots).properties(height=420, width="container")
