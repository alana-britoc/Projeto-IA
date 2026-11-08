# graficos da comparacao final (#7), salvos em results/figures/
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import PROJECTS, ROOT  # noqa: E402
from src.evaluation.metrics import METRIC_NAMES  # noqa: E402

FIGURES_DIR = ROOT / "results" / "figures"

# paleta categorica validada (8 cores, checada para daltonismo). Nao da pra gerar uma 9a cor
# que continue distinguivel, entao modelo novo precisa de cor em MODEL_COLORS
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# cor fixa por modelo (#26): a cor segue o modelo, e nao a posicao no ranking, entao um modelo
# que sobe de posicao nao repinta os outros. Os 8 modelos do grupo usam as 8 cores da paleta
MODEL_COLORS = {"lapredict": PALETTE[0], "random_forest": PALETTE[1],
                "hist_gradient_boosting": PALETTE[2], "mlp": PALETTE[3], "lightgbm": PALETTE[4],
                "xgboost": PALETTE[5], "svm": PALETTE[6], "lr_jit": PALETTE[7]}
# os modelos de deep learning do artigo sao referencia: ficam em cinza, e como cinza nao
# separa bem tres series, cada um tem tambem uma forma (painel) ou hachura (barras)
REFERENCE_COLORS = {"deepjit": "#52514e", "cc2vec": "#807f78", "dbn_jit": "#aeada4"}
REFERENCE_MARKERS = {"deepjit": "triangle-up", "cc2vec": "square", "dbn_jit": "diamond"}
REFERENCE_HATCH = "///"


def model_colors(models):
    """Cor de cada modelo. Modelo desconhecido (ex.: nos testes) pega uma cor livre da paleta."""
    fixed = {**MODEL_COLORS, **REFERENCE_COLORS}
    colors = {m: fixed[m] for m in models if m in fixed}
    free = [c for c in PALETTE if c not in colors.values()]
    unknown = [m for m in models if m not in fixed]
    if len(unknown) > len(free):
        raise ValueError(f"sem cor livre para {unknown}: adicione o modelo em MODEL_COLORS")
    colors.update(zip(unknown, free))
    return {m: colors[m] for m in models}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


SETTING_COLORS = {"WP": PALETTE[0], "CP": PALETTE[1]}


def plot_metrics(comp, order, baseline="lapredict", path=None):
    """Um painel por metrica: cada ponto e um projeto x cenario, o traco preto e a media."""
    path = path or FIGURES_DIR / "comparacao_metricas.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for ax, metric in zip(axes.flat, METRIC_NAMES):
        _style(ax)
        models = [m for m in order if comp.loc[comp["model"] == m, metric].notna().any()]
        for y, model in enumerate(models):
            sub = comp[comp["model"] == model]
            for setting, offset in (("WP", -0.12), ("CP", 0.12)):
                vals = sub.loc[sub["setting"] == setting, metric].dropna()
                ax.scatter(vals, [y + offset] * len(vals), s=36, color=SETTING_COLORS[setting],
                           edgecolors=SURFACE, linewidths=1.5, zorder=3,
                           label=setting if y == 0 else None)
            ax.plot([sub[metric].mean()] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
        base = comp.loc[comp["model"] == baseline, metric].mean()
        ax.axvline(base, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_title(METRIC_NAMES[metric], loc="left", color=INK, fontsize=11, fontweight="bold")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Modelos por métrica (6 projetos × WP/CP; traço = média; tracejado = média do "
                 f"{baseline})", x=0.01, ha="left", color=INK, fontsize=12)
    notes = []
    deep = sorted(set(comp.loc[comp["source"] == "autores", "model"]))
    if deep:
        notes.append(f"{', '.join(deep)}: só AUC e PR-AUC, das predições dos autores")
    paper = sorted(set(comp.loc[comp["source"] == "artigo", "model"]))
    if paper:
        notes.append(f"{', '.join(paper)}: só a AUC publicada no artigo")
    if notes:
        fig.text(0.01, 0.01, "; ".join(notes), color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_auc_by_project(comp, order, path=None):
    """Barras agrupadas de AUC: um grupo por projeto, um painel por cenario."""
    path = path or FIGURES_DIR / "auc_por_projeto.png"
    colors = model_colors(order)
    n = len(order)
    width = 0.8 / n
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), facecolor=SURFACE, sharey=True)
    for ax, setting in zip(axes, ("WP", "CP")):
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        sub = comp[comp["setting"] == setting].set_index(["model", "project"])["auc"]
        for i, model in enumerate(order):
            vals = [sub.get((model, p), float("nan")) for p in PROJECTS]
            xs = [x - 0.4 + width * (i + 0.5) for x in range(len(PROJECTS))]
            # gap de superficie entre barras vizinhas
            ax.bar(xs, vals, width=width, color=colors[model], edgecolor=SURFACE, linewidth=1,
                   hatch=REFERENCE_HATCH if model in REFERENCE_COLORS else None,
                   label=model if setting == "WP" else None)
        ax.set_xticks(range(len(PROJECTS)), PROJECTS)
        ax.set_ylim(0.5, None)
        ax.set_title(setting, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_ylabel("AUC", color=INK_2, fontsize=9)
    # duas linhas pra nao cobrir o titulo
    fig.legend(loc="upper left", bbox_to_anchor=(0.01, 0.955), frameon=False, ncol=min(n, 5),
               fontsize=9, labelcolor=INK_2)
    fig.suptitle("AUC por projeto (eixo começa em 0,5, o acaso)", x=0.01, ha="left", color=INK,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_effect_sizes(tests, baseline="lapredict", path=None):
    """Cliff's delta contra o baseline, um painel por metrica. Faixa cinza = efeito negligivel."""
    path = path or FIGURES_DIR / "cliffs_delta_lapredict.png"
    metrics = [m for m in METRIC_NAMES if m in set(tests["metric"])]
    rows = int(tests["metric"].value_counts().max())
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics) + 1, 3.8),
                             facecolor=SURFACE, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        _style(ax)
        sub = tests[tests["metric"] == metric].reset_index(drop=True)
        ax.axvspan(-0.147, 0.147, color=GRID, alpha=0.6, zorder=0)
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        colors = [PALETTE[0] if d > 0 else PALETTE[1] for d in sub["delta"]]
        ax.barh(range(len(sub)), sub["delta"], height=0.55, color=colors, zorder=2)
        for y, r in sub.iterrows():
            mark = "*" if r["p_holm"] < 0.05 else ""
            ha, x = ("left", r["delta"] + 0.04) if r["delta"] >= 0 else ("right", r["delta"] - 0.04)
            ax.text(x, y, f"{r['delta']:+.2f}{mark}", va="center", ha=ha, fontsize=8, color=INK_2)
        ax.set_yticks(range(len(sub)), sub["model"])
        # mesma altura de linha em todos os paineis, mesmo com um modelo so
        ax.set_ylim(rows - 0.5, -0.5)
        # folga pro rotulo nao encostar no nome
        ax.set_xlim(-1.4, 1.4)
        ax.set_title(METRIC_NAMES[metric], loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.suptitle(f"Cliff's delta contra o {baseline} (positivo = melhor que o {baseline}; "
                 "* = p < 0,05 no Wilcoxon com Holm)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


MODEL_NAMES = {"random_forest": "Random Forest", "hist_gradient_boosting": "HistGradientBoosting"}


def plot_feature_importance(imp, path=None):
    path = path or FIGURES_DIR / "importancia_features.png"
    models = list(dict.fromkeys(imp["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 6), facecolor=SURFACE,
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        _style(ax)
        sub = imp[imp["model"] == model]
        order = sub.groupby("feature")["importance_mean"].mean().sort_values(ascending=False)
        for y, feature in enumerate(order.index):
            for setting, offset in (("WP", -0.12), ("CP", 0.12)):
                vals = sub.loc[(sub["feature"] == feature) & (sub["setting"] == setting),
                               "importance_mean"]
                ax.scatter(vals, [y + offset] * len(vals), s=30, color=SETTING_COLORS[setting],
                           edgecolors=SURFACE, linewidths=1.2, zorder=3,
                           label=setting if y == 0 else None)
            ax.plot([order[feature]] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_yticks(range(len(order)), order.index)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel("queda da AUC ao embaralhar a feature", color=INK_2, fontsize=9)
        ax.set_title(MODEL_NAMES.get(model, model), loc="left", color=INK, fontsize=11,
                     fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, fontsize=9,
               labelcolor=INK_2)
    fig.suptitle("Importância por permutação no teste (traço = média dos 6 projetos × WP/CP)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
