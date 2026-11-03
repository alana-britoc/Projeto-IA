# graficos da comparacao final (#7), salvos em results/figures/
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import PROJECTS, ROOT  # noqa: E402

FIGURES_DIR = ROOT / "results" / "figures"

# paleta categorica em ordem fixa: a cor segue o modelo, nao a posicao no ranking
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def model_colors(models):
    if len(models) > len(PALETTE):
        raise ValueError(f"no maximo {len(PALETTE)} modelos por grafico")
    return dict(zip(models, PALETTE))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


METRIC_NAMES = {"auc": "AUC", "pr_auc": "PR-AUC", "f1": "F1", "mcc": "MCC"}
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
    paper = sorted(set(comp.loc[comp["source"] == "artigo", "model"]))
    if paper:
        fig.text(0.01, 0.01, f"{', '.join(paper)}: AUC publicada no artigo (sem as outras métricas)",
                 color=MUTED, fontsize=8)
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
                   label=model if setting == "WP" else None)
        ax.set_xticks(range(len(PROJECTS)), PROJECTS)
        ax.set_ylim(0.5, None)
        ax.set_title(setting, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_ylabel("AUC", color=INK_2, fontsize=9)
    fig.legend(loc="upper right", frameon=False, ncol=n, fontsize=9, labelcolor=INK_2)
    fig.suptitle("AUC por projeto (eixo começa em 0,5, o acaso)", x=0.01, ha="left", color=INK,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
        ax.set_xlim(-1.15, 1.15)
        ax.set_title(METRIC_NAMES[metric], loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.suptitle(f"Cliff's delta contra o {baseline} (positivo = melhor que o {baseline}; "
                 "* = p < 0,05 no Wilcoxon com Holm)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
