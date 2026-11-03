# graficos da comparacao final (#7), salvos em results/figures/
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import ROOT  # noqa: E402

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
