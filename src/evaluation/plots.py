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
