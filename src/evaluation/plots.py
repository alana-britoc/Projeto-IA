# graficos da comparacao final (#7), salvos em results/figures/
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import PROJECTS, ROOT, SETTINGS  # noqa: E402
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


# as 3 primeiras cores da paleta sao as que se distinguem entre si em qualquer combinacao
SETTING_COLORS = {"WP": PALETTE[0], "CP": PALETTE[1], "CP_crono": PALETTE[2]}


def _settings(comp):
    return [s for s in SETTINGS if s in set(comp["setting"])]


def metric_grid(n):
    """Linhas e colunas pra ter um painel por metrica (2 linhas)."""
    return 2, -(-n // 2)


def plot_metrics(comp, order, baseline="lapredict", path=None):
    """Um painel por metrica: cada ponto e um projeto x cenario, o traco preto e a media."""
    path = path or FIGURES_DIR / "comparacao_metricas.png"
    rows, cols = metric_grid(len(METRIC_NAMES))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 7.5), facecolor=SURFACE)
    for ax in axes.flat[len(METRIC_NAMES):]:
        ax.set_visible(False)
    settings = _settings(comp)
    # um ponto por cenario em cada linha, lado a lado (+-0,12 com 2 cenarios, como antes)
    step = 0.24 if len(settings) == 2 else 0.2
    offsets = {s: step * (i - (len(settings) - 1) / 2) for i, s in enumerate(settings)}
    for ax, metric in zip(axes.flat, METRIC_NAMES):
        _style(ax)
        ax.set_title(METRIC_NAMES[metric], loc="left", color=INK, fontsize=11, fontweight="bold")
        models = [m for m in order if comp.loc[comp["model"] == m, metric].notna().any()]
        if not models:
            # ex.: resultados rodados antes da metrica existir
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "sem resultados", transform=ax.transAxes, ha="center",
                    color=MUTED, fontsize=10)
            continue
        for y, model in enumerate(models):
            sub = comp[comp["model"] == model]
            for setting, offset in offsets.items():
                vals = sub.loc[sub["setting"] == setting, metric].dropna()
                ax.scatter(vals, [y + offset] * len(vals), s=36, color=SETTING_COLORS[setting],
                           edgecolors=SURFACE, linewidths=1.5, zorder=3,
                           label=setting if y == 0 else None)
            ax.plot([sub[metric].mean()] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2, zorder=4)
        base = comp.loc[comp["model"] == baseline, metric].mean()
        ax.axvline(base, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.set_yticks(range(len(models)), models)
        ax.set_ylim(len(models) - 0.5, -0.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=len(settings), fontsize=9,
               labelcolor=INK_2)
    fig.suptitle(f"Modelos por métrica (6 projetos × {'/'.join(settings)}; traço = média; "
                 f"tracejado = média do {baseline})", x=0.01, ha="left", color=INK, fontsize=12)
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
    settings = _settings(comp)
    fig, axes = plt.subplots(len(settings), 1, figsize=(11, 3.5 * len(settings)),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, setting in zip(axes[:, 0], settings):
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
                   label=model if setting == settings[0] else None)
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


def plot_effect_sizes(tests, baseline="lapredict", path=None, scenario=None):
    """Cliff's delta contra o baseline, um painel por metrica. Faixa cinza = efeito negligivel.

    `scenario` entra no titulo (ex.: "WP + CP cronológico" na comparacao realista da #24).
    """
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
    where = f", {scenario}" if scenario else ""
    fig.suptitle(f"Cliff's delta contra o {baseline}{where} (positivo = melhor que o {baseline}; "
                 "* = p < 0,05 no Wilcoxon com Holm)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
