import pandas as pd
import pytest

from src.data.loader import PROJECTS
from src.evaluation.compare import PAPER_MODELS, all_tests, build_comparison, model_order
from src.evaluation.metrics import METRIC_NAMES
from src.evaluation.plots import (MODEL_COLORS, PALETTE, REFERENCE_COLORS, REFERENCE_MARKERS,
                                  metric_grid, model_colors, plot_auc_by_project,
                                  plot_effect_sizes, plot_metrics)


def test_grade_tem_um_painel_para_cada_metrica():
    # com 6 metricas (#25) a grade 2x2 antiga descartava Popt e Recall@20% sem avisar
    rows, cols = metric_grid(len(METRIC_NAMES))
    assert rows * cols >= len(METRIC_NAMES)
    assert metric_grid(4) == (2, 2)
    assert metric_grid(6) == (2, 3)
from src.models.registry import MODELS


@pytest.fixture
def comp():
    rows = []
    for i, p in enumerate(PROJECTS):
        for s in ("WP", "CP"):
            for model, auc in (("lapredict", 0.70 + 0.01 * i), ("outro", 0.66 + 0.012 * i)):
                rows.append({"model": model, "project": p, "setting": s, "seed": 0,
                             "auc": auc, "pr_auc": auc / 2, "f1": auc / 3, "mcc": auc / 4})
    return build_comparison(pd.DataFrame(rows))


def test_model_colors_segue_a_ordem_fixa():
    assert model_colors(["a", "b"]) == {"a": PALETTE[0], "b": PALETTE[1]}


def test_cor_segue_o_modelo_e_nao_a_posicao():
    # se um modelo sobe no ranking, os outros nao podem trocar de cor
    assert model_colors(["svm", "lapredict"]) == model_colors(["lapredict", "svm"])
    assert model_colors(["mlp"])["mlp"] == MODEL_COLORS["mlp"]


def test_todo_modelo_do_registry_tem_cor_fixa_sem_repetir():
    assert set(MODEL_COLORS) == set(MODELS)
    assert sorted(MODEL_COLORS.values()) == sorted(PALETTE)


def test_modelos_do_artigo_ficam_em_cinza_com_forma_propria():
    assert set(REFERENCE_COLORS) == set(PAPER_MODELS.values())
    assert not set(REFERENCE_COLORS.values()) & set(PALETTE)
    assert set(REFERENCE_MARKERS) == set(REFERENCE_COLORS)
    colors = model_colors(["lapredict", "deepjit"])
    assert colors["deepjit"] == REFERENCE_COLORS["deepjit"]


def test_modelo_desconhecido_pega_uma_cor_livre():
    colors = model_colors(["lapredict", "outro"])
    assert colors["outro"] in PALETTE
    assert colors["outro"] != colors["lapredict"]


def test_model_colors_limita_a_paleta():
    with pytest.raises(ValueError):
        model_colors([str(i) for i in range(len(PALETTE) + 1)])


def test_graficos_sao_salvos(comp, tmp_path):
    order = model_order(comp)
    paths = [plot_metrics(comp, order, path=tmp_path / "m.png"),
             plot_auc_by_project(comp, order, path=tmp_path / "a.png"),
             plot_effect_sizes(all_tests(comp), path=tmp_path / "d.png")]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
