import pandas as pd
import pytest

from src.data.loader import PROJECTS
from src.evaluation.compare import all_tests, build_comparison, model_order
from src.evaluation.plots import (PALETTE, model_colors, plot_auc_by_project, plot_effect_sizes,
                                  plot_metrics)


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
