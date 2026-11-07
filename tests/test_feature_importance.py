import json

import numpy as np
import pandas as pd
import pytest

from src.data.loader import FEATURES
from src.evaluation import feature_importance as fi
from src.evaluation.plots import plot_feature_importance

LA = FEATURES.index("la")
PARAMS = {"random_forest": {"n_estimators": 30, "max_depth": 5},
          "hist_gradient_boosting": {"max_iter": 50}}


def _dados(n, rng, rotulo):
    X = rng.normal(size=(n, len(FEATURES)))
    return X, rotulo(X).astype(int)


def _so_la(X):
    return X[:, LA] > 0


def _xor(X):
    # so da pra acertar olhando as duas features juntas
    return (X[:, 0] > 0) ^ (X[:, 1] > 0)


def test_best_params_le_a_linha_certa(tmp_path):
    rows = [{"project": "qt", "setting": "WP", "seed": 0, "best_params": '{"max_depth": 5}'},
            {"project": "qt", "setting": "CP", "seed": 0, "best_params": '{"max_depth": 10}'},
            {"project": "qt", "setting": "WP", "seed": 1, "best_params": '{"max_depth": 20}'}]
    pd.DataFrame(rows).to_csv(tmp_path / "experimentos_random_forest.csv", index=False)
    assert fi.best_params("random_forest", "qt", "CP", metrics_dir=tmp_path) == {"max_depth": 10}
    assert fi.best_params("random_forest", "qt", "WP", metrics_dir=tmp_path) == {"max_depth": 5}


@pytest.mark.parametrize("model_name", fi.MODELS)
def test_feature_que_define_o_rotulo_fica_no_topo(model_name):
    rng = np.random.default_rng(0)
    X, y = _dados(600, rng, _so_la)
    model = fi.fit(model_name, PARAMS[model_name], X[:400], y[:400])
    imp = fi.importance(model, X[400:], y[400:])
    assert list(imp.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(imp) == len(FEATURES)
    assert imp.loc[imp["importance_mean"].idxmax(), "feature"] == "la"


def test_interaction_test_detecta_quando_precisa_combinar_features():
    rng = np.random.default_rng(0)
    X, y = _dados(2000, rng, _xor)
    model = fi.fit("hist_gradient_boosting", PARAMS["hist_gradient_boosting"], X[:1500], y[:1500])
    r = fi.interaction_test(model, X[:1500], y[:1500], X[1500:], y[1500:])
    assert r["auc_com_interacoes"] > 0.9
    assert r["auc_sem_interacoes"] < 0.6
    assert r["diferenca"] == pytest.approx(r["auc_com_interacoes"] - r["auc_sem_interacoes"])


def test_interaction_test_empata_quando_nao_tem_interacao():
    rng = np.random.default_rng(0)
    X, y = _dados(2000, rng, _so_la)
    model = fi.fit("hist_gradient_boosting", PARAMS["hist_gradient_boosting"], X[:1500], y[:1500])
    r = fi.interaction_test(model, X[:1500], y[:1500], X[1500:], y[1500:])
    assert abs(r["diferenca"]) < 0.03


def test_run_one_junta_os_dois_modelos(tmp_path, monkeypatch):
    for model_name, params in PARAMS.items():
        pd.DataFrame([{"project": "qt", "setting": "WP", "seed": 0,
                       "best_params": json.dumps(params)}]).to_csv(
            tmp_path / f"experimentos_{model_name}.csv", index=False)
    rng = np.random.default_rng(0)
    X, y = _dados(500, rng, _so_la)

    def fake_split(project, setting, features):
        return {"X_train": X[:400], "y_train": y[:400], "X_test": X[400:], "y_test": y[400:]}

    monkeypatch.setattr(fi, "load_split", fake_split)
    imp, inter = fi.run_one("qt", "WP", metrics_dir=tmp_path)
    assert list(imp.columns) == ["model", "project", "setting", "feature", "importance_mean",
                                 "importance_std"]
    assert len(imp) == 2 * len(FEATURES)
    assert set(imp["model"]) == set(fi.MODELS)
    assert inter["project"] == "qt" and inter["setting"] == "WP"
    assert {"auc_com_interacoes", "auc_sem_interacoes", "diferenca"} <= set(inter)


def _imp():
    rows = []
    for setting in ("WP", "CP"):
        for i, feature in enumerate(FEATURES):
            rows.append({"model": "random_forest", "project": "qt", "setting": setting,
                         "feature": feature, "importance_mean": 0.1 if feature == "la" else i / 1000,
                         "importance_std": 0.0})
    return pd.DataFrame(rows)


def test_ranking_poe_a_maior_importancia_em_primeiro():
    rank = fi.ranking(_imp())
    assert rank.iloc[0]["feature"] == "la"
    assert rank.iloc[0]["posicao"] == 1
    assert sorted(rank["posicao"]) == list(range(1, len(FEATURES) + 1))


def test_grafico_e_salvo(tmp_path):
    path = plot_feature_importance(_imp(), path=tmp_path / "imp.png")
    assert path.exists() and path.stat().st_size > 0
