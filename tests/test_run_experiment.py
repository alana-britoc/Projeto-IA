import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation import run_experiment


def fake_split(project, setting, features):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, len(features)))
    y = (X[:, 0] + rng.normal(size=240) > 0).astype(int)
    return {"X_train": X[:200], "y_train": y[:200],
            "X_test": X[200:], "y_test": y[200:], "ids_test": np.arange(40),
            "effort_test": rng.integers(0, 50, size=40).astype(float)}


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_experiment, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(run_experiment, "load_split", fake_split)
    return tmp_path


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_experiment", *args])
    run_experiment.main()


def test_run_one_com_busca_registra_n_iter(output):
    row = run_experiment.run_one("lr_jit", "qt", "WP", 0, n_iter=2)
    assert row["n_iter"] == 2
    assert row["cv_auc"] is not None
    assert "logisticregression__C" in row["best_params"]
    assert 0 <= row["auc"] <= 1


def test_run_one_registra_as_metricas_de_esforco(output):
    row = run_experiment.run_one("lapredict", "qt", "WP", 0)
    assert 0 <= row["popt"] <= 1
    assert 0 <= row["recall_20"] <= 1


def test_busca_nao_ve_os_20_porcento_finais_do_treino(output, monkeypatch):
    original = run_experiment.tune
    vistos = []

    def tune_espiao(model, params, X, y, seed, n_iter):
        vistos.append(X.copy())
        return original(model, params, X, y, seed, n_iter)

    monkeypatch.setattr(run_experiment, "tune", tune_espiao)
    run_experiment.run_one("lr_jit", "qt", "WP", 0, n_iter=2)
    # fake_split tem 200 commits de treino
    X_train = fake_split("qt", "WP", run_experiment.get_model("lr_jit")[0])["X_train"]
    assert np.array_equal(vistos[0], X_train[:160])


def test_run_one_sem_busca_registra_zero(output):
    row = run_experiment.run_one("lapredict", "qt", "WP", 0, n_iter=20)
    assert row["n_iter"] == 0
    assert row["cv_auc"] is None
    assert row["best_params"] == "{}"


def test_modelo_final_usa_os_parametros_da_busca(output):
    model = run_experiment.get_model("lr_jit", 0)[1]
    d = fake_split("qt", "WP", ["a", "b"])
    best, params, _ = run_experiment.tune(model, run_experiment.get_model("lr_jit", 0)[2],
                                          d["X_train"], d["y_train"], 0, 2)
    assert best.get_params()["logisticregression__C"] == params["logisticregression__C"]


def test_save_junta_e_substitui_a_mesma_execucao(output):
    base = {"model": "m", "project": "qt", "setting": "WP"}
    run_experiment.save({**base, "seed": 0, "auc": 0.1})
    run_experiment.save({**base, "seed": 1, "auc": 0.2})
    run_experiment.save({**base, "seed": 0, "auc": 0.3})
    df = pd.read_csv(output / "experimentos_m.csv").sort_values("seed")
    assert df["seed"].tolist() == [0, 1]
    assert df["auc"].tolist() == [0.3, 0.2]


def test_grade_interrompida_mantem_o_que_ja_rodou(output, monkeypatch):
    original = run_experiment.run_one

    def falha_na_seed_1(model_name, project, setting, seed, n_iter):
        if seed == 1:
            raise RuntimeError("interrompido")
        return original(model_name, project, setting, seed, n_iter)

    monkeypatch.setattr(run_experiment, "run_one", falha_na_seed_1)
    with pytest.raises(RuntimeError):
        run_main(monkeypatch, "--models", "lapredict", "--projects", "qt",
                 "--settings", "WP", "--seeds", "0", "1")
    assert pd.read_csv(output / "experimentos_lapredict.csv")["seed"].tolist() == [0]


@pytest.mark.parametrize("args", [["--models", "nao_existe"], ["--projects", "nao_existe"],
                                  ["--settings", "wp"]])
def test_argumento_invalido_falha_antes_de_rodar(output, monkeypatch, args):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *args)
    assert not list(output.glob("*.csv"))


def test_aceita_o_cenario_cp_crono(output, monkeypatch):
    run_main(monkeypatch, "--models", "lapredict", "--projects", "qt", "--settings", "CP_crono")
    assert pd.read_csv(output / "experimentos_lapredict.csv")["setting"].tolist() == ["CP_crono"]


def test_cada_modelo_salva_no_proprio_arquivo(output):
    base = {"project": "qt", "setting": "WP", "seed": 0}
    run_experiment.save({**base, "model": "a", "auc": 0.1})
    run_experiment.save({**base, "model": "b", "auc": 0.2})
    assert pd.read_csv(output / "experimentos_a.csv")["auc"].tolist() == [0.1]
    assert pd.read_csv(output / "experimentos_b.csv")["auc"].tolist() == [0.2]
