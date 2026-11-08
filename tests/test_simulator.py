import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest

from src.data.loader import FEATURES, ROOT
from src.evaluation.plots import plot_calibration
from src.models import simulator, train_simulator


@pytest.fixture(scope="module")
def bundle():
    return simulator.load()


def test_modelo_salvo_tem_menos_de_10_mb():
    assert simulator.MODEL_PATH.stat().st_size < train_simulator.MAX_MB * 1e6


def test_modelo_salvo_registra_versao_do_sklearn_e_as_features(bundle):
    assert bundle["sklearn_version"]
    assert bundle["features"] == FEATURES
    assert set(bundle["medians"]) == set(FEATURES)


def test_modelo_salvo_respeita_o_limite_de_tamanho(bundle):
    assert bundle["params"]["n_estimators"] == train_simulator.SIZE_LIMITS["n_estimators"]
    assert bundle["forest"].get_params()["max_depth"] == train_simulator.SIZE_LIMITS["max_depth"]


def test_calibracao_melhora_o_brier(bundle):
    brier = bundle["brier"]
    assert bundle["method"] in train_simulator.METHODS
    assert brier[bundle["method"]] < brier[train_simulator.UNCALIBRATED]


def test_probabilidade_media_perto_da_taxa_real(bundle):
    # sem calibracao a media dava uns 43%, quase o dobro da taxa real
    assert np.mean(bundle["quantiles"]) == pytest.approx(bundle["defect_rate"], abs=0.02)


def test_commit_mediano_fica_abaixo_de_30_porcento(bundle):
    assert 0.1 < simulator.predict(bundle, bundle["medians"]) < 0.3


@pytest.mark.parametrize("la, nf", [(10, 2), (0, 1), (5000, 80)])
def test_probabilidade_fica_entre_0_e_1(bundle, la, nf):
    prob = simulator.predict(bundle, {**bundle["medians"], "la": la, "nf": nf})
    assert 0 <= prob <= 1


def test_commit_grande_e_mais_arriscado_que_um_pequeno(bundle):
    m = bundle["medians"]
    grande = simulator.predict(bundle, {**m, "la": 2000, "ld": 500, "nf": 30, "entrophy": 4.0})
    pequeno = simulator.predict(bundle, {**m, "la": 1, "ld": 1, "nf": 1, "entrophy": 0.0})
    assert grande > pequeno


def test_percentil_vai_de_0_a_100(bundle):
    assert simulator.percentile(bundle, 0.0) == 0
    assert simulator.percentile(bundle, 1.01) == 100
    assert 0 < simulator.percentile(bundle, 0.5) < 100


def test_todas_as_features_tem_campo_e_grupo():
    assert set(simulator.FIELDS) == set(FEATURES)
    grupos = [f for features in simulator.GROUPS.values() for f in features]
    assert sorted(grupos) == sorted(FEATURES)


def test_chosen_params_pega_a_combinacao_mais_escolhida(tmp_path):
    path = tmp_path / "rf.csv"
    pd.DataFrame({"best_params": ['{"max_depth": 5}', '{"max_depth": 10}',
                                  '{"max_depth": 10}']}).to_csv(path, index=False)
    params, vezes, total = train_simulator.chosen_params(path)
    assert params == {"max_depth": 10}
    assert (vezes, total) == (2, 3)


def test_split_project_separa_em_ordem_80_10_10():
    X = np.arange(100).reshape(-1, 1)
    parts = train_simulator.split_project(X, np.arange(100))
    assert [len(parts[k][1]) for k in ("train", "calib", "eval")] == [80, 10, 10]
    assert parts["train"][1].max() < parts["calib"][1].min() < parts["eval"][1].min()


def _dados_sinteticos(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    y = (X[:, 4] + rng.normal(size=n) > 0).astype(int)
    return X, y


def test_calibrate_fica_com_o_menor_brier():
    X, y = _dados_sinteticos(900)
    forest = RandomForestClassifier(n_estimators=20, class_weight="balanced",
                                    random_state=0).fit(X[:600], y[:600])
    model, method, probs, brier = train_simulator.calibrate(forest, X[600:750], y[600:750],
                                                           X[750:], y[750:])
    assert method == min(train_simulator.METHODS, key=brier.get)
    assert set(probs) == set(train_simulator.METHODS) | {train_simulator.UNCALIBRATED}
    assert 0 <= model.predict_proba(X[:5])[:, 1].min()


def test_grafico_de_calibracao_e_salvo(tmp_path):
    X, y = _dados_sinteticos(300)
    probs = {"sem calibração": 1 / (1 + np.exp(-X[:, 4])), "sigmoid": np.full(300, y.mean())}
    path = plot_calibration(y, probs, path=tmp_path / "calib.png")
    assert path.exists() and path.stat().st_size > 0


def test_train_e_save_com_dados_sinteticos(tmp_path):
    X, y = _dados_sinteticos(500)
    data = train_simulator.split_project(X, y)
    b, probs = train_simulator.train(data, {"n_estimators": 20, "max_depth": 4})
    assert b["sklearn_version"] == sklearn.__version__
    assert (b["n_train"], b["n_calib"], b["n_eval"]) == (400, 50, 50)
    assert b["medians"]["la"] == pytest.approx(np.median(X[:, 4]))
    assert 0.5 < b["auc"] <= 1
    assert len(probs[b["method"]]) == 50
    size = train_simulator.save(b, tmp_path / "m.joblib")
    assert 0 < size < 1
    prob = simulator.predict(simulator.load(tmp_path / "m.joblib"), dict(zip(FEATURES, X[0])))
    assert 0 <= prob <= 1


def test_pagina_abre_com_as_medianas_e_responde_ao_campo(bundle):
    at = AppTest.from_file(str(ROOT / "src" / "simulador.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.number_input) == 13 and len(at.checkbox) == 1
    assert at.number_input(key="sim_la").value == int(bundle["medians"]["la"])
    antes = at.metric[0].value
    at.number_input(key="sim_la").set_value(3000).run()
    assert not at.exception
    assert at.metric[0].value != antes
    assert any("Demonstração" in w.value for w in at.warning)


def test_app_tem_as_duas_paginas():
    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.switch_page("simulador.py").run()
    assert not at.exception
    assert any("Simulador" in t.value for t in at.title)
