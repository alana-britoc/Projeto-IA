import numpy as np
import pandas as pd
import pytest
import sklearn
from streamlit.testing.v1 import AppTest

from src.data.loader import FEATURES, ROOT
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
    assert bundle["model"].get_params()["max_depth"] == train_simulator.SIZE_LIMITS["max_depth"]


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


def test_train_e_save_com_dados_sinteticos(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(FEATURES)))
    y = (X[:, 4] + rng.normal(size=300) > 0).astype(int)
    b = train_simulator.train(X, y, {"n_estimators": 20, "max_depth": 4})
    assert b["sklearn_version"] == sklearn.__version__
    assert b["n_train"] == 300
    assert b["medians"]["la"] == pytest.approx(np.median(X[:, 4]))
    assert 0.5 < b["oob_auc"] <= 1
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
