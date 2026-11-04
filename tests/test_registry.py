import numpy as np
import pytest

from src.data.loader import FEATURES
from src.models.registry import MODELS, get_model


def test_modelo_desconhecido_da_erro():
    with pytest.raises(KeyError):
        get_model("nao_existe")


def test_lapredict_usa_so_la_e_nao_tem_busca():
    features, _, params = get_model("lapredict")
    assert features == ["la"]
    assert params is None


def test_lr_jit_usa_as_14_features():
    features, _, params = get_model("lr_jit")
    assert features == FEATURES
    assert params


@pytest.mark.parametrize("name", ["svm", "mlp"])
def test_svm_e_mlp_usam_as_14_features_e_tem_busca(name):
    features, _, params = get_model(name)
    assert features == FEATURES
    assert params


@pytest.mark.parametrize("name", ["svm", "mlp"])
def test_svm_e_mlp_padronizam_as_features(name):
    _, model, _ = get_model(name)
    assert "standardscaler" in model.named_steps


@pytest.mark.parametrize("name", ["svm", "mlp"])
def test_svm_e_mlp_usam_a_seed(name):
    _, model, _ = get_model(name, seed=3)
    seeds = {v for k, v in model.get_params().items() if k.endswith("random_state")}
    assert seeds == {3}


@pytest.mark.parametrize("name", list(MODELS))
def test_params_existem_no_modelo(name):
    # se errar o nome do parametro a busca so quebra la no meio do experimento
    _, model, params = get_model(name)
    for p in params or {}:
        assert p in model.get_params()


@pytest.mark.parametrize("name", list(MODELS))
def test_treina_e_devolve_probabilidade(name):
    features, model, _ = get_model(name, seed=1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(features)))
    y = np.array([0, 1] * 20)
    prob = model.fit(X, y).predict_proba(X)[:, 1]
    assert prob.shape == (40,)
    assert ((prob >= 0) & (prob <= 1)).all()
