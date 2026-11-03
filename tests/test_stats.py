import numpy as np
import pytest

from src.evaluation.stats import cliffs_delta, delta_magnitude, holm


def test_cliffs_delta_dominancia_total():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0


def test_cliffs_delta_amostras_iguais_da_zero():
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_cliffs_delta_valor_conhecido():
    # pares: 3>1, 3>2, 1<2, 1=1 -> (2 - 1) / 4
    assert cliffs_delta([3, 1], [1, 2]) == pytest.approx(0.25)


def test_cliffs_delta_amostra_vazia_quebra():
    with pytest.raises(ValueError):
        cliffs_delta([], [1, 2])


@pytest.mark.parametrize("delta, esperado", [
    (0.0, "negligível"), (0.1, "negligível"), (-0.2, "pequeno"),
    (0.4, "médio"), (-0.474, "grande"), (1.0, "grande"),
])
def test_delta_magnitude(delta, esperado):
    assert delta_magnitude(delta) == esperado


def test_holm_valores_conhecidos():
    # ordenado: 0.01*3, 0.02*2, 0.04*1 -> 0.03, 0.04, 0.04 (monotono)
    adj = holm([0.04, 0.01, 0.02])
    assert adj == pytest.approx([0.04, 0.03, 0.04])


def test_holm_nao_passa_de_um():
    assert holm([0.6, 0.7]).max() == 1.0


def test_holm_ignora_nan():
    adj = holm([0.01, np.nan, 0.02])
    assert np.isnan(adj[1])
    assert adj[0] == pytest.approx(0.02)
    assert adj[2] == pytest.approx(0.02)
