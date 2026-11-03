import pytest

from src.evaluation.stats import cliffs_delta


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
