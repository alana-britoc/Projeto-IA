import numpy as np
import pytest

from src.evaluation.metrics import best_threshold, compute_metrics


def test_classificador_perfeito():
    y = [0, 0, 1, 1]
    m = compute_metrics(y, [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["mcc"] == 1.0


def test_limiar_muda_f1_mas_nao_auc():
    y = [0, 0, 1, 1]
    prob = [0.1, 0.3, 0.35, 0.9]
    alto = compute_metrics(y, prob, threshold=0.5)
    baixo = compute_metrics(y, prob, threshold=0.3)
    assert alto["auc"] == baixo["auc"]
    assert alto["recall"] == 0.5
    assert baixo["recall"] == 1.0


def test_sem_predicoes_positivas_nao_quebra():
    m = compute_metrics([0, 1, 0, 1], [0.1, 0.2, 0.1, 0.3], threshold=0.9)
    assert m["f1"] == 0
    assert m["mcc"] == 0


def test_best_threshold_separa_as_classes():
    y = np.array([0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    t = best_threshold(y, prob)
    assert 0.3 < t <= 0.6
    assert compute_metrics(y, prob, t)["f1"] == pytest.approx(1.0)
