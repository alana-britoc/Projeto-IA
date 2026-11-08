import numpy as np
import pytest

from src.evaluation.metrics import (METRIC_NAMES, best_threshold, compute_metrics, popt,
                                    recall_at_effort)


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


def test_best_threshold_nao_arredonda_as_probabilidades():
    y = np.array([1, 0])
    prob = np.array([0.12346, 0.1])
    t = best_threshold(y, prob)
    assert t == pytest.approx(0.12346)
    assert compute_metrics(y, prob, t)["f1"] == pytest.approx(1.0)


def test_best_threshold_com_probabilidades_muito_baixas():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.00001, 0.00002, 0.00003, 0.00004])
    t = best_threshold(y, prob)
    assert compute_metrics(y, prob, t)["f1"] == pytest.approx(1.0)


# metricas de esforco (#25). Exemplo calculado a mao: 3 commits, esforco 1, 1 e 2 linhas
# (total 4), o 1o e o 3o com defeito. Curvas (esforco acumulado, defeitos acumulados):
# otima  c0, c2, c1: (0,0) (.25,.5) (.75,1) (1,1) -> area 0,6875
# pior   c1, c2, c0: (0,0) (.25,0) (.75,.5) (1,1) -> area 0,3125
# modelo c0, c1, c2: (0,0) (.25,.5) (.5,.5) (1,1) -> area 0,5625
# Popt = 1 - (0,6875 - 0,5625) / (0,6875 - 0,3125) = 2/3
Y3 = [1, 0, 1]
EFFORT3 = [1, 1, 2]


def test_popt_no_exemplo_calculado_a_mao():
    # densidades 0,9 / 0,5 / 0,1 -> ordem c0, c1, c2
    assert popt(Y3, [0.9, 0.5, 0.2], EFFORT3) == pytest.approx(2 / 3)


def test_popt_do_modelo_otimo_e_1_e_do_pior_e_0():
    assert popt(Y3, [0.9, 0.1, 0.8], EFFORT3) == pytest.approx(1.0)
    assert popt(Y3, [0.1, 0.9, 0.3], EFFORT3) == pytest.approx(0.0)


def test_ordem_usa_densidade_e_nao_so_a_probabilidade():
    # o commit grande tem probabilidade maior, mas o pequeno tem mais risco por linha
    y = [0, 1]
    effort = [100, 1]
    assert popt(y, [0.6, 0.3], effort) == pytest.approx(1.0)


def test_recall_com_20_por_cento_do_esforco():
    # 10 commits de 1 linha: 20% do esforco = os 2 primeiros da ordem
    y = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    prob = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    assert recall_at_effort(y, prob, [1] * 10) == pytest.approx(0.5)


def test_recall_nao_conta_commit_revisado_pela_metade():
    # o 1o commit ja passa de 20% das linhas, entao nenhum e revisado inteiro
    assert recall_at_effort([1, 0], [0.9, 0.1], [5, 5]) == 0


def test_commit_sem_linhas_alteradas_conta_como_uma_linha():
    # la + ld = 0 acontece (ex.: so renomeia arquivo) e nao pode dividir por zero
    assert 0 <= popt([1, 0, 1], [0.5, 0.5, 0.5], [0, 3, 1]) <= 1


def test_compute_metrics_inclui_esforco_quando_recebe_as_linhas():
    m = compute_metrics(Y3, [0.9, 0.5, 0.2], effort=EFFORT3)
    assert m["popt"] == pytest.approx(2 / 3)
    assert "recall_20" in m
    assert "popt" not in compute_metrics(Y3, [0.9, 0.5, 0.2])
    assert {"popt", "recall_20"} <= set(METRIC_NAMES)
