# auc e pr-auc nao dependem do limiar, o resto depende
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

# nome bonito pra tabela e grafico
METRIC_NAMES = {"auc": "AUC", "pr_auc": "PR-AUC", "f1": "F1", "mcc": "MCC", "popt": "Popt",
                "recall_20": "Recall@20%"}

# metricas de esforco (#25): quantos defeitos se acha revisando uma parte das linhas.
# Esforco de um commit = la + ld. A ordem de revisao e a densidade de risco (probabilidade /
# esforco), como no effort-aware de Kamei et al. (2013): um commit pequeno com risco medio vem
# antes de um enorme com risco um pouco maior. Commit sem linha alterada conta como 1 linha.
EFFORT_CUTOFF = 0.2


def _effort(effort):
    return np.maximum(np.asarray(effort, dtype=float), 1.0)


def _review_order(score, effort):
    # maior score primeiro; empate -> commit menor primeiro
    return np.lexsort((effort, -np.asarray(score, dtype=float)))


def _area(y_true, effort, order):
    """Area sob a curva esforco acumulado (x) x defeitos encontrados (y), as duas de 0 a 1."""
    x = np.concatenate([[0.0], np.cumsum(effort[order]) / effort.sum()])
    y = np.concatenate([[0.0], np.cumsum(y_true[order]) / y_true.sum()])
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def popt(y_true, y_prob, effort):
    """1 - (area otima - area do modelo) / (area otima - area pior). 1 = otimo, 0 = pior."""
    y_true = np.asarray(y_true, dtype=float)
    effort = _effort(effort)
    if y_true.sum() in (0, len(y_true)):
        return float("nan")
    density = np.asarray(y_prob, dtype=float) / effort
    model = _area(y_true, effort, _review_order(density, effort))
    # otimo: defeitos primeiro, menores antes; pior: sem defeito primeiro, maiores antes
    best = _area(y_true, effort, np.lexsort((effort, -y_true)))
    worst = _area(y_true, effort, np.lexsort((-effort, y_true)))
    return 1 - (best - model) / (best - worst)


def recall_at_effort(y_true, y_prob, effort, cutoff=EFFORT_CUTOFF):
    """Fracao dos commits com defeito achados revisando ate `cutoff` das linhas, na ordem do
    modelo. So conta commit revisado inteiro."""
    y_true = np.asarray(y_true, dtype=float)
    effort = _effort(effort)
    if y_true.sum() == 0:
        return float("nan")
    order = _review_order(np.asarray(y_prob, dtype=float) / effort, effort)
    inside = np.cumsum(effort[order]) <= cutoff * effort.sum()
    return float(y_true[order][inside].sum() / y_true.sum())


def best_threshold(y_true, y_prob):
    # pega o limiar com maior f1 (rodar na validacao, nunca no teste)
    # o mcc usa esse mesmo limiar, nao tem um proprio (decidido na #10)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # o ultimo ponto da curva nao tem limiar
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    scores = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return float(thresholds[int(np.argmax(scores))])


def compute_metrics(y_true, y_prob, threshold=0.5, effort=None):
    """Metricas de classificacao; com `effort` (linhas de cada commit), tambem Popt e Recall@20%."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
    if effort is not None:
        out["popt"] = popt(y_true, y_prob, effort)
        out["recall_20"] = recall_at_effort(y_true, y_prob, effort)
    return out
