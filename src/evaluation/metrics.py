# auc e pr-auc nao dependem do limiar, o resto depende
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_threshold(y_true, y_prob):
    # pega o limiar com maior f1 (rodar na validacao, nunca no teste)
    # o mcc usa esse mesmo limiar, nao tem um proprio (decidido na #10)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # o ultimo ponto da curva nao tem limiar
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    scores = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return float(thresholds[int(np.argmax(scores))])


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
