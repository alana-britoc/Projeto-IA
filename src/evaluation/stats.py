# testes estatisticos da comparacao final (#7)
import numpy as np


def cliffs_delta(a, b):
    # P(a > b) - P(a < b), vai de -1 a 1. positivo = a tende a ser maior
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("cliffs_delta precisa de duas amostras nao vazias")
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff)).sum() / diff.size)
