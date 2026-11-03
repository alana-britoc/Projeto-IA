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


def delta_magnitude(delta):
    # limiares de Romano et al. (2006), os mais usados em engenharia de software
    d = abs(delta)
    if d < 0.147:
        return "negligível"
    if d < 0.33:
        return "pequeno"
    if d < 0.474:
        return "médio"
    return "grande"
