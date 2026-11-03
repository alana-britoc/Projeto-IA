# testes estatisticos da comparacao final (#7)
import numpy as np
from scipy.stats import wilcoxon


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


def holm(p_values):
    # correcao de Holm-Bonferroni, pq comparamos varios modelos contra o mesmo baseline
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    vals = p[ok]
    m = len(vals)
    order = np.argsort(vals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * vals[i])
        adj[i] = min(running, 1.0)
    out[ok] = adj
    return out


def paired_test(model, baseline):
    """Wilcoxon pareado (bicaudal) + Cliff's delta de model contra baseline.

    Cada par e um projeto x cenario, com a media das seeds de cada modelo.
    """
    model = np.asarray(model, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if model.shape != baseline.shape:
        raise ValueError("model e baseline precisam ter os mesmos pares")
    keep = ~(np.isnan(model) | np.isnan(baseline))
    model, baseline = model[keep], baseline[keep]
    diff = model - baseline
    out = {"n": len(diff), "median_diff": float(np.median(diff)) if len(diff) else np.nan,
           "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
           "statistic": np.nan, "p_value": np.nan, "delta": np.nan, "magnitude": ""}
    if len(diff) == 0:
        return out
    if np.allclose(diff, 0):
        # wilcoxon quebra se todas as diferencas sao zero
        out.update(statistic=0.0, p_value=1.0)
    else:
        res = wilcoxon(model, baseline)
        out.update(statistic=float(res.statistic), p_value=float(res.pvalue))
    out["delta"] = cliffs_delta(model, baseline)
    out["magnitude"] = delta_magnitude(out["delta"])
    return out
