"""Comparacao final (#7): junta os resultados de todos os modelos e compara com o LApredict.

Le todos os results/metrics/experimentos_*.csv, entao modelo novo entra sozinho.
"""
import pandas as pd

from src.data.loader import ROOT

METRICS_DIR = ROOT / "results" / "metrics"
METRICS = ["auc", "pr_auc", "f1", "mcc"]
KEYS = ["model", "project", "setting"]


def load_results(metrics_dir=METRICS_DIR):
    files = sorted(metrics_dir.glob("experimentos_*.csv"))
    if not files:
        raise FileNotFoundError(f"nenhum experimentos_*.csv em {metrics_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def aggregate(results):
    # media das seeds: o par do teste estatistico e projeto x cenario
    # (o lapredict nao tem busca, entao as seeds dele sao identicas)
    agg = results.groupby(KEYS, as_index=False)[METRICS].mean()
    agg["source"] = "nosso"
    return agg
