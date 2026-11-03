"""Comparacao final (#7): junta os resultados de todos os modelos e compara com o LApredict.

Le todos os results/metrics/experimentos_*.csv, entao modelo novo entra sozinho.
"""
import pandas as pd

from src.data.loader import ROOT
from src.evaluation.reproduce import REFERENCE, load_reference

METRICS_DIR = ROOT / "results" / "metrics"
METRICS = ["auc", "pr_auc", "f1", "mcc"]
KEYS = ["model", "project", "setting"]
# modelos do artigo que a gente nao roda. LApredict e LR-JIT ficam de fora pq ja temos os nossos
PAPER_MODELS = {"DeepJIT": "deepjit", "CC2Vec": "cc2vec", "DBN-JIT": "dbn_jit"}


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


def paper_results(skip=(), path=REFERENCE):
    # o artigo so publica AUC, o resto fica NaN ate a #6 calcular das predicoes dos autores
    ref = load_reference(path)
    rows = []
    for paper_name, model in PAPER_MODELS.items():
        if model in skip:
            continue
        for col, auc in ref[paper_name].items():
            project, setting = col.rsplit("_", 1)
            if project == "Mean":
                continue
            rows.append({"model": model, "project": project, "setting": setting, "auc": auc})
    df = pd.DataFrame(rows, columns=KEYS + ["auc"]).reindex(columns=KEYS + METRICS)
    df["source"] = "artigo"
    return df


def build_comparison(results, path=REFERENCE):
    ours = aggregate(results)
    # se alguem rodar o modelo de verdade (ex: #6), o nosso substitui o do artigo
    paper = paper_results(skip=set(ours["model"]), path=path)
    return pd.concat([ours, paper], ignore_index=True)
