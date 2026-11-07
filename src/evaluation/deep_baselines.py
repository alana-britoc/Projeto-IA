"""Calcula AUC e PR-AUC do DeepJIT e do CC2Vec a partir das predições dos autores.

As predições vêm de ResultAnalysis/RQ4-T8/{deepjit,cc2vec}/<projeto>_<WP|CP>.result
(repositório ZZR0/ISSTA21-JIT-DP). Cada arquivo tem duas colunas separadas por tab:
rótulo real e probabilidade prevista.

Uso (da raiz do projeto):
    python -m src.evaluation.deep_baselines <caminho/para/RQ4-T8>

Saída: results/metrics/deep_baselines.csv (model, project, setting, auc, pr_auc).
"""
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import compute_metrics

DEFAULT_BASE = "../ISSTA21-JIT-DP/ResultAnalysis/RQ4-T8"
OUT = Path("results/metrics/deep_baselines.csv")
MODELS = ["deepjit", "cc2vec"]

def load_results(base):
    rows = []
    for model in MODELS:
        for f in sorted((Path(base) / model).glob("*.result")):
            project, setting = f.stem.rsplit("_", 1)  # ex.: openstack_WP
            df = pd.read_csv(f, sep="\t", header=None, names=["y_true", "y_score"])
            m = compute_metrics(df["y_true"], df["y_score"])
            rows.append({
                "model": model,
                "project": project,
                "setting": setting,
                "auc": m["auc"],
                "pr_auc": m["pr_auc"],
            })
    return pd.DataFrame(rows)


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE)
    if not base.exists():
        sys.exit(f"Pasta não encontrada: {base}. Clone ISSTA21-JIT-DP e passe o caminho de RQ4-T8.")
    df = load_results(base)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()