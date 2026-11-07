import sys
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import compute_metrics

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "../ISSTA21-JIT-DP/ResultAnalysis/RQ4-T8")
OUT = Path("results/metrics/deep_baselines.csv")

rows = []
for model in ["deepjit", "cc2vec"]:
    for f in sorted((BASE / model).glob("*.result")):
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

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))