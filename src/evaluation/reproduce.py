"""Reproduz LApredict e LR-JIT (Tabela 8 do artigo) e compara com os números publicados.

Uso: python -m src.evaluation.reproduce
Saída: results/metrics/reproducao.csv
"""
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.loader import FEATURES, PROJECTS, ROOT, load_split
from src.preprocessing.scaling import standardize

MODELS = {"LApredict": ["la"], "LR-JIT": FEATURES}
REFERENCE = ROOT / "data" / "reference" / "rq4_t8.csv"


def load_reference(path=REFERENCE):
    """Lê data/reference/rq4_t8.csv (copiado de ResultAnalysis/RQ4-T8)."""
    rows = [l.rstrip(", \n").split(", ")
            for l in open(path)]
    cols = [f"{p}_{t}" for p, t in zip(rows[0][1:], rows[1][1:])]
    return {r[0]: dict(zip(cols, map(float, r[1:]))) for r in rows[2:]}


def run():
    ref = load_reference()
    out = []
    for model, feats in MODELS.items():
        for project in PROJECTS:
            for setting in ("WP", "CP"):
                d = load_split(project, setting, feats)
                Xtr, Xte = standardize(d["X_train"], d["X_test"], faithful=True)
                t0 = time.perf_counter()
                clf = LogisticRegression(max_iter=7000).fit(Xtr, d["y_train"])
                train_s = time.perf_counter() - t0
                auc = roc_auc_score(d["y_test"], clf.predict_proba(Xte)[:, 1])
                paper = ref[model][f"{project}_{setting}"]
                out.append({"model": model, "project": project, "setting": setting,
                            "n_train": len(d["y_train"]), "n_test": len(d["y_test"]),
                            "auc": auc, "auc_paper": paper, "diff": auc - paper,
                            "train_s": train_s})
                print(f"{model:9} {project:9} {setting}  AUC={auc:.4f}  artigo={paper:.4f}  "
                      f"diff={auc - paper:+.4f}")
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results/metrics/reproducao.csv", index=False)
    print("\nMédias:\n", df.groupby(["model", "setting"])[["auc", "auc_paper"]].mean().round(4))


if __name__ == "__main__":
    run()
