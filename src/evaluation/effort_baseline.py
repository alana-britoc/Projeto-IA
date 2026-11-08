"""Referencia sem modelo para as metricas de esforco (#25).

Com a ordem por densidade de risco (probabilidade / linhas), um modelo que da a mesma
probabilidade pra todo commit revisa do menor commit pro maior. Essa ordem ja tira um Popt
alto, entao e a referencia que os modelos precisam superar nas metricas de esforco (ver
Yang et al., 2016: modelos simples por tamanho competem com os supervisionados).

Depende so dos commits de teste, entao nao precisa rodar a grade. Precisa do dataset.

Uso: python -m src.evaluation.effort_baseline
Saida: results/metrics/esforco_referencia.csv
"""
import numpy as np
import pandas as pd

from src.data.loader import PROJECTS, ROOT, SETTINGS, load_split
from src.evaluation.metrics import popt, recall_at_effort

OUTPUT = ROOT / "results" / "metrics" / "esforco_referencia.csv"


def size_only(y_true, effort):
    """Popt e Recall@20% revisando do menor commit pro maior (probabilidade constante)."""
    constant = np.ones(len(y_true))
    return {"popt": popt(y_true, constant, effort),
            "recall_20": recall_at_effort(y_true, constant, effort)}


def reference_table(projects=PROJECTS, settings=SETTINGS):
    rows = []
    for project in projects:
        for setting in settings:
            # so o teste importa; features=["la"] so pra carregar rapido
            d = load_split(project, setting, ["la"])
            rows.append({"project": project, "setting": setting,
                         **size_only(d["y_test"], d["effort_test"])})
    return pd.DataFrame(rows, columns=["project", "setting", "popt", "recall_20"])


def main():
    df = reference_table()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(df.round(3).to_string(index=False))
    print("\nMédia por cenário:")
    print(df.groupby("setting")[["popt", "recall_20"]].mean().round(3).to_string())
    print(f"\nSalvo em {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
