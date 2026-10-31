"""Carrega as 14 features de Kamei com o mesmo split do artigo-base (ISSTA'21).

- WP (within-project): raw/<p>/<p>_k_feature.csv, primeiros 80% treino, últimos 20% teste
  (mesma regra de JIT_Baseline/prepare_data.py; o CSV já está em ordem cronológica).
- CP (cross-project): raw/<p>/cross/k_train.csv (outros 5 projetos) e k_test.csv.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

PROJECTS = ["qt", "openstack", "jdt", "platform", "gerrit", "go"]
FEATURES = ["ns", "nd", "nf", "entrophy", "la", "ld", "lt", "fix",
            "ndev", "age", "nuc", "exp", "rexp", "sexp"]


def _clean(df, features):
    """Mesmo pré-processamento do original: bool -> int, NaN -> média do próprio arquivo."""
    X = df[features].replace({True: 1, False: 0}).astype(float)
    X = X.fillna(X.mean())
    y = (df["bug"].astype(int) > 0).astype(int).to_numpy()
    return X.to_numpy(), y, df["_id"].to_numpy()


def load_split(project, setting="WP", features=FEATURES):
    """Retorna dict com X_train, y_train, X_test, y_test, ids_test."""
    if setting == "WP":
        df = pd.read_csv(RAW / project / f"{project}_k_feature.csv")
        cut = int(len(df) * 0.8)
        train, test = df.iloc[:cut], df.iloc[cut:]
    elif setting == "CP":
        train = pd.read_csv(RAW / project / "cross" / "k_train.csv")
        test = pd.read_csv(RAW / project / "cross" / "k_test.csv")
    else:
        raise ValueError(f"setting inválido: {setting}")

    X_train, y_train, _ = _clean(train, features)
    X_test, y_test, ids_test = _clean(test, features)
    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test, "ids_test": ids_test}
