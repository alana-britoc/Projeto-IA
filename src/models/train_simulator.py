"""Treina o Random Forest do simulador de commit (#20) e salva em models/simulador.joblib

Usa os commits dos 6 projetos e a combinacao de hiperparametros que a busca da #4 escolheu
mais vezes, com menos arvores e profundidade menor pro arquivo caber em 10 MB.
Precisa do dataset em data/raw/, mas so roda uma vez: o arquivo salvo e versionado.

Uso: python -m src.models.train_simulator
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data.loader import FEATURES, PROJECTS, RAW, ROOT, _clean

OUTPUT = ROOT / "models" / "simulador.joblib"
RF_RESULTS = ROOT / "results" / "metrics" / "experimentos_random_forest.csv"
MAX_MB = 10
# com as 264 arvores da busca o arquivo dava 59 MB
SIZE_LIMITS = {"n_estimators": 100, "max_depth": 12}


def chosen_params(path=RF_RESULTS):
    # a mais escolhida entre os 60 experimentos (6 projetos x 2 cenarios x 5 seeds)
    best = pd.read_csv(path)["best_params"].value_counts()
    return json.loads(best.index[0]), int(best.iloc[0]), int(best.sum())


def load_all_commits():
    X, y = [], []
    for project in PROJECTS:
        df = pd.read_csv(RAW / project / f"{project}_k_feature.csv")
        Xp, yp, _ = _clean(df, FEATURES)
        X.append(Xp)
        y.append(yp)
    return np.vstack(X), np.concatenate(y)


def train(X, y, params, seed=0):
    model = RandomForestClassifier(**params, oob_score=True, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    # out-of-bag: cada commit avaliado so pelas arvores que nao viram ele
    oob = model.oob_decision_function_[:, 1]
    return {
        "model": model,
        "features": FEATURES,
        "params": params,
        "medians": dict(zip(FEATURES, np.median(X, axis=0).tolist())),
        "oob_quantiles": np.quantile(oob, np.linspace(0, 1, 1001)),
        "oob_auc": float(roc_auc_score(y, oob)),
        "defect_rate": float(y.mean()),
        "n_train": int(len(y)),
        "sklearn_version": sklearn.__version__,
    }


def save(bundle, path=OUTPUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path.stat().st_size / 1e6


def main():
    params, times, total = chosen_params()
    print(f"Hiperparâmetros da busca (escolhidos em {times} de {total} experimentos): {params}")
    params = {**params, **SIZE_LIMITS}
    print(f"Com o limite de tamanho: {params}")
    X, y = load_all_commits()
    bundle = train(X, y, params)
    size = save(bundle)
    print(f"{bundle['n_train']} commits, {bundle['defect_rate']:.1%} com defeito")
    print(f"AUC out-of-bag: {bundle['oob_auc']:.4f}")
    print(f"Salvo em {OUTPUT.relative_to(ROOT)} ({size:.1f} MB, scikit-learn "
          f"{bundle['sklearn_version']})")
    if size > MAX_MB:
        raise SystemExit(f"o arquivo passou de {MAX_MB} MB, diminua n_estimators ou max_depth")


if __name__ == "__main__":
    main()
