"""Treina e calibra o modelo do simulador de commit (#20, #29)

Uso: python -m src.models.train_simulator
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.data.loader import FEATURES, PROJECTS, RAW, ROOT, _clean
from src.evaluation import plots

OUTPUT = ROOT / "models" / "simulador.joblib"
RF_RESULTS = ROOT / "results" / "metrics" / "experimentos_random_forest.csv"
CALIBRATION_CSV = ROOT / "results" / "metrics" / "calibracao_simulador.csv"
MAX_MB = 10
# com as 264 arvores da busca o arquivo dava 59 MB
SIZE_LIMITS = {"n_estimators": 100, "max_depth": 12}
METHODS = ["sigmoid", "isotonic"]
UNCALIBRATED = "sem calibração"


def chosen_params(path=RF_RESULTS):
    # a mais escolhida entre todos os experimentos do random forest
    best = pd.read_csv(path)["best_params"].value_counts()
    return json.loads(best.index[0]), int(best.iloc[0]), int(best.sum())


def split_project(X, y):
    # em ordem de data: 80% treino, 10% calibracao, 10% avaliacao
    a, b = int(len(y) * 0.8), int(len(y) * 0.9)
    return {"train": (X[:a], y[:a]), "calib": (X[a:b], y[a:b]), "eval": (X[b:], y[b:])}


def load_all_commits():
    parts = {"train": [], "calib": [], "eval": []}
    for project in PROJECTS:
        df = pd.read_csv(RAW / project / f"{project}_k_feature.csv")
        X, y, _ = _clean(df, FEATURES)
        for name, part in split_project(X, y).items():
            parts[name].append(part)
    return {name: (np.vstack([X for X, _ in p]), np.concatenate([y for _, y in p]))
            for name, p in parts.items()}


def calibrate(forest, X_calib, y_calib, X_eval, y_eval):
    probs = {UNCALIBRATED: forest.predict_proba(X_eval)[:, 1]}
    models = {}
    for method in METHODS:
        models[method] = CalibratedClassifierCV(FrozenEstimator(forest), method=method)
        models[method].fit(X_calib, y_calib)
        probs[method] = models[method].predict_proba(X_eval)[:, 1]
    brier = {name: float(brier_score_loss(y_eval, p)) for name, p in probs.items()}
    best = min(METHODS, key=brier.get)
    return models[best], best, probs, brier


def train(data, params, seed=0):
    X_train, y_train = data["train"]
    X_eval, y_eval = data["eval"]
    forest = RandomForestClassifier(**params, random_state=seed, n_jobs=-1).fit(X_train, y_train)
    model, method, probs, brier = calibrate(forest, *data["calib"], X_eval, y_eval)
    X_all = np.vstack([X for X, _ in data.values()])
    bundle = {
        "model": model,
        "forest": forest,
        "method": method,
        "brier": brier,
        "features": FEATURES,
        "params": params,
        "medians": dict(zip(FEATURES, np.median(X_all, axis=0).tolist())),
        "quantiles": np.quantile(probs[method], np.linspace(0, 1, 1001)),
        "auc": float(roc_auc_score(y_eval, probs[method])),
        "defect_rate": float(y_eval.mean()),
        "n_train": int(len(y_train)),
        "n_calib": int(len(data["calib"][1])),
        "n_eval": int(len(y_eval)),
        "sklearn_version": sklearn.__version__,
    }
    return bundle, probs


def save(bundle, path=OUTPUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path.stat().st_size / 1e6


def main():
    params, times, total = chosen_params()
    print(f"Hiperparâmetros da busca (escolhidos em {times} de {total} experimentos): {params}")
    params = {**params, **SIZE_LIMITS}
    print(f"Com o limite de tamanho: {params}")
    data = load_all_commits()
    bundle, probs = train(data, params)
    y_eval = data["eval"][1]

    calib = pd.DataFrame({"metodo": list(bundle["brier"]), "brier": list(bundle["brier"].values()),
                          "prob_media": [probs[m].mean() for m in bundle["brier"]]})
    calib["escolhido"] = calib["metodo"] == bundle["method"]
    calib.to_csv(CALIBRATION_CSV, index=False)
    fig = plots.plot_calibration(y_eval, probs)
    size = save(bundle)

    print(f"Treino {bundle['n_train']}, calibração {bundle['n_calib']}, "
          f"avaliação {bundle['n_eval']} commits")
    print(calib.round(4).to_string(index=False))
    print(f"Defeitos na avaliação: {bundle['defect_rate']:.1%}  |  AUC: {bundle['auc']:.4f}")
    print(f"Salvo em {OUTPUT.relative_to(ROOT)} ({size:.1f} MB, scikit-learn "
          f"{bundle['sklearn_version']}), {CALIBRATION_CSV.relative_to(ROOT)} e "
          f"{fig.relative_to(ROOT)}")
    if size > MAX_MB:
        raise SystemExit(f"o arquivo passou de {MAX_MB} MB, diminua n_estimators ou max_depth")


if __name__ == "__main__":
    main()
