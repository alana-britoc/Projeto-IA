"""Roda modelo x projeto x cenario x seed e salva em results/metrics/experimentos.csv

ex: python -m src.evaluation.run_experiment --models lr_jit --projects qt --seeds 0 1 2
"""
import argparse
import json
import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.data.loader import PROJECTS, ROOT, load_split
from src.evaluation.metrics import best_threshold, compute_metrics
from src.models.registry import MODELS, get_model

OUTPUT = ROOT / "results" / "metrics" / "experimentos.csv"


def tune(model, params, X, y, seed, n_iter):
    # refit=False pq o run_one treina o modelo final depois
    search = RandomizedSearchCV(model, params, n_iter=n_iter, scoring="roc_auc",
                                cv=TimeSeriesSplit(n_splits=3), random_state=seed, n_jobs=-1,
                                refit=False)
    search.fit(X, y)
    best = clone(model).set_params(**search.best_params_)
    return best, search.best_params_, search.best_score_


def run_one(model_name, project, setting, seed, n_iter=20):
    features, model, params = get_model(model_name, seed)
    d = load_split(project, setting, features)
    X, y = d["X_train"], d["y_train"]

    best_params, cv_auc, tune_s = {}, None, 0.0
    if params and n_iter > 0:
        t0 = time.perf_counter()
        model, best_params, cv_auc = tune(model, params, X, y, seed, n_iter)
        tune_s = time.perf_counter() - t0

    cut = int(len(y) * 0.8)
    val_model = clone(model).fit(X[:cut], y[:cut])
    threshold = best_threshold(y[cut:], val_model.predict_proba(X[cut:])[:, 1])

    t0 = time.perf_counter()
    model.fit(X, y)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_prob = model.predict_proba(d["X_test"])[:, 1]
    test_s = time.perf_counter() - t0

    row = {"model": model_name, "project": project, "setting": setting, "seed": seed,
           "n_train": len(y), "n_test": len(d["y_test"])}
    row.update(compute_metrics(d["y_test"], y_prob, threshold))
    row.update({"n_iter": n_iter if params else 0, "cv_auc": cv_auc,
                "best_params": json.dumps(best_params, default=float),
                "tune_s": tune_s, "train_s": train_s, "test_s": test_s})
    return row


def save(row):
    # salva a cada execucao pra nao perder a grade se parar no meio
    new = pd.DataFrame([row])
    # junta com o que ja tinha rodado antes
    if OUTPUT.exists():
        old = pd.read_csv(OUTPUT)
        keys = ["model", "project", "setting", "seed"]
        old = old[~old.set_index(keys).index.isin(new.set_index(keys).index)]
        new = pd.concat([old, new], ignore_index=True)
    # escreve em outro arquivo antes pra nao corromper o csv se cair na escrita
    tmp = OUTPUT.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(OUTPUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--projects", nargs="+", default=PROJECTS, choices=PROJECTS)
    parser.add_argument("--settings", nargs="+", default=["WP", "CP"], choices=["WP", "CP"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-iter", type=int, default=20,
                        help="combinações testadas na busca de hiperparâmetros (0 desliga)")
    args = parser.parse_args()

    for model_name in args.models:
        for project in args.projects:
            for setting in args.settings:
                for seed in args.seeds:
                    r = run_one(model_name, project, setting, seed, args.n_iter)
                    save(r)
                    print(f"{model_name:10} {project:9} {setting} seed={seed}  "
                          f"AUC={r['auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
                          f"F1={r['f1']:.4f}  MCC={r['mcc']:.4f}", flush=True)

    print(f"\nSalvo em {OUTPUT}")


if __name__ == "__main__":
    main()
