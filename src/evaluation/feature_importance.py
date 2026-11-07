import argparse
import json

import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src.data.loader import FEATURES, PROJECTS, load_split
from src.evaluation import plots
from src.evaluation.compare import METRICS_DIR
from src.models.registry import get_model

MODELS = ["random_forest", "hist_gradient_boosting"]
SEED = 0
N_REPEATS = 10


def best_params(model_name, project, setting, seed=SEED, metrics_dir=METRICS_DIR):
    df = pd.read_csv(metrics_dir / f"experimentos_{model_name}.csv")
    row = df[(df["project"] == project) & (df["setting"] == setting) & (df["seed"] == seed)]
    return json.loads(row["best_params"].item())


def fit(model_name, params, X, y, seed=SEED):
    _, model, _ = get_model(model_name, seed)
    return model.set_params(**params).fit(X, y)


def importance(model, X, y, seed=SEED):
    # n_jobs=1 pq o random forest ja usa todos os nucleos
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=N_REPEATS,
                               random_state=seed, n_jobs=1)
    return pd.DataFrame({"feature": FEATURES, "importance_mean": r.importances_mean,
                         "importance_std": r.importances_std})


def interaction_test(model, X_train, y_train, X_test, y_test):
    # mesmo modelo, mas cada arvore so usa uma feature
    additive = clone(model).set_params(interaction_cst="no_interactions").fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    auc_add = roc_auc_score(y_test, additive.predict_proba(X_test)[:, 1])
    return {"auc_com_interacoes": auc, "auc_sem_interacoes": auc_add,
            "diferenca": auc - auc_add}


def run_one(project, setting, metrics_dir=METRICS_DIR):
    d = load_split(project, setting, FEATURES)
    imps, inter = [], None
    for model_name in MODELS:
        params = best_params(model_name, project, setting, metrics_dir=metrics_dir)
        model = fit(model_name, params, d["X_train"], d["y_train"])
        imp = importance(model, d["X_test"], d["y_test"])
        imps.append(imp.assign(model=model_name, project=project, setting=setting))
        if model_name == "hist_gradient_boosting":
            inter = {"project": project, "setting": setting,
                     **interaction_test(model, d["X_train"], d["y_train"],
                                        d["X_test"], d["y_test"])}
    cols = ["model", "project", "setting", "feature", "importance_mean", "importance_std"]
    return pd.concat(imps, ignore_index=True)[cols], inter


def ranking(imp):
    out = imp.groupby(["model", "feature"], as_index=False)["importance_mean"].mean()
    out["posicao"] = out.groupby("model")["importance_mean"].rank(ascending=False).astype(int)
    return out.sort_values(["model", "posicao"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projects", nargs="+", default=PROJECTS, choices=PROJECTS)
    parser.add_argument("--settings", nargs="+", default=["WP", "CP"], choices=["WP", "CP"])
    args = parser.parse_args()

    imps, inters = [], []
    for project in args.projects:
        for setting in args.settings:
            imp, inter = run_one(project, setting)
            imps.append(imp)
            inters.append(inter)
            print(f"{project:9} {setting}  AUC com interações={inter['auc_com_interacoes']:.4f}  "
                  f"sem={inter['auc_sem_interacoes']:.4f}", flush=True)

    imp = pd.concat(imps, ignore_index=True)
    inter = pd.DataFrame(inters)
    imp.to_csv(METRICS_DIR / "importancia_features.csv", index=False)
    inter.to_csv(METRICS_DIR / "importancia_interacoes.csv", index=False)
    fig = plots.plot_feature_importance(imp)

    rank = ranking(imp)
    print("\n" + rank.pivot(index="feature", columns="model", values="posicao")
          .sort_values(MODELS[0]).to_string())
    print(f"\nDiferença média de AUC com e sem interações: {inter['diferenca'].mean():+.4f} "
          f"(maior em {(inter['diferenca'] > 0).sum()} de {len(inter)} casos)")
    print(f"Salvo em {METRICS_DIR / 'importancia_features.csv'}, "
          f"{METRICS_DIR / 'importancia_interacoes.csv'} e {fig}")


if __name__ == "__main__":
    main()
