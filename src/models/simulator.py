"""Carrega o modelo do simulador de commit (#20) e calcula a probabilidade de defeito."""
import joblib
import numpy as np
import pandas as pd
import shap
import sklearn

from src.data.loader import ROOT

MODEL_PATH = ROOT / "models" / "simulador.joblib"

# nome na tela, se e inteiro, valor minimo e explicacao de cada feature de Kamei
FIELDS = {
    "la": ("Linhas adicionadas", True, 0, "Linhas adicionadas pelo commit."),
    "ld": ("Linhas removidas", True, 0, "Linhas removidas pelo commit."),
    "lt": ("Linhas nos arquivos antes do commit", True, 0,
           "Tamanho dos arquivos alterados antes da mudança."),
    "ns": ("Subsistemas alterados", True, 1,
           "Número de subsistemas (pastas de primeiro nível) alterados."),
    "nd": ("Diretórios alterados", True, 1, "Número de diretórios alterados."),
    "nf": ("Arquivos alterados", True, 1, "Número de arquivos alterados."),
    "entrophy": ("Entropia da mudança", False, 0.0,
                 "Quão espalhada a mudança está entre os arquivos. 0 = tudo num arquivo só."),
    "ndev": ("Desenvolvedores que já mexeram nos arquivos", True, 1,
             "Quantas pessoas diferentes já alteraram esses arquivos."),
    "age": ("Dias desde a última alteração", False, 0.0,
            "Média de dias desde a última mudança nos arquivos alterados."),
    "nuc": ("Mudanças anteriores nos arquivos", True, 0,
            "Número de commits anteriores que mexeram nesses arquivos."),
    "exp": ("Commits anteriores do autor", True, 0, "Experiência do autor no projeto."),
    "rexp": ("Experiência recente do autor", False, 0.0,
             "Commits anteriores do autor, com peso maior para os mais recentes."),
    "sexp": ("Commits do autor no subsistema", True, 0,
             "Experiência do autor nos subsistemas alterados."),
    "fix": ("O commit corrige um defeito", True, 0,
            "Marque se o commit é uma correção de bug."),
}

GROUPS = {
    "Tamanho": ["la", "ld", "lt"],
    "Difusão": ["ns", "nd", "nf", "entrophy"],
    "Histórico dos arquivos": ["ndev", "age", "nuc"],
    "Experiência do autor": ["exp", "rexp", "sexp"],
    "Propósito": ["fix"],
}


def load(path=MODEL_PATH):
    return joblib.load(path)


def same_sklearn(bundle):
    return bundle["sklearn_version"] == sklearn.__version__


def predict(bundle, values):
    x = np.array([[float(values[f]) for f in bundle["features"]]])
    return float(bundle["model"].predict_proba(x)[0, 1])


def percentile(bundle, prob):
    # % dos commits da avaliacao com probabilidade menor que essa
    q = bundle["quantiles"]
    return 100 * np.searchsorted(q, prob) / len(q)


def load_explainer(bundle):
    return shap.TreeExplainer(bundle["forest"])


def contributions(explainer, bundle, values):
    # quanto cada feature empurra o score do random forest (antes da calibracao)
    x = np.array([[float(values[f]) for f in bundle["features"]]])
    sv = explainer.shap_values(x)[0, :, 1]
    out = pd.DataFrame({"feature": bundle["features"],
                        "nome": [FIELDS[f][0] for f in bundle["features"]],
                        "valor": x[0], "contribuicao": sv})
    out = out.reindex(out["contribuicao"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True), float(explainer.expected_value[1])
