# modelos do experimento. pra add um novo e so colocar no MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import FEATURES


def _logistic(seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=7000, random_state=seed))


MODELS = {
    "lapredict": {"features": ["la"], "build": _logistic},
    "lr_jit": {"features": FEATURES, "build": _logistic},
}


def get_model(name, seed=0):
    if name not in MODELS:
        raise KeyError(f"modelo desconhecido: {name}. Opções: {', '.join(MODELS)}")
    spec = MODELS[name]
    return spec["features"], spec["build"](seed)
