# modelos do experimento. pra add um novo e so colocar no MODELS
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data.loader import FEATURES


def _logistic(seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=7000, random_state=seed))


def _svm(seed):
    # SVC com kernel e probability=True fica inviavel no CP (~75 mil commits).
    # LinearSVC nao tem predict_proba, entao a calibracao (sigmoid) gera a probabilidade.
    # o cv=3 da calibracao e estratificado sem embaralhar, so dentro do treino.
    # n_jobs=1 pq a busca ja roda em paralelo, e um pool dentro do outro quebra no Windows
    return make_pipeline(StandardScaler(),
                         CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=seed),
                                                method="sigmoid", cv=3, n_jobs=1))


def _mlp(seed):
    # early_stopping separa uma validacao aleatoria, e nao cronologica, dentro do treino.
    # desligado: o treino para quando a loss do treino estabiliza (n_iter_no_change)
    return make_pipeline(StandardScaler(),
                         MLPClassifier(max_iter=200, early_stopping=False, random_state=seed))


MODELS = {
    # lapredict sem busca pra ficar igual ao do artigo
    "lapredict": {"features": ["la"], "build": _logistic},
    "lr_jit": {"features": FEATURES, "build": _logistic,
               "params": {"logisticregression__C": loguniform(1e-3, 1e2)}},
    "svm": {"features": FEATURES, "build": _svm,
            "params": {"calibratedclassifiercv__estimator__C": loguniform(1e-3, 1e2),
                       "calibratedclassifiercv__estimator__class_weight": [None, "balanced"]}},
    "mlp": {"features": FEATURES, "build": _mlp,
            "params": {"mlpclassifier__hidden_layer_sizes": [(16,), (32,), (64,),
                                                             (32, 16), (64, 32)],
                       "mlpclassifier__alpha": loguniform(1e-5, 1e-1),
                       "mlpclassifier__learning_rate_init": loguniform(1e-4, 1e-2),
                       "mlpclassifier__batch_size": [128, 256, 512]}},
}


def get_model(name, seed=0):
    if name not in MODELS:
        raise KeyError(f"modelo desconhecido: {name}. Opções: {', '.join(MODELS)}")
    spec = MODELS[name]
    return spec["features"], spec["build"](seed), spec.get("params")
