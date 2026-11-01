import numpy as np


def standardize(X_train, X_test, faithful=False):
    """faithful=True replica o original (cada conjunto padronizado com as próprias
    estatísticas, o que usa informação do teste). False = padrão correto (fit no treino)."""
    if faithful:
        z = lambda X: (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        return z(X_train), z(X_test)
    mu, sd = X_train.mean(0), X_train.std(0)
    sd = np.where(sd == 0, 1, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd
