"""Тот же базис; матрицы A и B находит библиотечный EDMDc."""
import numpy as np
from observables import lift, decoder
from model import diagnostics, spectrum


def fit_model(X, Y, U, powers, dt):
    import pykoopman as pk

    scale = np.maximum(np.max(np.abs(X), axis=0), 1e-12)
    Z = lift(X, powers, scale)
    Znext = lift(Y, powers, scale)

    # Функции уже вычислены в Z: Identity не добавляет новые наблюдаемые.
    estimator = pk.Koopman(
        observables=pk.observables.Identity(),
        regressor=pk.regression.EDMDc(),
    )
    estimator.fit(Z, y=Znext, u=U, dt=dt)

    model = dict(A=np.array(estimator.A), B=np.array(estimator.B),
                 C=decoder(powers, scale), powers=powers, scale=scale,
                 backend='pykoopman', ridge=0.0)
    model.update(diagnostics(np.column_stack((Z, U))))
    model.update(spectrum(model))
    return model
