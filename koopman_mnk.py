"""Находим A и B методом наименьших квадратов: Znext ≈ Z @ A.T + U @ B.T."""
import numpy as np
from observables import lift, decoder
from model import diagnostics, spectrum


def fit_model(X, Y, U, powers, ridge=0.0):
    # Один масштаб для текущих и следующих состояний.
    scale = np.maximum(np.max(np.abs(X), axis=0), 1e-12)
    Z = lift(X, powers, scale)
    target = lift(Y, powers, scale)
    design = np.column_stack((Z, U))

    # Нормировка столбцов улучшает численную обусловленность МНК.
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    D = design / norms
    if ridge > 0:
        D = np.vstack((D, np.sqrt(ridge) * np.eye(D.shape[1])))
        target = np.vstack((target, np.zeros((design.shape[1], target.shape[1]))))

    coefficients = np.linalg.lstsq(D, target, rcond=None)[0]
    coefficients = coefficients / norms[:, None]
    dimension = len(powers)
    A = coefficients[:dimension].T.copy()
    B = coefficients[dimension:].T.copy()

    # Если в базис включена 1, она остаётся равной 1 на следующем шаге.
    for i in np.flatnonzero(np.all(powers == 0, axis=1)):
        A[i] = 0
        A[i, i] = 1
        B[i] = 0

    model = dict(A=A, B=B, C=decoder(powers, scale),
                 powers=powers, scale=scale, ridge=ridge, backend='mnk')
    model.update(diagnostics(design))
    model.update(spectrum(model))
    return model
