"""Общий формат, диагностика и прогноз. Здесь нет алгоритма обучения."""
from pathlib import Path
import numpy as np
from observables import lift


def diagnostics(design: np.ndarray) -> dict:
    """Ранг и обусловленность исходной и нормированной матриц регрессоров."""
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    s = np.linalg.svd(design/norms, compute_uv=False)
    rank = int(np.sum(s > max(design.shape)*np.finfo(float).eps*s[0]))
    return dict(rank=rank, design_columns=design.shape[1],
                condition=float(s[0]/s[-1]) if rank == design.shape[1] else np.inf,
                condition_raw=float(np.linalg.cond(design)))


def spectrum(m: dict) -> dict:
    """A.T w = lambda w; psi(x)=Phi(x).T w (без сопряжения w).

    Это собственные функции конечномерного приближения при u=0,
    а не доказанные собственные функции исходной нелинейной системы.
    """
    values, vectors = np.linalg.eig(m['A'].T)
    return dict(eigenvalues=values, eigenfunction_coefficients=vectors,
                rho_A=float(np.max(np.abs(values))))


def eigenfunctions(m: dict, states: np.ndarray) -> np.ndarray:
    """Значения приближённых собственных функций в заданных точках."""
    return lift(states, m['powers'], m['scale']) @ m['eigenfunction_coefficients']


def predict_one(m: dict, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Один шаг от истинных состояний; поддерживает пакет траекторий."""
    z = lift(states, m['powers'], m['scale'])
    return (z @ m['A'].T + controls @ m['B'].T) @ m['C'].T


def rollout(m: dict, initial: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Подъём только один раз, затем z[k+1]=Az[k]+Bu[k] без истинных точек."""
    z = lift(initial, m['powers'], m['scale'])
    out = np.empty((len(initial), controls.shape[1]+1, initial.shape[1]))
    out[:, 0] = initial
    for k in range(controls.shape[1]):
        z = z @ m['A'].T + controls[:, k] @ m['B'].T
        out[:, k+1] = z @ m['C'].T
    return out


def save_model(path, model, dt, degree):
    """Сохранить матрицы, базис и масштабы в NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model, dt=dt, degree=degree)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
