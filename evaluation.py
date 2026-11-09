"""Сравнение готовой модели с настоящими траекториями; без обучения."""
import csv
import numpy as np
from model import predict_one, rollout


def rmse(error, axis=None):
    """Корень из среднего квадрата ошибки."""
    return np.sqrt(np.mean(error**2, axis=axis))


def evaluate(model, data, c):
    times = data['times']
    actual = data['states']
    controls = data['controls']
    if c.forecast_horizon is not None:
        steps = round(c.forecast_horizon / c.dt)
        times = times[:steps + 1]
        actual = actual[:, :steps + 1]
        controls = controls[:, :steps]

    one_step = predict_one(model, actual[:, :-1], controls)
    predicted = rollout(model, actual[:, 0], controls)
    error = predicted[:, 1:] - actual[:, 1:]
    one_error = one_step - actual[:, 1:]

    metrics = dict(
        dimension=len(model['A']),
        rank=int(model['rank']),
        condition=float(model['condition']),
        rho_A=float(model['rho_A']),
        RMSE_one_step=float(rmse(one_error)),
        RMSE_rollout=float(rmse(error)),
        MAE=float(np.mean(np.abs(error))),
        RMSE_final=float(rmse(error[:, -1])),
        mean_final_distance=float(np.mean(np.linalg.norm(error[:, -1], axis=-1))),
        n_trajectories=len(actual),
        duration=float(times[-1] - times[0]),
    )
    for j in range(actual.shape[-1]):
        metrics[f'RMSE_state_{j}'] = float(rmse(error[..., j]))
        metrics[f'RMSE_one_step_state_{j}'] = float(rmse(one_error[..., j]))

    # Ошибка на всём интервале (0, horizon], а не только в его конце.
    elapsed = times[1:] - times[0]
    for horizon in c.horizons:
        if horizon <= elapsed[-1] + 1e-10:
            mask = elapsed <= horizon + 1e-10
            metrics[f'RMSE_{horizon:g}s'] = float(rmse(error[:, mask]))

    predictions = dict(times=times, actual=actual, predicted=predicted,
                       one_step=one_step, controls=controls)
    return metrics, predictions


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
