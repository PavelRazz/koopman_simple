"""Подготовка численного эталона и поля исходной ОДУ, без обучения."""
from dataclasses import replace
import numpy as np
import system
from config import Config
from dataset import simulate


def prepare_reference(data: dict, c: Config) -> dict:
    """Поле f(x,y,u) для отдельных фиксированных u, не для смешанного датасета.

    Применимо к двум состояниям и одному входу. Для систем большей размерности
    нужно явно выбрать сечение и значения остальных компонент.
    """
    reference_config = replace(c, integrator='DOP853', rtol=1e-11, atol=1e-13)
    indices = np.linspace(0, len(data['states'])-1, min(6, len(data['states'])), dtype=int)
    initial = data['states'][indices, 0]
    # Три значения внутри заданного диапазона, по умолчанию -0.5, 0, 0.5.
    lo, hi = c.control_low[0], c.control_high[0]
    values = np.unique(lo+(hi-lo)*np.array([.25, .5, .75]))
    nt = round(c.duration / c.dt)
    trajectories = []
    for u in values:
        control = np.full((nt, 1), u)
        trajectories.append(np.stack([simulate(x0, control, reference_config) for x0 in initial]))
    combined = np.concatenate([data['states'].reshape(-1, 2)] + [x.reshape(-1, 2) for x in trajectories])
    lower, upper = combined.min(axis=0), combined.max(axis=0)
    margin = .07*np.maximum(upper-lower, .1)
    lower, upper = lower-margin, upper+margin
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 23), np.linspace(lower[1], upper[1], 19))
    params = c.parameters
    fields = []
    for u in values:
        f = np.array([system.rhs(0., point, np.array([u]), params)
                      for point in np.column_stack((xx.ravel(), yy.ravel()))])
        fields.append(f.reshape(xx.shape+(2,)))
    return dict(values=values, trajectories=trajectories, xx=xx, yy=yy,
                fields=fields, lower=lower, upper=upper)
