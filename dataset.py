"""Генерация траекторий, пар X/Y/U и сохранение в NPZ."""
from dataclasses import asdict
import json
import numpy as np
from scipy.integrate import solve_ivp
import system


def simulate(initial, controls, c):
    """Интегрируем одну траекторию. На каждом интервале dt управление постоянно."""
    q = np.array(initial, dtype=float)
    states = [q.copy()]

    def f(t, state, u):
        return system.rhs(t, state, u, c.parameters)

    for k, u in enumerate(controls):
        if c.integrator == 'RK4':
            h = c.dt / c.substeps
            for j in range(c.substeps):
                t = k * c.dt + j * h
                k1 = f(t, q, u)
                k2 = f(t + h/2, q + h*k1/2, u)
                k3 = f(t + h/2, q + h*k2/2, u)
                k4 = f(t + h, q + h*k3, u)
                q = q + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        else:
            solution = solve_ivp(
                lambda t, state: f(t, state, u),
                (k*c.dt, (k+1)*c.dt), q,
                method=c.integrator, rtol=c.rtol, atol=c.atol,
            )
            q = solution.y[:, -1]
        states.append(q.copy())
    return np.array(states)


def pairs(states, controls):
    """Каждая строка: текущее состояние X, следующее Y, управление U."""
    X = states[:, :-1].reshape(-1, system.N_STATES)
    Y = states[:, 1:].reshape(-1, system.N_STATES)
    U = controls.reshape(-1, system.N_CONTROLS)
    return X, Y, U


def generate_dataset(c, directory):
    """При каждом вызове создаём train, validation и test заново."""
    directory.mkdir(parents=True, exist_ok=True)
    steps = round(c.duration / c.dt)
    times = np.arange(steps + 1) * c.dt
    seeds = np.random.SeedSequence(c.seed).spawn(3)
    names = ('train', 'validation', 'test')
    counts = (c.n_train, c.n_validation, c.n_test)
    hold = steps if c.control_mode == 'constant' else round(c.control_hold / c.dt)

    for name, count, seed in zip(names, counts, seeds):
        rng = np.random.default_rng(seed)
        initial = rng.uniform(c.initial_low, c.initial_high, (count, system.N_STATES))
        controls = np.empty((count, steps, system.N_CONTROLS))
        states = np.empty((count, steps + 1, system.N_STATES))
        for i in range(count):
            # Одна случайная величина на каждый участок постоянного управления.
            blocks = (steps + hold - 1) // hold
            values = rng.uniform(c.control_low, c.control_high, (blocks, system.N_CONTROLS))
            controls[i] = np.repeat(values, hold, axis=0)[:steps]
            states[i] = simulate(initial[i], controls[i], c)
        X, Y, U = pairs(states, controls)
        np.savez_compressed(
            directory / f'{name}.npz',
            times=times, states=states, controls=controls,
            initial_states=initial, X=X, Y=Y, U=U,
            state_names=system.STATE_NAMES, control_names=system.CONTROL_NAMES,
        )
    (directory / 'config.json').write_text(
        json.dumps(asdict(c), ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(f'Создан датасет: {c.n_train}/{c.n_validation}/{c.n_test}')


def load_data(path):
    """Прочитать массивы из NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
