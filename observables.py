"""Базис из мономов, появляющихся в производных исходных координат."""
import numpy as np


def lie_derivative(power: tuple, field: list) -> dict:
    """L_f(q**power) = sum_i power[i]*q**(power-e_i)*f_i(q)."""
    result = {}
    for i, exponent in enumerate(power):
        if exponent == 0:
            continue
        base = list(power)
        base[i] -= 1
        for term, coefficient in field[i].items():
            target = tuple(a+b for a, b in zip(base, term))
            result[target] = result.get(target, 0.0) + exponent*coefficient
    # Точные нулевые коэффициенты не порождают лишние наблюдаемые.
    return {p: float(a) for p, a in result.items() if a != 0}


def derivative_powers(drift, max_degree, include_bias=False):
    """Обход по очереди с детерминированным порядком новых мономов.

    Обрабатываем ВСЕ принятые функции до исчерпания очереди. Члены выше
    max_degree не добавляем и не дифференцируем: это явное правило усечения.
    """
    n = len(drift)
    result = [tuple(row) for row in np.eye(n, dtype=int)]
    if include_bias:
        result.insert(0, (0,)*n)
    seen = set(result)
    cursor = 0
    while cursor < len(result):
        terms = lie_derivative(result[cursor], drift)
        cursor += 1
        # Сначала меньшая степень; внутри степени x^a перед x^(a-1)y.
        for p in sorted(terms, key=lambda v: (sum(v), tuple(-k for k in v))):
            if p not in seen and sum(p) <= max_degree:
                seen.add(p)
                result.append(p)
    return np.asarray(result, dtype=int)


def monomial_name(power: tuple, names: tuple) -> str:
    """Человекочитаемая запись монома."""
    terms = [name if p == 1 else f'{name}^{p}' for name, p in zip(names, power) if p]
    return '*'.join(terms) or '1'


def basis_report(powers: np.ndarray, drift: list, inputs: list, names: tuple) -> dict:
    """Полные производные физических (ненормированных) мономов.

    L_truncated — коэффициенты только сохранённых автономных функций.
    d(v)/dt = L_truncated*v + remainder(q) + sum_j u[j]*L_gj(v).
    Эта непрерывная аналитическая матрица НЕ является обученной дискретной A.
    """
    index = {tuple(p): i for i, p in enumerate(powers)}
    L = np.zeros((len(powers), len(powers)))
    rows = []
    for i, p in enumerate(powers):
        autonomous = lie_derivative(tuple(p), drift)
        def describe(terms: dict) -> list:
            return [dict(powers=list(map(int, q)), monomial=monomial_name(q, names),
                         coefficient=a, included=q in index) for q, a in terms.items()]
        for q, a in autonomous.items():
            if q in index:
                L[i, index[q]] = a
        controlled = [lie_derivative(tuple(p), g) for g in inputs]
        rows.append(dict(observable=monomial_name(p, names), drift=describe(autonomous),
                         missing_drift=describe({q:a for q,a in autonomous.items() if q not in index}),
                         control=[describe(terms) for terms in controlled]))
    return dict(dimension=len(powers),
                observables=[monomial_name(p, names) for p in powers],
                powers=powers.tolist(), L_truncated=L.tolist(), derivatives=rows,
                drift_closed=all(not r['missing_drift'] for r in rows),
                has_state_dependent_control=any(sum(q)>0 for p in powers for g in inputs
                                               for q in lie_derivative(tuple(p), g)),
                note='Построение по f0 при u=0. Управляемые члены показаны отдельно; '
                     'дискретная модель Az+Bu — МНК-приближение, не точное поднятие.')


def lift(states: np.ndarray, powers: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Значения выбранных мономов нормированных координат."""
    return np.prod((np.asarray(states)/scale)[..., None, :]**powers, axis=-1)


def decoder(powers: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """C выбирает линейные мономы и возвращает физические единицы."""
    C = np.zeros((len(scale), len(powers)))
    for j, unit in enumerate(np.eye(len(scale), dtype=int)):
        matches = np.flatnonzero(np.all(powers == unit, axis=1))
        C[j, matches[0]] = scale[j]
    return C
