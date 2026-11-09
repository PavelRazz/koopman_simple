"""Одна система: x' = y, y' = -epsilon*(1-x²)*y - omega²*x + u."""
import numpy as np

STATE_NAMES = ('x', 'y')
CONTROL_NAMES = ('u',)
N_STATES = 2
N_CONTROLS = 1


def rhs(t, state, control, parameters):
    """Численные производные для интегрирования траектории."""
    x, y = state
    epsilon = parameters['epsilon']
    omega = parameters['omega']
    dx = y
    dy = -epsilon * (1 - x**2) * y - omega**2 * x + control[0]
    return np.array([dx, dy])


def polynomial_fields(parameters):
    """Те же уравнения для построения базиса: rhs = f0 + g*u.

    Ключ (a, b) означает x**a * y**b, значение — коэффициент.
    При замене системы измените и rhs, и эту запись.
    """
    epsilon = parameters['epsilon']
    omega = parameters['omega']
    drift = [
        {(0, 1): 1.0},
        {(2, 1): epsilon, (0, 1): -epsilon, (1, 0): -omega**2},
    ]
    inputs = [[{}, {(0, 0): 1.0}]]
    return drift, inputs
