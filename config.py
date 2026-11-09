"""Единые настройки. Для новой системы меняйте system.py и этот файл."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    parameters: dict = field(default_factory=lambda: {'epsilon': 0.1, 'omega': 1.0})
    dt: float = 0.02
    duration: float = 5.0
    n_train: int = 400
    n_validation: int = 100
    n_test: int = 100
    seed: int = 42
    initial_low: tuple = (-0.5, -0.5)
    initial_high: tuple = (0.5, 0.5)
    control_low: tuple = (-1.0,)
    control_high: tuple = (1.0,)
    control_mode: str = 'constant'  # случайное постоянное u для КАЖДОЙ траектории
    control_hold: float = 0.5
    integrator: str = 'RK4'  # также RK45, DOP853, Radau, BDF, LSODA
    substeps: int = 4
    rtol: float = 1e-9
    atol: float = 1e-11
    degrees: tuple = (1, 3, 5, 7)  # предел степени при расширении по производным
    include_bias: bool = False  # начинаем с x,y, как на фотографии
    ridge: float = 0.0  # штраф на коэффициенты после нормировки столбцов
    horizons: tuple = (0.5, 1.0, 2.0, 5.0)
    forecast_horizon: float | None = None  # None = вся тестовая траектория
    dpi: int = 150
    plot_trajectories: int = 12

CONFIG = Config()
