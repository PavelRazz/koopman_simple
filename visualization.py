"""Только графики готовых массивов — без интегрирования и обучения."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation import rmse
from config import Config


def plot_dataset(data: dict, output: Path, c: Config) -> None:
    """Все компоненты состояния/управления
    фазовая проекция первых двух."""
    output.mkdir(parents=True, exist_ok=True)
    t, q, u = (data[k] for k in ('times', 'states', 'controls'))
    n, m = q.shape[-1], u.shape[-1]
    count = min(len(q), c.plot_trajectories)
    fig, axes = plt.subplots(n+m, 1, figsize=(10, 2.4*(n+m)), squeeze=False, layout='constrained')
    for j in range(n):
        axes[j, 0].plot(t, q[:count, :, j].T, alpha=.6)
        axes[j, 0].set_ylabel(str(data['state_names'][j]))
    for j in range(m):
        for i in range(count):
            axes[n+j, 0].stairs(u[i, :, j], t, alpha=.6)
        axes[n+j, 0].set_ylabel(str(data['control_names'][j]))
    axes[-1, 0].set_xlabel('Time, s')
    fig.suptitle(f'Dataset: first {count} of {len(q)} trajectories')
    fig.savefig(output/'dataset.png', dpi=c.dpi)
    plt.close(fig)
    if n >= 2:
        fig, ax = plt.subplots(layout='constrained')
        for trajectory in q:
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=.2, lw=.7)
        ax.set(xlabel=str(data['state_names'][0]), ylabel=str(data['state_names'][1]),
               title=f'Phase projection: all {len(q)} trajectories')
        fig.savefig(output/'phase.png', dpi=c.dpi)
        plt.close(fig)


def plot_comparison(predictions: dict, models: dict, output: Path, c: Config) -> None:
    """Одинаковая первая тестовая траектория и ошибки по всему тесту."""
    output.mkdir(parents=True, exist_ok=True)
    ref = next(iter(predictions.values()))
    t, actual = ref['times'], ref['actual']
    n = actual.shape[-1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 3*n), squeeze=False, layout='constrained')
    errfig, err_axes = plt.subplots(n, 1, figsize=(11, 3*n), squeeze=False, layout='constrained')
    for j in range(n):
        axes[j, 0].plot(t, actual[0, :, j], 'k', lw=2, label='True')
    for label, d in predictions.items():
        for j in range(n):
            axes[j, 0].plot(t, d['predicted'][0, :, j], '--', label=label)
            error = d['predicted'][..., j] - actual[..., j]
            err_axes[j, 0].plot(t, rmse(error, axis=0), label=label)
    for j in range(n):
        axes[j, 0].set_ylabel(f'state[{j}]')
        axes[j, 0].grid(alpha=.2)
        err_axes[j, 0].set(ylabel=f'RMSE state[{j}]', yscale='symlog')
        err_axes[j, 0].grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    err_axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel('Time, s')
    err_axes[-1, 0].set_xlabel('Time, s')
    fig.savefig(output/'comparison.png', dpi=c.dpi)
    plt.close(fig)
    errfig.savefig(output/'errors.png', dpi=c.dpi)
    plt.close(errfig)
    fig, ax = plt.subplots(layout='constrained')
    angle = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(angle), np.sin(angle), 'k:', lw=.8)
    for label, m in models.items():
        ev = m['eigenvalues']
        ax.scatter(ev.real, ev.imag, label=label, alpha=.6)
    ax.set(xlabel='Re(lambda)', ylabel='Im(lambda)', title='A spectrum: approximate zero-input dynamics')
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    fig.savefig(output/'spectrum.png', dpi=c.dpi)
    plt.close(fig)


def plot_phase_reference(data: dict, reference: dict, output: Path, c: Config) -> None:
    """Слева исходный датасет
    рядом эталоны при фиксированных управлениях."""
    output.mkdir(parents=True, exist_ok=True)
    count = len(reference['values'])
    fig, axes = plt.subplots(1, count+1, figsize=(5.1*(count+1), 5.4),
                             sharex=True, sharey=True, layout='constrained')
    for q in data['states']:
        axes[0].plot(q[:, 0], q[:, 1], alpha=.16, lw=.65)
    axes[0].set_title(f"Датасет: {len(data['states'])} траекторий\nРазные управления u")
    for ax, u, trajectories, field in zip(axes[1:], reference['values'], reference['trajectories'], reference['fields']):
        speed = np.hypot(field[..., 0], field[..., 1])
        dx = np.divide(field[..., 0], speed, out=np.zeros_like(speed), where=speed>1e-14)
        dy = np.divide(field[..., 1], speed, out=np.zeros_like(speed), where=speed>1e-14)
        ax.quiver(reference['xx'], reference['yy'], dx, dy,
                  angles='xy', pivot='mid', color='#74808c', alpha=.5, width=.003,
                  scale=30, headwidth=3.5)
        for i, q in enumerate(trajectories):
            color = plt.cm.tab10(i % 10)
            ax.plot(q[:, 0], q[:, 1], color=color, lw=1.7)
            ax.scatter(*q[0], color=color, marker='o', s=25, zorder=4)
            ax.scatter(*q[-1], color=color, marker='x', s=38, zorder=4)
            k = len(q)//2
            ax.annotate('', xy=q[min(k+3, len(q)-1)], xytext=q[k],
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.7))
        ax.set_title(f'Численный эталон: u = {u:g}\nИсходная ОДУ + поле направлений')
    for ax in axes:
        ax.set(xlabel=str(data['state_names'][0]), xlim=(reference['lower'][0], reference['upper'][0]),
               ylim=(reference['lower'][1], reference['upper'][1]))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.15)
    axes[0].set_ylabel(str(data['state_names'][1]))
    fig.suptitle(f'Фазовые траектории за {c.duration:g} с\n'
                 '● — начало, × — конец. Стрелки поля нормированы: показывают направление, не скорость.\n'
                 'Справа одинаковые начальные точки; эталон — решение ОДУ методом DOP853, не модель Купмана.', fontsize=11)
    fig.savefig(output/'phase_with_reference.png', dpi=c.dpi)
    plt.close(fig)
