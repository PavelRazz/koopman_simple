"""Запуск проекта: python main.py all. Настройки находятся в config.py."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import system
from config import CONFIG, ROOT
from dataset import generate_dataset, load_data
from observables import derivative_powers, basis_report
from model import save_model, load_model
from evaluation import evaluate, write_csv


def read_splits(root):
    return {
        name: load_data(root / 'data' / f'{name}.npz')
        for name in ('train', 'validation', 'test')
    }


def train_backend(method, data, c, root):
    """Обучение на train; выбор предела степени по validation."""
    if method == 'mnk':
        from koopman_mnk import fit_model
    else:
        from koopman_pykoopman import fit_model

    train = data['train']
    drift, inputs = system.polynomial_fields(c.parameters)
    output = root / 'results' / method
    basis_dir = root / 'results' / 'basis'
    basis_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for degree in c.degrees:
        powers = derivative_powers(drift, degree, c.include_bias)
        report = basis_report(powers, drift, inputs, system.STATE_NAMES)
        (basis_dir / f'basis_degree_{degree}.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
        )
        print(f'Предел {degree}: ' + ', '.join(report['observables']))

        if method == 'mnk':
            model = fit_model(train['X'], train['Y'], train['U'], powers, ridge=c.ridge)
        else:
            model = fit_model(train['X'], train['Y'], train['U'], powers, dt=c.dt)
        save_model(output / f'model_degree_{degree}.npz', model, c.dt, degree)

        metrics, _ = evaluate(model, data['validation'], c)
        rows.append(dict(method=method, degree=degree, **metrics))
        print(f'{method}: validation RMSE = {metrics["RMSE_rollout"]:.6g}')

    write_csv(output / 'validation.csv', rows)
    best = min(rows, key=lambda row: row['RMSE_rollout'])
    selection = dict(best_degree=best['degree'], criterion='validation RMSE_rollout')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')


def evaluate_saved(methods, data, c, root):
    """Все модели оцениваются на одном тестовом датасете."""
    rows = []
    predictions = {}
    models = {}
    for method in methods:
        for degree in c.degrees:
            output = root / 'results' / method
            model = load_model(output / f'model_degree_{degree}.npz')
            metrics, pred = evaluate(model, data['test'], c)
            rows.append(dict(method=method, degree=degree, **metrics))
            label = f'{method} d={degree}'
            predictions[label] = pred
            models[label] = model
            np.savez_compressed(output / f'predictions_degree_{degree}.npz', **pred)
            print(f'{label}: test RMSE = {metrics["RMSE_rollout"]:.6g}')

    write_csv(root / 'results' / 'comparison.csv', rows)
    from visualization import plot_comparison
    plot_comparison(predictions, models, root / 'results' / 'plots', c)


def plot_data(data, c, root):
    from visualization import plot_dataset, plot_phase_reference
    from phase_reference import prepare_reference

    output = root / 'results' / 'plots'
    plot_dataset(data['train'], output, c)
    reference = prepare_reference(data['train'], c)
    plot_phase_reference(data['train'], reference, output, c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='all',
                        help='all, dataset, plot, mnk, pykoopman, evaluate')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--quick', action='store_true', help='Маленький эксперимент для быстрого запуска')
    parser.add_argument('--methods', nargs='+', default=['mnk', 'pykoopman'])
    parser.add_argument('--degrees', nargs='+', type=int)
    args = parser.parse_args()

    c = CONFIG
    if args.quick:
        c = replace(c, n_train=24, n_validation=8, n_test=8, duration=1.0, degrees=(1, 3))
    if args.degrees:
        c = replace(c, degrees=tuple(args.degrees))
    root = args.root.resolve()

    if args.mode in ('dataset', 'all'):
        generate_dataset(c, root / 'data')
    if args.mode == 'dataset':
        return

    data = read_splits(root)
    if args.mode in ('plot', 'all'):
        plot_data(data, c, root)
    if args.mode in ('mnk', 'pykoopman'):
        train_backend(args.mode, data, c, root)
    if args.mode == 'all':
        for method in args.methods:
            train_backend(method, data, c, root)
    if args.mode in ('evaluate', 'all'):
        evaluate_saved(args.methods, data, c, root)


if __name__ == '__main__':
    main()
