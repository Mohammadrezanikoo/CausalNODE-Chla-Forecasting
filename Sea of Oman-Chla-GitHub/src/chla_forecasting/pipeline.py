"""Train manuscript-related forecasters with explicit horizon alignment and no scaler leakage."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .causal import load_causal_parents
from .data import PAPER_TOP_DRIVERS, Samples, ScaledSamples, load_daily_csv, make_samples, scale_samples
from .interpretability import elementary_effects, stochastic_predictions
from .metrics import calculate_metrics
from .models import create_model


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_tensors(name: str, scaled: ScaledSamples, indices: np.ndarray,
                 device: torch.device) -> tuple[torch.Tensor, ...]:
    indices = np.asarray(indices, dtype=int)
    if name == 'causal_ode':
        return (torch.from_numpy(scaled.current[indices]).to(device),
                torch.from_numpy(scaled.exog[indices]).to(device))
    return (torch.from_numpy(scaled.seq[indices]).to(device),)


def iter_loader(name: str, scaled: ScaledSamples, indices: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    xx = (scaled.current[indices], scaled.exog[indices]) if name == 'causal_ode' else (scaled.seq[indices],)
    parts = [torch.tensor(t, dtype=torch.float32) for t in xx]
    yy = torch.tensor(scaled.target[indices], dtype=torch.float32)
    return DataLoader(TensorDataset(*parts, yy), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer,
              name: str, horizon: int, device: torch.device) -> float:
    model.train(mode=optimizer is not None)
    errors, n = 0.0, 0
    for values in loader:
        tensors = [v.to(device) for v in values]
        xx, target = tensors[:-1], tensors[-1]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(*xx, horizon=horizon) if name == 'causal_ode' else model(*xx)
            loss = nn.functional.mse_loss(pred, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        errors += float(loss.item()) * len(target)
        n += len(target)
    return errors / max(1, n)


def model_parameters(name: str, hidden: int, dropout: float, layers: int,
                     heads: int, integration_steps: int) -> dict:
    return {'hidden': hidden, 'dropout': dropout,
            'layers': layers, 'heads': heads, 'integration_steps': integration_steps}


def instantiate(name: str, samples: Samples, params: dict, device: torch.device) -> nn.Module:
    return create_model(name, n_features=samples.seq.shape[-1],
                        n_drivers=samples.exog.shape[-1], **params).to(device)


def fit_one(name: str, samples: Samples, scaled: ScaledSamples,
            train_idx: np.ndarray, val_idx: np.ndarray, horizon: int, params: dict,
            epochs: int, batch_size: int, lr: float, weight_decay: float,
            patience: int, device: torch.device) -> tuple[nn.Module, list[dict]]:
    model = instantiate(name, samples, params, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = iter_loader(name, scaled, train_idx, batch_size, True)
    val_loader = iter_loader(name, scaled, val_idx, batch_size, False)
    best_loss, best_weights, stale = float('inf'), None, 0
    log = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, opt, name, horizon, device)
        val_loss = run_epoch(model, val_loader, None, name, horizon, device)
        log.append({'epoch': epoch, 'train_mse_scaled': train_loss, 'val_mse_scaled': val_loss})
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, log


def evaluate(name: str, model: nn.Module, scaled: ScaledSamples,
             indices: np.ndarray, horizon: int, device: torch.device,
             batch_size: int = 512) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start + batch_size]
            xx = make_tensors(name, scaled, ix, device)
            predictions = model(*xx, horizon=horizon) if name == 'causal_ode' else model(*xx)
            output.append(predictions.cpu().numpy())
    return scaled.target_scaler.inverse_transform(np.vstack(output)).ravel()


def optimize(name: str, samples: Samples, horizon: int, params: dict,
             trials: int, folds: int, cv_epochs: int, batch_size: int,
             device: torch.device, seed: int) -> tuple[dict, pd.DataFrame]:
    import optuna
    chronological_train = np.flatnonzero(samples.split != 'test')
    if len(chronological_train) <= folds + 10:
        raise ValueError('Not enough training data for cross-validation')
    splitter = TimeSeriesSplit(n_splits=folds)

    def objective(trial):
        tuned = dict(params)
        if name != 'transformer':
            tuned['hidden'] = trial.suggest_categorical('hidden', [32, 64, 96, 128])
        tuned['dropout'] = trial.suggest_float('dropout', 0.0, 0.4)
        if name != 'causal_ode':
            tuned['layers'] = trial.suggest_int('layers', 1, 2)
        if name == 'transformer':
            tuned['hidden'] = trial.suggest_categorical('d_model', [32, 64, 128])
            tuned['heads'] = trial.suggest_categorical('heads', [2, 4, 8])
        if name == 'causal_ode':
            tuned['integration_steps'] = trial.suggest_int('integration_steps', 1, 6)
        lr = trial.suggest_float('lr', 1e-5, 3e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
        rmses = []
        for fold, (i_train, i_val) in enumerate(splitter.split(chronological_train)):
            tr_idx, va_idx = chronological_train[i_train], chronological_train[i_val]
            fold_scaled = scale_samples(samples, tr_idx)
            set_random_seed(seed + fold)
            model, _ = fit_one(name, samples, fold_scaled, tr_idx, va_idx, horizon,
                               tuned, cv_epochs, batch_size, lr, weight_decay,
                               patience=min(8, cv_epochs), device=device)
            y_pred = evaluate(name, model, fold_scaled, va_idx, horizon, device)
            rmses.append(float(np.sqrt(np.mean((y_pred - samples.target[va_idx].ravel()) ** 2))))
            trial.report(float(np.mean(rmses)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(objective, n_trials=trials)
    selected = dict(params)
    selected.update({k: v for k, v in study.best_params.items()
                     if k in selected and k != 'd_model'})
    if name == 'transformer' and 'd_model' in study.best_params:
        selected['hidden'] = study.best_params['d_model']
    selected['lr'] = study.best_params['lr']
    selected['weight_decay'] = study.best_params['weight_decay']
    return selected, study.trials_dataframe()


def train_and_export(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(torch.get_num_threads(), args.threads))
    set_random_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    daily, missing = load_daily_csv(args.data)
    baseline_drivers = ([c for c in daily.columns if c != 'chl'] if args.baseline == 'all'
                        else list(PAPER_TOP_DRIVERS))
    parents = load_causal_parents(args.causal_links, alpha=args.alpha,
                                  max_lag=args.max_lag,
                                  top_k_drivers=args.top_k_drivers)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summary, coverage = [], []
    for model_name in (['causal_ode', 'lstm', 'transformer'] if args.model == 'all' else [args.model]):
        for horizon in args.horizons:
            samples = make_samples(daily, horizon, parents, baseline_drivers,
                                   lookback=args.lookback, lag_max=args.max_lag)
            tr_idx = np.flatnonzero(samples.split == 'train')
            va_idx = np.flatnonzero(samples.split == 'val')
            if len(tr_idx) < 2 or not len(va_idx):
                raise ValueError('Chronological train/validation split produced no samples')
            run_dir = output / model_name / f'tplus{horizon}'
            run_dir.mkdir(parents=True, exist_ok=True)
            selected = model_parameters(model_name, args.hidden, args.dropout,
                                        args.layers, args.heads, args.integration_steps)
            lr, decay = args.lr, args.weight_decay
            if args.trials > 0:
                tuned, trials_df = optimize(model_name, samples, horizon, selected,
                                            args.trials, args.folds, args.cv_epochs,
                                            args.batch_size, device, args.seed)
                trials_df.to_csv(run_dir / 'optuna_trials.csv', index=False)
                lr, decay = tuned.pop('lr'), tuned.pop('weight_decay')
                selected = tuned
            # Scalers and early-stopping fitting are restricted to *training* examples.
            scaled = scale_samples(samples, tr_idx)
            model, log = fit_one(model_name, samples, scaled, tr_idx, va_idx,
                                 horizon, selected, args.epochs, args.batch_size,
                                 lr, decay, args.patience, device)
            pd.DataFrame(log).to_csv(run_dir / 'training_history.csv', index=False)
            pred = np.full(len(samples.split), np.nan)
            for partition in ['train', 'val', 'test']:
                idx = np.flatnonzero(samples.split == partition)
                if len(idx):
                    pred[idx] = evaluate(model_name, model, scaled, idx, horizon, device)
                    metrics = calculate_metrics(samples.target[idx], pred[idx], samples.target_dates[idx])
                    summary.append(dict(model=model_name, horizon=horizon, split=partition, **metrics))
                    print(f'{model_name} t+{horizon} {partition}: NSE={metrics["nse"]:.3f}, RMSE={metrics["rmse_mg_m3"]:.4f}')
            frame = pd.DataFrame({
                'issue_date': samples.issue_dates, 'target_date': samples.target_dates,
                'horizon_days': horizon, 'split': samples.split,
                'observed_chl_mg_m3': samples.target.ravel(), 'predicted_chl_mg_m3': pred,
            })
            if args.mc_passes > 1:
                u_mean, u_sd = np.full(len(samples.split), np.nan), np.full(len(samples.split), np.nan)
                idx = np.flatnonzero(samples.split == 'test')
                for start in range(0, len(idx), args.batch_size):
                    ix = idx[start:start + args.batch_size]
                    inputs = make_tensors(model_name, scaled, ix, device)
                    kwargs = {'horizon': horizon} if model_name == 'causal_ode' else {}
                    mu, sd = stochastic_predictions(model, inputs, args.mc_passes, **kwargs)
                    # Affine inverse scaling: std only scales, no mean shift.
                    u_mean[ix] = mu * scaled.target_scaler.scale_[0] + scaled.target_scaler.mean_[0]
                    u_sd[ix] = sd * scaled.target_scaler.scale_[0]
                frame['mc_dropout_mean_mg_m3'] = u_mean
                frame['mc_dropout_std_mg_m3'] = u_sd
            frame.to_csv(run_dir / 'predictions.csv', index=False)
            if args.sensitivity:
                idx = np.flatnonzero(samples.split == 'test')
                inputs = make_tensors(model_name, scaled, idx, device)
                kwargs = {'horizon': horizon} if model_name == 'causal_ode' else {}
                names = samples.causal_features if model_name == 'causal_ode' else samples.seq_features
                scores = elementary_effects(model, inputs, names,
                                            variable_tensor=1 if model_name == 'causal_ode' else 0,
                                            n_trajectories=args.sensitivity_trajectories, **kwargs)
                pd.DataFrame({'feature': list(scores), 'mu_star_approx': list(scores.values())})\
                  .sort_values('mu_star_approx', ascending=False).to_csv(run_dir / 'oat_sensitivity.csv', index=False)
            torch.save({'state_dict': model.state_dict(), 'model': model_name,
                        'horizon': horizon, 'model_parameters': selected,
                        'seq_features': samples.seq_features,
                        'causal_features': samples.causal_features,
                        'scaler_seq_mean': scaled.seq_scaler.mean_,
                        'scaler_seq_scale': scaled.seq_scaler.scale_,
                        'scaler_exog_mean': scaled.exog_scaler.mean_,
                        'scaler_exog_scale': scaled.exog_scaler.scale_,
                        'scaler_target_mean': scaled.target_scaler.mean_,
                        'scaler_target_scale': scaled.target_scaler.scale_}, run_dir / 'checkpoint.pth')
            (run_dir / 'run_metadata.json').write_text(json.dumps({
                'model': model_name, 'horizon': horizon, 'parameters': selected,
                'learning_rate': lr, 'weight_decay': decay,
                'model_version': 'cleaned-reimplementation-not-validated-against-manuscript',
                'notes': 'Historical PCMCI link provenance not independently verified' if 'reference' in str(args.causal_links) else 'Confirm PCMCI data provenance',
                'missing_calendar_dates': missing,
                'total_samples': len(samples.target),
                'train_samples': len(tr_idx), 'validation_samples': len(va_idx),
                'test_samples': int((samples.split == 'test').sum()),
                'causal_features': samples.causal_features,
                'baseline_features': samples.seq_features,
            }, indent=2), encoding='utf-8')
            coverage.append(dict(model=model_name, horizon=horizon, usable_samples=len(samples.target)))
    pd.DataFrame(summary).to_csv(output / 'all_metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'sample_coverage.csv', index=False)
    print(f'Completed. Local outputs: {output} (excluded by .gitignore)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Daily CSV containing date, chl and exogenous variables')
    parser.add_argument('--causal-links', default='reference/pcmci_chl_drivers.csv',
                        help='PCMCI links; regenerate on training dates for proper out-of-sample evaluation')
    parser.add_argument('--out', default='results')
    parser.add_argument('--model', choices=['all', 'causal_ode', 'lstm', 'transformer'], default='all')
    parser.add_argument('--horizons', type=int, nargs='+', choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument('--baseline', choices=['paper-top-seven', 'all'], default='paper-top-seven')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--top-k-drivers', type=int, default=7)
    parser.add_argument('--max-lag', type=int, default=7)
    parser.add_argument('--lookback', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--integration-steps', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--trials', type=int, default=0,
                        help='Optuna trial count (use 100 for paper-method rerun)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--cv-epochs', type=int, default=20)
    parser.add_argument('--mc-passes', type=int, default=0, help='Use 100 for MC dropout analysis')
    parser.add_argument('--sensitivity', action='store_true', help='Morris-style screening; NOT full Morris design')
    parser.add_argument('--sensitivity-trajectories', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    train_and_export(args)


if __name__ == '__main__':
    main()
