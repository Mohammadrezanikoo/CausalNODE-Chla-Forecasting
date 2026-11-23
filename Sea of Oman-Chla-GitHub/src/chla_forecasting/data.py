"""Chronological, origin-aligned features for multi-step chlorophyll-a prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = "chl"
PAPER_TOP_DRIVERS = (
    "northward_turbulent_surface_stress",
    "friction_velocity",
    "evaporation",
    "boundary_layer_height",
    "no3",
    "total_precipitation",
    "10m_u_component_of_wind",
)


def load_daily_csv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a daily time series; explicitly mark missing calendar days as NaN.

    Do not silently interpolate, forward-fill, or shift time in an irregular series.
    Input may contain US-style M/D/YYYY dates or ISO dates.
    """
    frame = pd.read_csv(path)
    if "date" not in frame or TARGET not in frame:
        raise ValueError("Input CSV must contain 'date' and 'chl' columns")
    frame["date"] = pd.to_datetime(frame["date"], format="mixed", dayfirst=False, errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate dates found; resolve data aggregation before forecasting")
    frame = frame.set_index("date").sort_index()
    for name in frame.columns:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    full = pd.date_range(frame.index.min(), frame.index.max(), freq="D", name="date")
    missing_dates = full.difference(frame.index).strftime("%Y-%m-%d").tolist()
    return frame.reindex(full), missing_dates


@dataclass
class Samples:
    """Raw, unscaled forecast samples for one prediction horizon."""

    seq: np.ndarray         # (sample, lookback, n_seq_features)
    exog: np.ndarray        # (sample, n_causal_parent_lag_pairs)
    current: np.ndarray     # observed Chl-a at issue time t, (sample, 1)
    target: np.ndarray      # observed Chl-a at t+h, (sample, 1)
    issue_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    seq_features: list[str]
    causal_features: list[str]
    split: np.ndarray       # 'train', 'val', or 'test' based on *target date*


def make_samples(
    daily: pd.DataFrame,
    horizon: int,
    causal_parents: Sequence[tuple[str, int]],
    baseline_drivers: Sequence[str] = PAPER_TOP_DRIVERS,
    lookback: int = 7,
    lag_max: int = 7,
    train_ratio: float = 0.70,
    validation_ratio_within_train: float = 0.20,
) -> Samples:
    """Only issue-time or historical values appear as inputs; targets are at t+h.

    The 70/30 and internal validation boundaries use calendar positions; the
    split is determined by target date, preventing label leakage at boundaries.
    """
    if horizon not in (1, 2, 3, 4):
        raise ValueError("This manuscript evaluates only horizons 1, 2, 3, and 4")
    if not (0 < train_ratio < 1 and 0 <= validation_ratio_within_train < 1):
        raise ValueError("Invalid chronological split fractions")
    if lookback < 1 or lag_max < 0:
        raise ValueError("Invalid lookback or maximum lag")
    if not causal_parents:
        raise ValueError("No causal parents selected; supply a valid PCMCI links file")
    parent_cols = [p for p, _ in causal_parents]
    missing = sorted(set([TARGET, *baseline_drivers, *parent_cols]) - set(daily.columns))
    if missing:
        raise ValueError(f"Missing required predictor columns: {missing}")
    for name, lag in causal_parents:
        if not (0 <= lag <= lag_max) or name == TARGET:
            raise ValueError(f"Invalid exogenous parent/lag: {name}, {lag}")
    cols = list(dict.fromkeys([*baseline_drivers, TARGET, *parent_cols]))
    vals = daily[cols].to_numpy(dtype=np.float64)
    lookup = {name: cols.index(name) for name in cols}
    n_days = len(daily)
    cutoff = int(n_days * train_ratio)
    val_cutoff = int(cutoff * (1 - validation_ratio_within_train))
    start = max(lookback - 1, lag_max)
    if n_days < start + horizon + 10:
        raise ValueError("Not enough daily observations for selected lag/forecast horizon")

    sequences, exogs, states, targets, issues, dates, splits = [], [], [], [], [], [], []
    seq_names = [*baseline_drivers, TARGET]
    for origin in range(start, n_days - horizon):
        future = origin + horizon
        seq = vals[origin - lookback + 1:origin + 1, [lookup[c] for c in seq_names]]
        exog = np.asarray([vals[origin - lag, lookup[name]] for name, lag in causal_parents])
        state = vals[origin, lookup[TARGET]]
        target = vals[future, lookup[TARGET]]
        # Keep observed data only, with no imputation or implicit time shifting.
        if not (np.isfinite(seq).all() and np.isfinite(exog).all()
                and np.isfinite(state) and np.isfinite(target)):
            continue
        sequences.append(seq)
        exogs.append(exog)
        states.append([state])
        targets.append([target])
        issues.append(daily.index[origin])
        dates.append(daily.index[future])
        splits.append("train" if future < val_cutoff else "val" if future < cutoff else "test")
    if not targets:
        raise ValueError("No complete samples after excluding missing input and target values")
    return Samples(
        seq=np.stack(sequences), exog=np.stack(exogs),
        current=np.asarray(states), target=np.asarray(targets),
        issue_dates=pd.DatetimeIndex(issues), target_dates=pd.DatetimeIndex(dates),
        seq_features=seq_names,
        causal_features=[f"{name}@t-{lag}" for name, lag in causal_parents],
        split=np.asarray(splits),
    )


@dataclass
class ScaledSamples:
    seq: np.ndarray
    exog: np.ndarray
    current: np.ndarray
    target: np.ndarray
    seq_scaler: StandardScaler
    exog_scaler: StandardScaler
    target_scaler: StandardScaler


def scale_samples(samples: Samples, fit_indices: np.ndarray) -> ScaledSamples:
    """Fit EVERY scaler solely on indices from the training partition/fold."""
    fit_indices = np.asarray(fit_indices, dtype=int)
    if fit_indices.size < 2:
        raise ValueError("At least 2 training samples are required")
    sequence_fit = samples.seq[fit_indices]
    seq_scaler = StandardScaler().fit(sequence_fit.reshape(-1, sequence_fit.shape[-1]))
    exog_scaler = StandardScaler().fit(samples.exog[fit_indices])
    target_scaler = StandardScaler().fit(samples.target[fit_indices])
    seq_shape = samples.seq.shape
    return ScaledSamples(
        seq=seq_scaler.transform(samples.seq.reshape(-1, seq_shape[-1])).reshape(seq_shape).astype(np.float32),
        exog=exog_scaler.transform(samples.exog).astype(np.float32),
        current=target_scaler.transform(samples.current).astype(np.float32),
        target=target_scaler.transform(samples.target).astype(np.float32),
        seq_scaler=seq_scaler,
        exog_scaler=exog_scaler,
        target_scaler=target_scaler,
    )
