"""Model evaluation with explicit physical units (mg/m^3)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def calculate_metrics(observed, predicted, dates=None) -> dict[str, float]:
    obs = np.asarray(observed, dtype=float).ravel()
    pred = np.asarray(predicted, dtype=float).ravel()
    if obs.shape != pred.shape or not len(obs):
        raise ValueError("Observed and predicted vectors must be nonempty and equal length")
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[valid], pred[valid]
    if not len(obs):
        raise ValueError("No finite predictions")
    residual = pred - obs
    denom = np.sum((obs - obs.mean()) ** 2)
    cc = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 and np.std(obs) and np.std(pred) else float('nan')
    out = {
        "n": len(obs), "cc": cc,
        "nse": float(1 - np.sum(residual ** 2) / denom) if denom else float('nan'),
        "rmse_mg_m3": float(np.sqrt(np.mean(residual ** 2))),
        "bias_mg_m3": float(np.mean(residual)),
        "emax_mg_m3": float(pred.max() - obs.max()),
        "emax_percent": float(100 * (pred.max() - obs.max()) / obs.max()) if obs.max() else float('nan'),
    }
    if dates is not None:
        dates = pd.DatetimeIndex(dates)[valid]
        frame = pd.DataFrame({"year": dates.year, "obs": obs, "pred": pred})
        yearly = frame.groupby("year")[["obs", "pred"]].max()
        yearly = yearly.loc[yearly.obs != 0]
        out["mean_abs_annual_peak_error_percent"] = float(
            (100 * (yearly.pred - yearly.obs).abs() / yearly.obs).mean()
        ) if not yearly.empty else float('nan')
    return out
