"""Read statistically selected lag-specific exogenous PCMCI links."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_causal_parents(
    path: str | Path,
    alpha: float = 0.05,
    max_lag: int = 7,
    top_k_drivers: int | None = 7,
) -> list[tuple[str, int]]:
    """Select strongest drivers by summed absolute partial correlation per lag.

    Archived links have unverified training-only provenance. To perform proper
    prospective testing, regenerate links using `chla-discover` on train ONLY.
    """
    df = pd.read_csv(path)
    mapping = {
        "Driver": "driver", "Lag (days)": "lag", "Effect_Strength": "effect",
        "Effect Strength": "effect", "P_value": "p_value", "P-value": "p_value",
        "Q_value": "q_value", "Q-value": "q_value",
    }
    df = df.rename(columns=mapping)
    if not {"driver", "lag", "effect"}.issubset(df.columns):
        raise ValueError("PCMCI CSV requires Driver, Lag (days), Effect_Strength fields")
    p_col = "q_value" if "q_value" in df else "p_value" if "p_value" in df else None
    if p_col is None:
        raise ValueError("PCMCI CSV requires P_value or Q_value to filter significant links")
    df = df.dropna(subset=["driver", "lag", "effect", p_col]).copy()
    df["lag"] = df["lag"].astype(int)
    df = df.loc[(df[p_col] < alpha) & (df.lag.between(0, max_lag)) & (df.driver != "chl")]
    if top_k_drivers is not None:
        if top_k_drivers < 1:
            raise ValueError("top_k_drivers must be positive")
        strength = df.assign(abs_effect=df.effect.abs()).groupby("driver")["abs_effect"].sum()
        selected = strength.sort_values(ascending=False).head(top_k_drivers).index
        df = df.loc[df.driver.isin(selected)]
    parents = sorted(set(zip(df.driver.astype(str), df.lag.astype(int))))
    if not parents:
        raise ValueError("No significant non-target parent links found")
    return parents
