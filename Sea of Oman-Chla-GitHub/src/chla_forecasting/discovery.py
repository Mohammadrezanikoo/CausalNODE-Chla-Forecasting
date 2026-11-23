"""Optional recomputation of PCMCI causal links on the training period ONLY."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from .data import load_daily_csv


def discover_links(data_csv: str, output_csv: str, alpha: float = 0.05,
                   lag_max: int = 7, train_ratio: float = 0.70,
                   variables: list[str] | None = None) -> pd.DataFrame:
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise RuntimeError('PCMCI requires: pip install -e ".[causal]"') from exc
    data, _ = load_daily_csv(data_csv)
    data = data.iloc[:int(len(data) * train_ratio)]
    names = variables or list(data.columns)
    if "chl" not in names:
        names = [*names, "chl"]
    missing = set(names) - set(data.columns)
    if missing:
        raise ValueError(f"Unknown variables: {sorted(missing)}")
    frame = data[names]
    if frame.isna().any().any():
        bad = frame.isna().sum()
        raise ValueError(
            "PCMCI input contains missing values; choose complete predictor variables "
            "or document and apply an approved missing-data strategy first: "
            + str(bad[bad > 0].to_dict())
        )
    ts_frame = pp.DataFrame(frame.to_numpy(dtype=float), var_names=names)
    pcmci = PCMCI(dataframe=ts_frame, cond_ind_test=ParCorr(significance='analytic'), verbosity=0)
    output = pcmci.run_pcmci(tau_min=0, tau_max=lag_max, pc_alpha=alpha,
                              alpha_level=alpha, fdr_method='fdr_bh')
    p = output['p_matrix']
    q = output.get('q_matrix')
    val = output['val_matrix']
    tid = names.index('chl')
    rows = []
    for idx, name in enumerate(names):
        if name == 'chl':
            continue
        for lag in range(lag_max + 1):
            qval = float(q[idx, tid, lag]) if q is not None else float(p[idx, tid, lag])
            if np.isfinite(qval) and qval < alpha:
                rows.append({'Driver': name, 'Lag (days)': lag,
                             'Effect_Strength': float(val[idx, tid, lag]),
                             'P_value': float(p[idx, tid, lag]), 'Q_value': qval})
    found = pd.DataFrame(rows, columns=['Driver','Lag (days)','Effect_Strength','P_value','Q_value'])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    found.to_csv(output_csv, index=False)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--lag-max', type=int, default=7)
    parser.add_argument('--variables', nargs='+', default=None,
                        help='Optional subset; default is all variables (requires no missing values)')
    args = parser.parse_args()
    result = discover_links(args.data, args.out, args.alpha, args.lag_max, variables=args.variables)
    print(f'Saved {len(result)} significant lag-specific links to {args.out}')


if __name__ == '__main__':
    main()
