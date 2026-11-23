# Sea of Oman chlorophyll-a forecasting — research code

Cleaned, consolidated Python code related to the manuscript:

> **Causal Neural Ordinary Differential Equation for Multi-Step Marine Chlorophyll-a Forecasting**  
> Zarei, Nikoo, Al-Rawas, Nazari, Hamouda, and Gandomi (manuscript supplied by the authors).

The manuscript describes PCMCI causal discovery, a causally constrained Neural ODE with RK4 integration, LSTM and Transformer baselines, t+1 through t+4 daily forecasts, five-fold Optuna tuning, model evaluation, sensitivity screening and MC-dropout uncertainty.

**Research integrity note:** This repository is a **cleaned reimplementation** of selected archived code, not a claim of bit-for-bit reproduction. It fixes or changes several choices in the supplied drafts; outputs must be recomputed and independently checked against the manuscript before they are cited as reproduced paper results. See [Scientific audit](docs/SCIENTIFIC_AUDIT.md).

## Project layout

```text
src/chla_forecasting/
  data.py              # Daily-index alignment, forecast origins, training-only scalers
  causal.py            # Load and rank lag-specific PCMCI parent variables
  discovery.py         # Optional PCMCI recomputation using Tigramite
  models.py            # Causal Neural ODE (RK4), LSTM, Transformer
  pipeline.py          # Chronological train / validation / test, tuning, prediction export
  metrics.py           # CC, NSE, RMSE, bias, signed EMax, yearly peak error
  interpretability.py  # Monte Carlo dropout and OAT sensitivity screening
reference/             # Small archived PCMCI summaries (provenance needs checking)
tests/                 # Executable unit tests
examples/              # No raw dataset or trained weights are redistributed
docs/SCIENTIFIC_AUDIT.md
```

## Installation

Python 3.10+ and a PyTorch-compatible CPU or GPU environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest -q
```

To recompute PCMCI from scratch, also install the optional Tigramite dependency:

```bash
python -m pip install -e ".[causal]"
```

## Data requirements

The input is a chronologically ordered CSV containing at least a `date` column, `chl` (mg/m³), and physical or biogeochemical predictor columns. The seven leading drivers discussed in the manuscript are:

- `northward_turbulent_surface_stress`
- `friction_velocity`
- `evaporation`
- `boundary_layer_height`
- `no3`
- `total_precipitation`
- `10m_u_component_of_wind`

`--baseline all` allows baselines to use all non-target environmental columns instead of the seven-driver subset. The causal ODE uses *significant, lag-specific exogenous parents* loaded from a CSV, not all baseline columns. Chl-a observed at the forecast issue time is used as the ODE initial state and as a baseline input. Forecasts are aligned to target time `t+h`, not `t`.

The original combined daily data file `ERA.csv` **is intentionally not included**. Its source providers and redistribution license terms should be checked first. The archived copy contains two missing calendar dates and 2,635 missing entries in `cloud_base_height`. The new code does **not** silently interpolate, backfill, or shift dates. See [data notes](data/README.md).

## Running forecasts

Copy your authorized merged CSV to `data/ERA.csv`. Then run:

```bash
# Basic, untuned workflow: all three models at t+1, t+2, t+3, t+4.
chla-train --data data/ERA.csv --causal-links reference/pcmci_chl_drivers.csv \
  --model all --horizons 1 2 3 4 --epochs 200 --out results
```

Using a reference PCMCI file allows a pipeline demonstration but **does not establish that causal discovery was fitted only on the training set**. For defensible held-out evaluation, recompute PCMCI using the chronological training dates and replace the link file:

```bash
# Requires complete data for the selected PCMCI variables; no automatic gap filling.
chla-discover --data data/ERA.csv --out results/pcmci_train_links.csv
chla-train --data data/ERA.csv --causal-links results/pcmci_train_links.csv \
  --model all --horizons 1 2 3 4 --epochs 200 --out results
```

The PCMCI command will reject missing input values, rather than silently changing the time series. You must resolve/document missing values in the research dataset first. The archived data are not suitable for the default complete-case PCMCI command without this step.

To run hyperparameter optimisation with the manuscript's stated **100 trials and five chronological folds**:

```bash
chla-train --data data/ERA.csv --causal-links results/pcmci_train_links.csv \
  --model all --horizons 1 2 3 4 --trials 100 --folds 5 \
  --cv-epochs 20 --epochs 200 --out results
```

**Note:** This will fit many neural networks and may take substantial computing resources. Default `--trials 0` performs no Optuna optimisation. The 100-trial command is method-aligned but uses a reimplemented search space and should not be represented as the original optimisation run.

For uncertainty and optional sensitivity screening:

```bash
chla-train --data data/ERA.csv --causal-links results/pcmci_train_links.csv \
  --model causal_ode --horizons 1 2 3 4 --mc-passes 100 \
  --sensitivity --sensitivity-trajectories 20 --out results
```

`--sensitivity` uses a documented **Morris-style randomized one-factor-at-a-time approximation** on standardized features. It is **not a full classical Morris sampling design**. To reproduce exact Morris results in the manuscript, use a verified implementation of the original design and perturbation bounds.

## Outputs

Each model and lead time writes to `results/{model}/tplus{h}/`:

- `predictions.csv`: issue date, forecast target date, split, observed and predicted chlorophyll-a
- `training_history.csv`: scaled training and validation MSE by epoch
- `run_metadata.json`: hyperparameters, date-gap audit, selected causal inputs, counts
- `checkpoint.pth`: trained weights plus scaling statistics (local only)
- `optuna_trials.csv`: when tuning is enabled
- `oat_sensitivity.csv`: when sensitivity screening is enabled
- Monte Carlo mean and standard deviation columns in `predictions.csv`, when requested

Overall results are summarized in `results/all_metrics.csv` and `results/sample_coverage.csv`. The `results/` directory and checkpoint files are excluded from Git by `.gitignore`.

## Scientific safeguards and limitations

1. **Time order.** Train/validation/test assignments use forecast *target dates*. Default splits correspond approximately to 56% training, 14% validation and 30% held-out test of the daily calendar, with past context allowed for forecast origins.
2. **No leakage from scaling.** Standardization parameters are learned only on the training partition. Within each CV fold, scalers are fitted only on that fold's preceding training dates.
3. **Causal parents.** The archived PCMCI summaries are included as *reference* materials, but their training-only provenance is not independently established. Recalculate them on the training period before claiming out-of-sample causal selection.
4. **Forecast dynamics.** The neural ODE uses separate training for each horizon and integrates via actual RK4 daily steps with exogenous forcing held at origin time, consistent with one option described by the manuscript. This deliberately differs from some draft scripts.
5. **Physical inference.** PCMCI conditional associations and perturbation sensitivities do not by themselves prove causal intervention effects without further assumptions.
6. **Validation status.** Tests verify implementation consistency and basic executability, **not** the manuscript's numerical performance or causal findings. No archived `*.pth` files were loaded (PyTorch pickle weights should be treated as untrusted).

## Publication and citation

Please cite the manuscript and the relevant ERA5 and Copernicus Marine data products in any published work. The authors should decide the intended license and verify permissions for sharing data/weights before public publication; no copyright license is assumed here.
