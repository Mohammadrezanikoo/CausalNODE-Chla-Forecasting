# Scientific and code-provenance audit

This audit was produced from the supplied manuscript, supplementary document, and `Era5.zip` code archive on 20 November 2026. It describes evidence in the supplied materials; it is not an independent rerun of the study.

## Original archived code mapped to the manuscript

| Paper component | Relevant original archive files | Repository implementation |
|---|---|---|
| Lagged causal driver features | `Causal t+1.txt`, `Causal t+2.txt`, `Causal t+3.txt`, `casual t+4.txt` | `causal.py`, `data.py` |
| RK4 Neural ODE at t+1 to t+4 | Same files, plus `Causal ODE (RK4) only.txt` | `models.py::CausalNeuralODE` |
| LSTM t+1 to t+4 | `LSTM t+1.txt`, `LSTM T+2.txt`, `LSTM t+3.txt`, `LSTM T+4.txt` | `models.py::LSTMForecaster` |
| Transformer t+1 to t+4 | `Trans t+1.txt`, `Trans t+2.txt`, `Trans t+3.txt`, `trans t+4.txt` | `models.py::TransformerForecaster` |
| Optuna / cross-validation | `Trans t+2.txt`, `trans t+4.txt`, `t+5.txt` (out-of-scope horizon) | `pipeline.py::optimize` (reimplemented five-fold time-series CV) |
| PCMCI evidence | `New folder/pcmci_chl_drivers.csv`, `pcmci_results_matrix.csv`, logs | `reference/*.csv`, optional `discovery.py` |
| Sensitivity and uncertainty | `senst+1lstm.txt`, figure and output files | `interpretability.py` (approximate sensitivity; documented scope) |
| Model evaluation | Metric functions repeated across forecasts | `metrics.py`, single implementation |

## Excluded (not useful as public reproducible source)

- Numerous duplicate figure PNGs and results spreadsheets (outputs, not code).
- `*.pth` trained checkpoints: multiple alternative configurations; weight-to-source matching cannot be verified, and loading untrusted pickled model files is risky.
- Large or license-sensitive NetCDF/ERA5/source data exports. The combined CSV is not included; data access and license terms require author verification.
- Broken/incomplete code drafts (`Causal ODE (RK4) only.txt` has syntax problems; `Causal t+2.txt` starts with unexpected indentation; `Trans t+2.txt` ends with an incomplete statement); these were consolidated by implementation rather than copied as runnable scripts.
- `t+5.txt` and counterfactual/Causal-LSTM/CPINO experiments not part of the manuscript's stated three-model t+1 to t+4 benchmarking. Figures and counterfactual outputs can be added separately if they are later determined to be essential to a particular manuscript result.
- `best.txt` and `std.txt` contain handwritten or hardcoded configuration/plotting output, not a sufficiently reproducible primary analysis pipeline.

## Manuscript-versus-code differences requiring author review

| Issue | Observed difference | How this repository handles it |
|---|---|---|
| Standardization leakage | Archived forecast scripts apply `StandardScaler.fit_transform` before splitting the full time series. | Refits scalers exclusively on training dates and separately within each Optuna fold. Historical numbers may change. |
| Important friction driver | Original forecast scripts hard-code six physical/biogeochemical drivers and omit `friction_velocity`; the manuscript identifies friction velocity among the dominant causal drivers. | Selects lagged exogenous parents from a PCMCI link table and includes friction velocity when selected. |
| Missing data and chronology | Archived merged `ERA.csv` has 12,050 rows spanning 1993-01-01 to 2025-12-30, but dates 1999-12-31 and 2024-12-31 are absent; `cloud_base_height` has 2,635 missing values. Manuscript §2.8 says no date/observation gaps and gives 2025-12-21 as the end date. | Reindexes to daily intervals and skips windows with missing required values; does not silently interpolate. The authors should reconcile the record dates, missing values and paper narrative. |
| Training/validation | Some scripts further subdivide a nominal 70% train set as 80/20. | Exposes 56% train, 14% validation, 30% test explicitly. |
| Optuna CV and trials | Manuscript specifies 100 trials and five folds. The provided Transformer Optuna scripts use a three-fold configuration in some runs; many other scripts use fixed tuned parameters. | Supports explicitly requested 100 trials and five time-series folds; **does not claim** to reproduce original Optuna trials or parameters. |
| Causal lag provenance | Manuscript states PCMCI discovery was applied to training only. The archive contains PCMCI outputs but no unambiguous end-to-end PCMCI source showing the fitting partition. | `discovery.py` performs fresh PCMCI on the first 70%; reference links are labeled as provenance-uncertain. |
| Forecast horizon integration | Some original t+3/t+4 RK4 scripts integrate a unit interval into a target several days ahead rather than integrating h daily steps. | Performs h daily RK4 integrations with substeps per day, training each horizon independently. This is a **scientific change** that may change performance. |
| LSTM/Transformer inputs | Some original LSTM/Transformer scripts use selected lagged feature vectors and slightly different target offsets; forecast origin alignment can be ambiguous. | Uses last `lookback` days through issue time `t`, with target at exactly `t+h`. Baseline drivers are configurable. |
| MC dropout | Archived analysis and manuscript refer to 100 stochastic passes. Some original Neural ODE network drafts contain no dropout layer. | Adds configurable dropout to model architectures so MC dropout is operative; adds opt-in `--mc-passes 100`. This changes model structure relative to some drafts. |
| Sensitivity method | Manuscript refers to Morris trajectories; archived code does not establish a complete validated global Morris design for every model. | Offers transparently labeled randomized OAT / Morris-style screening, not a claim of strict Morris global sensitivity indices. |
| Numeric reproduction | Manuscript reports Causal Neural ODE correlation coefficients 0.992/0.964/0.932/0.894 and NSE 0.982/0.925/0.863/0.786 for t+1/t+2/t+3/t+4. | **Not independently reproduced**. Use data, configs and held-out checks to reconcile before release as fully reproducible. |

## Interpretation notes

The original archive has **330 entries** across outputs, scripts and data. Only selected manuscript-related algorithmic components were carried forward. A second path—preserving an exact historical runnable snapshot—would require locating the final original source, exact feature list, random seeds, train/test boundaries, checkpoint mapping, and frozen software environment. Without these, a cleaned repository should not advertise identical paper metrics.

The supplementary material contains figures for t+1 through t+4 LSTM and Transformer predictions. Their visual agreement with new output should be assessed after a complete rerun; the images alone are not a source of inference-time model weights or exact preprocessing configuration.

## QA performed

- Static Python syntax checks across the curated repository.
- Unit tests for target dates, leakage-safe scaling, model forward shapes and missing-day handling.
- Short execution test of all three models using 260 rows of the attached CSV (one epoch each). **Not a meaningful performance test**.
- No original checkpoints were unpickled or executed.

## Before public GitHub release

1. Verify that the proposed model input selection (seven leading PCMCI drivers vs all significant links) matches the final manuscript experiments.
2. Check whether the exact reported 100-trial/5-fold settings or random seeds can be reconstructed from the original study logs.
3. Reconcile `ERA.csv` completeness and end-date discrepancies with manuscript §2.8.
4. Run training-only causal discovery on a scientifically justified complete/handled dataset, then rerun forecasts.
5. Confirm results and figures, select a project license, and verify whether external data or trained checkpoints can be redistributed.
