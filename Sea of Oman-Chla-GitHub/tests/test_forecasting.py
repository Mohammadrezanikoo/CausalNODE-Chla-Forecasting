import numpy as np
import pandas as pd
import torch

from chla_forecasting.causal import load_causal_parents
from chla_forecasting.data import PAPER_TOP_DRIVERS, load_daily_csv, make_samples, scale_samples
from chla_forecasting.metrics import calculate_metrics
from chla_forecasting.models import create_model


def mock_data(n=100):
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    t = np.arange(n)
    frame = pd.DataFrame({name: np.sin((t + j) / 10) for j, name in enumerate(PAPER_TOP_DRIVERS)}, index=dates)
    frame['chl'] = 0.8 + 0.2 * np.sin(t / 11)
    frame.index.name = 'date'
    return frame


def test_issue_date_horizon_and_chronological_splits():
    daily = mock_data()
    parents = [('friction_velocity', 3), ('evaporation', 0)]
    s = make_samples(daily, 4, parents)
    assert len(s.target) > 50
    assert ((s.target_dates - s.issue_dates).days == 4).all()
    assert s.split[-1] == 'test'
    assert np.flatnonzero(s.split == 'train').max() < np.flatnonzero(s.split == 'test').min()
    np.testing.assert_allclose(s.target.ravel(), daily.loc[s.target_dates, 'chl'])
    np.testing.assert_allclose(s.current.ravel(), daily.loc[s.issue_dates, 'chl'])
    assert s.causal_features == ['friction_velocity@t-3', 'evaporation@t-0']


def test_scalers_only_fit_training_indices():
    daily = mock_data()
    s = make_samples(daily, 1, [('friction_velocity', 0)])
    idx = np.flatnonzero(s.split == 'train')
    scaled = scale_samples(s, idx)
    np.testing.assert_allclose(scaled.target_scaler.mean_[0], s.target[idx].mean())
    np.testing.assert_allclose(scaled.seq_scaler.mean_, s.seq[idx].reshape(-1, s.seq.shape[-1]).mean(axis=0))


def test_models_output_and_rk4_multiday():
    for name in ['causal_ode', 'lstm', 'transformer']:
        model = create_model(name, n_features=8, n_drivers=3, hidden=16, heads=4)
        with torch.no_grad():
            prediction = model(torch.ones(2, 1), torch.ones(2, 3), horizon=4) if name == 'causal_ode' else model(torch.ones(2, 7, 8))
        assert prediction.shape == (2, 1)
        assert torch.isfinite(prediction).all()


def test_metrics_and_missing_calendar_rows(tmp_path):
    m = calculate_metrics([1, 2, 3], [1, 2, 3])
    assert m['nse'] == 1 and m['rmse_mg_m3'] == 0
    sample = mock_data(100)
    sample = sample.drop(sample.index[30])
    csv = tmp_path / 'sample.csv'
    sample.reset_index().to_csv(csv, index=False)
    got, missing = load_daily_csv(csv)
    assert len(missing) == 1
    assert pd.isna(got.loc['2023-01-31', 'chl'])
    made = make_samples(got, 2, [('friction_velocity', 1)])
    assert not made.target_dates.isin(pd.to_datetime(missing)).any()
