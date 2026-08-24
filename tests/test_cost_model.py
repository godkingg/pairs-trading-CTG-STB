"""Kiểm tra cost model: phí chỉ phát sinh khi vị thế THAY ĐỔI, và Sharpe net luôn <= gross
khi có giao dịch (phí không thể làm return tốt hơn)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import backtest_with_cost, sharpe_ratio


def _make_spread(n: int = 300, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = 0.85 * values[t - 1] + rng.normal(0, 1)
    return pd.Series(values, index=dates)


def test_zero_cost_rate_means_net_equals_gross():
    spread = _make_spread()
    notional = pd.Series(100.0, index=spread.index)
    bt = backtest_with_cost(spread, notional, cost_rate=0.0)
    pd.testing.assert_series_equal(bt["net_ret"], bt["gross_ret"], check_names=False)


def test_positive_cost_rate_never_improves_sharpe_when_trades_exist():
    spread = _make_spread()
    notional = pd.Series(100.0, index=spread.index)
    bt = backtest_with_cost(spread, notional, cost_rate=0.0015)

    n_trades = (bt["position"].diff().fillna(0) != 0).sum()
    assert n_trades > 0, "Test setup cần ít nhất vài lần đổi vị thế"

    sharpe_gross = sharpe_ratio(bt["gross_ret"])
    sharpe_net = sharpe_ratio(bt["net_ret"])
    assert sharpe_net <= sharpe_gross


def test_cost_is_zero_on_days_without_position_change():
    spread = _make_spread()
    notional = pd.Series(100.0, index=spread.index)
    bt = backtest_with_cost(spread, notional, cost_rate=0.0015)

    unchanged_days = bt["position"].diff().fillna(0) == 0
    assert (bt.loc[unchanged_days, "cost"] == 0).all()
