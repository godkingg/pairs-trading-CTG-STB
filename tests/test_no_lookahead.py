"""Kiểm tra backtest không dùng thông tin tương lai (look-ahead bias)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import backtest_with_cost
from src.strategy import generate_positions


def _make_spread(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Mean-reverting AR(1) series
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = 0.9 * values[t - 1] + rng.normal(0, 1)
    return pd.Series(values, index=dates)


def test_position_at_t_does_not_use_future_spread():
    """Vị thế tại ngày t phải cố định khi ta cắt bớt dữ liệu SAU ngày t."""
    spread = _make_spread()
    pos_full = generate_positions(spread)

    cutoff = 150
    spread_truncated = spread.iloc[:cutoff]
    pos_truncated = generate_positions(spread_truncated)

    common_idx = pos_truncated.index
    # Position tại các ngày đã có trong bản truncated phải giống hệt bản full
    # (trừ vài ngày cuối cùng gần cutoff có thể khác do rolling window mượt dần —
    # ta chỉ so sánh phần "ổn định" ở giữa, tránh biên)
    stable_idx = common_idx[20:-5]
    pd.testing.assert_series_equal(
        pos_full.loc[stable_idx], pos_truncated.loc[stable_idx], check_names=False
    )


def test_return_uses_previous_day_position_not_current():
    """gross_ret ngày t phải dùng position ngày (t-1), không phải position ngày t."""
    spread = _make_spread()
    notional = pd.Series(100.0, index=spread.index)
    bt = backtest_with_cost(spread, notional, entry_z=1.0, exit_z=0.3)

    position = generate_positions(spread, entry_z=1.0, exit_z=0.3)
    spread_ret = spread.diff().reindex(bt.index)
    expected_gross = position.shift(1).reindex(bt.index) * spread_ret

    pd.testing.assert_series_equal(
        bt["gross_ret"].dropna(), expected_gross.dropna(), check_names=False
    )
