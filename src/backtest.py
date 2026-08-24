"""Backtest engine dùng chung: cost model, purged K-Fold CV, walk-forward.

Nguồn logic: Ngày 13. Đây là engine "chuẩn" mà mọi project pairs-trading
sau này nên tái sử dụng thay vì viết lại backtest loop mỗi lần.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.features import dynamic_spread, fit_kalman_hedge_ratio, fit_static_hedge_ratio, static_spread
from src.strategy import generate_positions

# Giả định chi phí giao dịch — xem reports/research_note.md để biết căn cứ chọn số này
TRANSACTION_COST = 0.0015  # 0.15% round-trip mỗi lần đổi vị thế
SLIPPAGE = 0.0008          # 0.08% ước lượng do bid-ask spread
TOTAL_COST_RATE = TRANSACTION_COST + SLIPPAGE


def backtest_with_cost(
    spread: pd.Series,
    notional: pd.Series,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    cost_rate: float = TOTAL_COST_RATE,
    window: int = 20,
) -> pd.DataFrame:
    """Backtest z-score mean-reversion, có transaction cost + slippage.

    Cost phát sinh mỗi khi vị thế THAY ĐỔI (entry hoặc exit), tính trên notional
    ước lượng = giá_y + |beta| * giá_x (tổng giá trị 2 chân của cặp).

    Trả về DataFrame: position, gross_ret (trước phí), cost, net_ret (sau phí).
    position.shift(1) được dùng khi tính return để tránh look-ahead bias.
    """
    position = generate_positions(spread, entry_z, exit_z, window)

    spread_ret = spread.diff().reindex(position.index)
    gross_ret = position.shift(1) * spread_ret

    position_change = position.diff().fillna(position.iloc[0]).abs().clip(upper=1)
    notional_aligned = notional.reindex(position.index)
    cost = position_change * cost_rate * notional_aligned

    net_ret = (gross_ret - cost).dropna()

    return pd.DataFrame({
        "position": position, "gross_ret": gross_ret, "cost": cost, "net_ret": net_ret,
    }).dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return float("nan")
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Max drawdown trên đường cumulative P&L (đơn vị giá, không phải %)."""
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    return drawdown.min()


def purged_kfold_splits(n_samples: int, n_splits: int = 5, embargo: int = 5):
    """Time-ordered K-Fold với purging/embargo quanh biên train/test.

    Cần purge vì rolling(window) z-score dùng thông tin xuyên biên giới —
    không purge sẽ bị data leakage giữa train và test.
    Yield (train_idx, test_idx) là mảng vị trí (positional), không phải label.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        test_start, test_end = current, current + fold_size
        test_idx = indices[test_start:test_end]

        embargo_start = max(0, test_start - embargo)
        embargo_end = min(n_samples, test_end + embargo)
        train_idx = np.concatenate([indices[:embargo_start], indices[embargo_end:]])

        yield train_idx, test_idx
        current = test_end


def grid_search_entry_exit(
    spread: pd.Series,
    notional: pd.Series,
    entry_grid: list[float],
    exit_grid: list[float],
    n_splits: int = 5,
    embargo: int = 5,
    min_test_obs: int = 25,
) -> pd.DataFrame:
    """Chọn (entry_z, exit_z) bằng Purged K-Fold CV thay vì chọn tay.

    Trả về bảng mean/std Sharpe (net phí) trên các fold cho mỗi cấu hình, sort giảm dần.
    """
    results = []
    n = len(spread)

    for entry_z, exit_z in product(entry_grid, exit_grid):
        if exit_z >= entry_z:
            continue

        fold_sharpes = []
        for _, test_idx in purged_kfold_splits(n, n_splits, embargo):
            test_dates = spread.index[test_idx]
            if len(test_dates) < min_test_obs:
                continue
            bt = backtest_with_cost(
                spread.loc[test_dates[0]:test_dates[-1]],
                notional.loc[test_dates[0]:test_dates[-1]],
                entry_z=entry_z, exit_z=exit_z,
            )
            if len(bt) > 5:
                fold_sharpes.append(sharpe_ratio(bt["net_ret"]))

        if fold_sharpes:
            results.append({
                "entry_z": entry_z, "exit_z": exit_z,
                "mean_sharpe_cv": np.nanmean(fold_sharpes),
                "std_sharpe_cv": np.nanstd(fold_sharpes),
                "n_folds_valid": len(fold_sharpes),
            })

    return pd.DataFrame(results).sort_values("mean_sharpe_cv", ascending=False).reset_index(drop=True)


def walk_forward(
    price_data: pd.DataFrame,
    sym_y: str,
    sym_x: str,
    train_window: int = 60,
    test_window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Rolling walk-forward: refit static/dynamic hedge ratio mỗi `train_window`
    ngày, backtest `test_window` ngày tiếp theo — không chồng lấn.

    Static OLS refit HOÀN TOÀN mới mỗi window. Kalman filter tiếp tục (continue)
    state từ window trước — đúng bản chất online filter.
    """
    records = []
    n = len(price_data)
    kalman_mean, kalman_cov = None, None

    for start in range(train_window, n - test_window, test_window):
        train = price_data.iloc[start - train_window:start]
        test = price_data.iloc[start:start + test_window]
        y_tr, x_tr = train[sym_y], train[sym_x]
        y_te, x_te = test[sym_y], test[sym_x]

        # --- Static ---
        a_s, b_s = fit_static_hedge_ratio(y_tr, x_tr)
        sp_static_train = static_spread(y_tr, x_tr, a_s, b_s)
        sp_static_test = static_spread(y_te, x_te, a_s, b_s)
        sp_static_full = pd.concat([sp_static_train.tail(20), sp_static_test])
        notional_static_full = pd.concat([
            (y_tr + abs(b_s) * x_tr).tail(20), y_te + abs(b_s) * x_te,
        ])
        bt_static = backtest_with_cost(sp_static_full, notional_static_full, entry_z, exit_z)
        bt_static = bt_static.loc[bt_static.index.isin(test.index)]

        # --- Dynamic (continue Kalman state) ---
        sm_train, sc_train = fit_kalman_hedge_ratio(y_tr, x_tr, kalman_mean, kalman_cov)
        sp_dyn_train, beta_tr = dynamic_spread(y_tr, x_tr, sm_train)
        kalman_mean, kalman_cov = sm_train[-1], sc_train[-1]

        sm_test, sc_test = fit_kalman_hedge_ratio(y_te, x_te, kalman_mean, kalman_cov)
        sp_dyn_test, beta_te = dynamic_spread(y_te, x_te, sm_test)
        kalman_mean, kalman_cov = sm_test[-1], sc_test[-1]

        sp_dyn_full = pd.concat([sp_dyn_train.tail(20), sp_dyn_test])
        notional_dyn_full = pd.concat([
            (y_tr + beta_tr.abs() * x_tr).tail(20), y_te + beta_te.abs() * x_te,
        ])
        bt_dynamic = backtest_with_cost(sp_dyn_full, notional_dyn_full, entry_z, exit_z)
        bt_dynamic = bt_dynamic.loc[bt_dynamic.index.isin(test.index)]

        records.append({
            "window_start": test.index[0], "window_end": test.index[-1],
            "sharpe_static": sharpe_ratio(bt_static["net_ret"]) if len(bt_static) > 3 else np.nan,
            "sharpe_dynamic": sharpe_ratio(bt_dynamic["net_ret"]) if len(bt_dynamic) > 3 else np.nan,
            "pnl_static": bt_static["net_ret"].sum(),
            "pnl_dynamic": bt_dynamic["net_ret"].sum(),
        })

    return pd.DataFrame(records)
