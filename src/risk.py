"""Risk metrics dùng cho research note và dashboard.

Không đưa lại toàn bộ VaR framework của Ngày 2 (Normal/GBM) — ở đây chỉ giữ
các hàm cần cho việc đánh giá 1 chiến lược pairs-trading cụ thể.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.mean() * periods_per_year


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def win_rate(returns: pd.Series) -> float:
    """Tỷ lệ % số ngày có return khác 0 mà return dương (chỉ tính ngày có giao dịch)."""
    active = returns[returns != 0]
    if len(active) == 0:
        return float("nan")
    return (active > 0).mean()


def hedge_ratio_position_table(
    beta: float, capital: float, price_y: float, price_x: float, direction: int,
) -> dict:
    """Bảng tỷ trọng cho 1 lệnh pairs trade tại thời điểm hiện tại.

    direction: +1 = long spread (mua y, bán beta*x), -1 = short spread (ngược lại).
    capital: vốn phân bổ cho leg y (VND); leg x tính theo beta để giữ hedge ratio.
    """
    qty_y = direction * capital / price_y
    qty_x = -direction * beta * capital / price_x
    return {
        "leg_y_qty": qty_y, "leg_y_notional": qty_y * price_y,
        "leg_x_qty": qty_x, "leg_x_notional": qty_x * price_x,
        "net_notional": qty_y * price_y + qty_x * price_x,
        "beta_hedge_ratio": beta,
    }
