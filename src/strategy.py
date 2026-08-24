"""Z-score mean-reversion strategy cho spread của cặp pairs trading.

Logic entry/exit dùng xuyên suốt Ngày 11-13: không thay đổi qua các version,
chỉ thêm cost model ở backtest.py (Ngày 13).
"""
from __future__ import annotations

import pandas as pd


def zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score của spread. window=20 ngày là mặc định đã dùng xuyên suốt."""
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()


def generate_positions(
    spread: pd.Series,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    window: int = 20,
) -> pd.Series:
    """Sinh vị thế (+1 long spread / -1 short spread / 0 flat) theo z-score.

    Long spread khi z < -entry_z (kỳ vọng spread tăng về mean).
    Short spread khi z > +entry_z (kỳ vọng spread giảm về mean).
    Đóng vị thế khi |z| < exit_z.

    Long spread = mua y, bán beta*x. Short spread = ngược lại.
    """
    z = zscore(spread, window).dropna()
    position = pd.Series(0, index=z.index)
    pos = 0

    for i in range(len(z)):
        z_t = z.iloc[i]
        if pos == 0:
            if z_t < -entry_z:
                pos = 1
            elif z_t > entry_z:
                pos = -1
        elif abs(z_t) < exit_z:
            pos = 0
        position.iloc[i] = pos

    return position
