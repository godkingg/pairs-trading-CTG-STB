"""Load dữ liệu giá cổ phiếu từ vnstock, có rate-limit handling.

Nguồn: vnstock (source="KBS"). Áp dụng cho universe ngân hàng niêm yết dùng
trong Ngày 11-13 (cointegration screening + pairs trading CTG-STB).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

BANK_UNIVERSE = [
    "ACB", "BID", "CTG", "HDB", "LPB",
    "MBB", "STB", "TCB", "VCB", "VIB", "VPB",
]

RATE_LIMIT_BATCH = 18   # số request tối đa trước khi nghỉ
RATE_LIMIT_SLEEP = 65   # giây nghỉ để tránh chạm limit 20/phút


def load_prices(
    symbols: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = ("close",),
) -> dict[str, pd.DataFrame]:
    """Tải giá lịch sử qua vnstock (source KBS), tự nghỉ giữa các batch để tránh rate limit.

    Trả về dict {field: DataFrame[date x symbol]}.
    """
    from vnstock import Quote  # import trễ — chỉ cần khi thực sự dùng vnstock

    raw: dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols, 1):
        if i > 1 and (i - 1) % RATE_LIMIT_BATCH == 0:
            print(f"⏳ Nghỉ {RATE_LIMIT_SLEEP}s để tránh rate limit...")
            time.sleep(RATE_LIMIT_SLEEP)

        print(f"[{i}/{len(symbols)}] Loading {sym} (vnstock)...")
        df = Quote(symbol=sym, source="KBS").history(start=start, end=end, interval="d")
        raw[sym] = df.set_index("time")[list(fields)]

    print(f"\n✅ vnstock: đã tải {len(raw)}/{len(symbols)} mã")

    return {
        field: pd.DataFrame({sym: d[field] for sym, d in raw.items()}).sort_index()
        for field in fields
    }


def load_prices_cafef(
    symbols: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = ("close",),
) -> dict[str, pd.DataFrame]:
    """Tải giá lịch sử qua CafeF (scrape trực tiếp) — dùng khi vnstock lỗi/rate-limit.

    Chậm hơn vnstock (mỗi mã phân trang theo chunk 3 tháng) nhưng độc lập với
    vnstock. Trả về cùng format với load_prices().
    """
    from src.cafef_provider import Stock

    field_map = {"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}
    raw: dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Loading {sym} (CafeF)...")
        df = Stock(sym).history(start=start, end=end)
        cols = [field_map[f] for f in fields]
        raw[sym] = df[cols].rename(columns={v: k for k, v in field_map.items()})

    print(f"\n✅ CafeF: đã tải {len(raw)}/{len(symbols)} mã")

    return {
        field: pd.DataFrame({sym: d[field] for sym, d in raw.items()}).sort_index()
        for field in fields
    }


def load_prices_auto(
    symbols: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = ("close",),
    source: str = "auto",
) -> tuple[dict[str, pd.DataFrame], str]:
    """Tải giá, tự fallback CafeF nếu vnstock lỗi (source="auto"), hoặc ép nguồn cụ thể.

    source: "auto" | "vnstock" | "cafef"
    Trả về (data, source_used) — source_used để ghi vào metadata dashboard.
    """
    if source == "cafef":
        return load_prices_cafef(symbols, start, end, fields), "cafef"

    if source == "vnstock":
        return load_prices(symbols, start, end, fields), "vnstock"

    # auto: thử vnstock trước, fallback CafeF nếu lỗi bất kỳ (mạng, rate limit, mã không tồn tại...)
    try:
        return load_prices(symbols, start, end, fields), "vnstock"
    except Exception as e:
        print(f"⚠️  vnstock lỗi ({type(e).__name__}: {e}) — chuyển sang CafeF (chậm hơn)...")
        return load_prices_cafef(symbols, start, end, fields), "cafef"


def log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Log-return, drop NaN dòng đầu."""
    return np.log(price_df / price_df.shift(1)).dropna()


def train_test_split_by_date(price_df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia in-sample / out-of-sample theo mốc thời gian (không shuffle, giữ thứ tự)."""
    is_df = price_df[price_df.index <= cutoff]
    oos_df = price_df[price_df.index > cutoff]
    return is_df, oos_df
