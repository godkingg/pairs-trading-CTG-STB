"""CafeF price-history scraper — dùng làm nguồn dữ liệu DỰ PHÒNG khi vnstock
lỗi hoặc bị rate-limit. Chậm hơn vnstock (scrape trực tiếp, phân trang theo
chunk 3 tháng) nhưng không phụ thuộc API bên thứ ba nào khác ngoài CafeF.

Không sửa logic — giữ nguyên như bản gốc, chỉ thêm docstring module.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

CAFEF_URL = (
    "https://cafef.vn/du-lieu/"
    "Ajax/PageNew/DataHistory/PriceHistory.ashx"
)

CAFEF_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://cafef.vn/du-lieu/lich-su-giao-dich-fpt-1.chn",
}

MAX_RETRIES    = 3
RETRY_BACKOFF  = 2.0   # seconds; doubled mỗi lần retry

EXCHANGE_ORDER = ("HOSE", "HNX", "UPCOM")

REQUEST_TIMEOUT = 15
PAGE_SIZE = 20
MAX_PAGES = 10

# CafeF trả ổn định ~65 phiên/chunk; 3 tháng là ngưỡng an toàn
CHUNK_MONTHS = 3

SUPPORTED_INTERVALS = {
    "1D": None,
    "1W": "W",
    "1M": "ME",
    "1Y": "YE",
}

_EXCHANGE_CACHE: dict[str, str] = {}


class SymbolNotFound(ValueError):
    pass


class CafeFPriceHistoryProvider:

    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        interval: str = "1D",
    ) -> pd.DataFrame:
        rule = self._validate_interval(interval)
        exchange = self._find_exchange(symbol)

        if start and end:
            rows = self._fetch_by_chunks(symbol=symbol, exchange=exchange, start=start, end=end)
        else:
            rows = self._fetch_pages(symbol=symbol, exchange=exchange)

        df = self._to_dataframe(rows)

        if start:
            df = df[df.index >= self._parse_date(start)]
        if end:
            df = df[df.index <= self._parse_date(end)]
        if limit:
            df = df.tail(limit)
        if rule:
            return self._resample(df, rule)

        return df

    def _fetch_by_chunks(
        self,
        *,
        symbol: str,
        exchange: str,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        start_date = self._parse_date(start)
        end_date = self._parse_date(end)
        all_rows: list[dict[str, Any]] = []
        current = start_date

        while current <= end_date:
            chunk_end = current + relativedelta(months=CHUNK_MONTHS) - relativedelta(days=1)
            if chunk_end > end_date:
                chunk_end = end_date

            logger.info("Fetching %s: %s -> %s", symbol, current.date(), chunk_end.date())

            rows = self._fetch_pages(
                symbol=symbol,
                exchange=exchange,
                start=current.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
            )
            all_rows.extend(rows)
            current = chunk_end + relativedelta(days=1)

        return all_rows

    def _fetch_pages(
        self,
        *,
        symbol: str,
        exchange: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for page in range(1, MAX_PAGES + 1):
            page_rows = self._request_page(
                symbol=symbol,
                exchange=exchange,
                page_index=page,
                start=start,
                end=end,
            )
            logger.debug("CafeF %s %s: page=%d, rows=%d", symbol, exchange, page, len(page_rows))

            if not page_rows:
                break
            rows.extend(page_rows)

        return rows

    def _find_exchange(self, symbol: str) -> str:
        symbol = symbol.upper()

        if symbol in _EXCHANGE_CACHE:
            return _EXCHANGE_CACHE[symbol]

        for exchange in EXCHANGE_ORDER:
            rows = self._request_page(symbol=symbol, exchange=exchange, page_index=1)
            if rows:
                _EXCHANGE_CACHE[symbol] = exchange
                logger.info("%s found on %s", symbol, exchange)
                return exchange

        raise SymbolNotFound(f"Không tìm thấy dữ liệu cho {symbol}")

    def _request_page(
        self,
        *,
        symbol: str,
        exchange: str,
        page_index: int,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "ExchangeType": exchange,
            "Symbol":       symbol.upper(),
            "PageIndex":    page_index,
            "PageSize":     PAGE_SIZE,
        }
        if start:
            params["StartDate"] = self._format_date(start)
        if end:
            params["EndDate"] = self._format_date(end)

        logger.debug("GET %s params=%s", CAFEF_URL, params)

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    CAFEF_URL,
                    params=params,
                    headers=CAFEF_HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                break
            except requests.exceptions.Timeout as exc:
                last_exc = exc
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Timeout %s/%s (params=%s) — retry in %.0fs",
                    attempt, MAX_RETRIES, params, wait,
                )
                time.sleep(wait)
        else:
            raise RuntimeError(
                f"CafeF timeout after {MAX_RETRIES} retries: {last_exc}"
            )

        if response.status_code != 200:
            raise RuntimeError(f"CafeF status {response.status_code}")

        data = response.json().get("Data", {})
        if not isinstance(data, dict):
            return []

        rows = data.get("Data")
        return rows if isinstance(rows, list) else []

    def _to_dataframe(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        records = []

        for row in rows:
            try:
                records.append({
                    "Date":      self._parse_date(row["Ngay"]),
                    "Open":      float(row.get("GiaMoCua", 0)),
                    "High":      float(row.get("GiaCaoNhat", 0)),
                    "Low":       float(row.get("GiaThapNhat", 0)),
                    "Close":     float(row.get("GiaDongCua", 0)),
                    "Adj Close": float(row.get("GiaDieuChinh", 0)),
                    "Volume":    int(row.get("KhoiLuongKhopLenh", 0)),
                })
            except Exception:
                continue

        if not records:
            return self._empty_frame()

        df = (
            pd.DataFrame(records)
            .drop_duplicates(subset=["Date"])
            .set_index("Date")
            .sort_index()
        )
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        result = (
            df.resample(rule)
            .agg({
                "Open":      "first",
                "High":      "max",
                "Low":       "min",
                "Close":     "last",
                "Adj Close": "last",
                "Volume":    "sum",
            })
            .dropna()
        )
        result["Volume"] = result["Volume"].astype("int64")
        return result

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            index=pd.DatetimeIndex([], name="Date"),
        )

    def _parse_date(self, value: Any) -> pd.Timestamp:
        if isinstance(value, pd.Timestamp):
            return value
        text = str(value)
        # Input nội bộ luôn là "YYYY-MM-DD"; response CafeF trả "DD/MM/YYYY"
        return pd.Timestamp(text) if "-" in text else pd.to_datetime(text, format="%d/%m/%Y")

    def _format_date(self, value: str) -> str:
        # CafeF nhận MM/DD/YYYY — chứng minh: EndDate=12/31/2025 không thể là %d/%m/%Y
        return self._parse_date(value).strftime("%m/%d/%Y")

    def _validate_interval(self, interval: str) -> str | None:
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(SUPPORTED_INTERVALS)}")
        return SUPPORTED_INTERVALS[interval]


class PriceHistory:
    """Thin wrapper quanh CafeFPriceHistoryProvider, giữ context symbol."""

    def __init__(self, symbol: str, provider: CafeFPriceHistoryProvider | None = None):
        self.symbol = symbol.upper()
        self._provider = provider or CafeFPriceHistoryProvider()

    def history(
        self,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
        interval: str = "1D",
    ) -> pd.DataFrame:
        return self._provider.fetch(
            symbol=self.symbol,
            start=start,
            end=end,
            limit=limit,
            interval=interval,
        )


class Stock(PriceHistory):
    """
    Entry-point chính cho người dùng thư viện.

    Example:
        stock = Stock("FPT")
        df = stock.history(start="2020-01-01", end="2025-12-31")
    """
    pass
