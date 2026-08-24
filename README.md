# Project 2 — Pairs Trading CTG-STB (Cointegration → Kalman → Cost-aware Backtest)

## Mục tiêu

Kiểm định giả thuyết: có tồn tại cặp cổ phiếu ngân hàng niêm yết Việt Nam
cointegrated đủ ổn định để trade mean-reversion, và dynamic hedge ratio
(Kalman filter) có thực sự cải thiện chiến lược so với static OLS hay không
— sau khi đã tính transaction cost và kiểm định qua walk-forward.

> ⚠️ **Kết luận ngắn gọn: KHÔNG.** Cặp được chọn (CTG-STB) không sống sót
> qua multiple-testing correction. Xem mục "Hạn chế & Rủi ro" trước khi dùng
> bất kỳ con số nào trong project này cho quyết định thực tế.

## Dataset

- Nguồn: `vnstock` (source `KBS`), tần suất ngày.
- Universe screening: 11 mã ngân hàng niêm yết (ACB, BID, CTG, HDB, LPB, MBB,
  STB, TCB, VCB, VIB, VPB).
- Cặp được chọn để trade: **CTG-STB**.
- In-sample: 2024-01-02 → 2025-12-30 (498 phiên).
- Out-of-sample: 2025-12-31 → 2026-07-30 (143 phiên).

## Phương pháp

```
Data (11 mã, vnstock)
  ↓
Engle-Granger cointegration test cho TẤT CẢ 55 cặp
  ↓
Multiple-testing correction (Bonferroni + BH/FDR)   ← CTG-STB KHÔNG qua được
  ↓
Hedge ratio: Static OLS  &  Dynamic Kalman filter
  ↓
Spread + ADF stationarity test + Half-life (AR1/OU)
  ↓
Z-score entry/exit (window=20, entry=1.5σ, exit=0.5σ)
  ↓
Backtest CÓ transaction cost (0.15%) + slippage (0.08%)
  ↓
Purged K-Fold CV (chọn entry_z/exit_z) + Walk-forward (29 window)
```

Chi tiết đầy đủ: [`reports/research_note.md`](reports/research_note.md).
Notebook gốc theo từng ngày: [`notebooks/`](notebooks/).

## Kết quả chính

| Metric | Static (OLS) | Dynamic (Kalman) |
|---|---|---|
| Sharpe Ratio (IS, **sau phí**) | 0.071 | 1.678 |
| Sharpe Ratio (OOS, **sau phí**) | 1.832 | 2.801 |
| Sharpe giảm do phí (IS) | −91% (0.771→0.071) | −49% (3.307→1.678) |
| Walk-forward Sharpe (mean±std, 29 window) | −0.06 ± 3.49 | 1.63 ± 1.65 |
| Walk-forward % window P&L dương | 48% | 86% |
| Max drawdown (walk-forward, đơn vị giá) | xem `reports/research_note.md` | xem `reports/research_note.md` |
| Win Rate | tính trong `src/risk.win_rate()` | tính trong `src/risk.win_rate()` |
| Cointegration p-value (raw / Bonferroni-adj) | 0.0041 / **0.2238** | — |

## Hạn chế & Rủi ro

- **CTG-STB không sống sót qua multiple-testing.** Đã test 55 cặp đồng thời;
  CTG-STB có p-value thấp nhất (0.0041) nhưng sau Bonferroni/BH-FDR, **0/55
  cặp** còn significant ở α=0.05. Toàn bộ phần còn lại của project (Kalman,
  backtest, walk-forward) chỉ mang tính minh họa phương pháp trên 1 cặp cụ
  thể, KHÔNG phải bằng chứng thống kê vững chắc rằng CTG-STB thực sự
  cointegrated.
- **Kết quả OOS dương có thể phản ánh Kalman "bắt regime" hơn là mean-reversion
  thật.** Beta động dịch chuyển có hệ thống từ ~0.46 lên vùng 0.58–0.65 trong
  giai đoạn quan sát — không loại trừ khả năng Kalman đang theo một xu hướng
  dài hạn (CTG outperform STB) thay vì dao động quanh 1 điểm cân bằng cố định
  như lý thuyết mean-reversion giả định. Xem mục 5 của `research_note.md`.
- **Sharpe sau phí giao dịch giảm mạnh, đặc biệt ở beta tĩnh** (−91%
  in-sample). Giả định phí 0.15% + slippage 0.08%/lượt round-trip — đây là
  ước lượng, không phải phí thực tế của một broker cụ thể; thanh khoản thấp
  hoặc lô lẻ có thể khiến slippage thực tế cao hơn.
- **Transaction cost là giả định**, chưa tính thuế, phí lưu ký, hoặc chênh
  lệch giá khớp lệnh thực tế (đặc biệt với biên độ giá trần/sàn của HOSE).
- **Chưa test out-of-time ngoài giai đoạn 2024–07/2026** — chưa biết chiến
  lược phản ứng thế nào khi regime lãi suất/thanh khoản thay đổi mạnh.
- **Dashboard "Terminal" (`dashboard/index.html`) dùng dữ liệu MINH HỌA** cho
  panel giá + tín hiệu (sinh bằng đúng tham số đã fit thật: α=8.7545,
  β=0.4595, λ=0.9395, spread std=1.244) vì môi trường build không truy cập
  được API vnstock. Trang `dashboard/backtest.html` dùng **số liệu THẬT**
  (walk-forward, cost, cointegration) trích trực tiếp từ output notebooks.

> **Lưu ý kỹ thuật:** Chart.js được nhúng sẵn trong `dashboard/vendor_chart.js`
> (không tải qua CDN) — dashboard chạy được **hoàn toàn offline**, không phụ
> thuộc internet để vẽ biểu đồ. Nếu vì lý do nào đó thư viện chart vẫn không
> load được, trang sẽ hiển thị cảnh báo ngay tại vị trí biểu đồ thay vì bị
> trắng trang — phần text/bảng số liệu luôn hiển thị đúng bất kể chart có vẽ
> được hay không.

## Sản phẩm (Product)

Ngoài pipeline nghiên cứu, project có 2 trang dashboard tĩnh (mở trực tiếp
bằng trình duyệt, không cần server):

| File | Nội dung |
|---|---|
| `dashboard/index.html` | Terminal: biểu đồ giá CTG/STB, spread + z-score với dải entry/exit, log tín hiệu vào/ra lệnh gần nhất, bảng tỷ trọng 2 chân lệnh theo hedge ratio hiện tại. Dữ liệu giá là **minh họa** (xem cảnh báo trong banner của trang). |
| `dashboard/backtest.html` | Kết quả backtest: Sharpe gross/net theo từng segment, walk-forward Sharpe & cumulative P&L qua 29 window, bảng cointegration + multiple-testing, và khối "Hạn chế & Rủi ro" hiển thị ngay trên trang. Dữ liệu ở đây là **số liệu thật**. |

## Cách chạy

### Cài đặt

```bash
pip install -r requirements.txt
```

### Cách nhanh nhất: `launcher.py` — tải data thật + build dashboard + tự mở trình duyệt

```bash
python launcher.py
```

Lệnh này sẽ: tải giá thật cho 11 mã ngân hàng qua **vnstock** (tự động fallback
sang **CafeF** — scrape trực tiếp — nếu vnstock lỗi hoặc bị rate-limit) → chạy
lại toàn bộ pipeline (cointegration screening, multiple-testing, static/dynamic
hedge ratio, backtest có phí, Purged K-Fold CV, walk-forward) → ghi kết quả
thật vào `dashboard/data/*.json` + `dashboard/embed_*.js` → tự mở
`dashboard/index.html` bằng trình duyệt mặc định.

Tùy chọn:

```bash
python launcher.py --source cafef            # ép dùng CafeF (chậm hơn, không phụ thuộc vnstock)
python launcher.py --pair CTG STB             # đổi cặp cổ phiếu cho trang Terminal (mặc định CTG STB)
python launcher.py --end 2026-08-24            # cập nhật đến ngày cụ thể (mặc định: hôm nay)
python launcher.py --no-open                   # chỉ tính toán, không tự mở trình duyệt
python launcher.py --help                      # xem đầy đủ tùy chọn
```

> ⚠️ Trước khi chạy `launcher.py` lần đầu, `dashboard/index.html` hiển thị dữ
> liệu **minh họa** (sinh bằng đúng tham số đã fit thật, xem banner trên
> trang) vì môi trường build gốc không có kết nối mạng ra ngoài. Chạy
> `launcher.py` một lần trên máy có internet là dashboard sẽ có data thật
> ngay lập tức, và các con số trên `dashboard/backtest.html` (hero, stat
> card, risk-box) cũng tự tính lại theo đúng data mới — không còn số cứng.

### Chạy lại pipeline nghiên cứu bằng notebook (nếu muốn xem từng bước)

```bash
# Mở notebooks theo thứ tự — mỗi notebook độc lập chạy được, tự tải data qua vnstock
jupyter notebook notebooks/day11_cointegration_multiple_testing.ipynb
jupyter notebook notebooks/day12_kalman_dynamic_hedge.ipynb
jupyter notebook notebooks/day13_cost_purgedcv_walkforward.ipynb
```

### Dùng lại code trong `src/` cho cặp/project khác

```python
from src.data_loader import load_prices, train_test_split_by_date
from src.features import screen_cointegrated_pairs, multiple_testing_correction, \
    fit_static_hedge_ratio, fit_kalman_hedge_ratio, static_spread, dynamic_spread, half_life
from src.backtest import backtest_with_cost, walk_forward, grid_search_entry_exit, sharpe_ratio

prices = load_prices(["CTG", "STB"], start="2024-01-01", end="2026-07-31")["close"]
is_df, oos_df = train_test_split_by_date(prices, cutoff="2025-12-31")

alpha, beta = fit_static_hedge_ratio(is_df["CTG"], is_df["STB"])
spread = static_spread(is_df["CTG"], is_df["STB"], alpha, beta)
```

### Chạy unit test

```bash
pytest tests/ -v
```

Test bao phủ: purged K-Fold không leak dữ liệu, backtest không dùng
look-ahead bias (position tại t chỉ dùng dữ liệu ≤ t), và cost model chỉ
tính phí khi vị thế thực sự thay đổi.

### Mở dashboard

```bash
open dashboard/index.html      # macOS
xdg-open dashboard/index.html  # Linux
# hoặc kéo-thả file .html vào trình duyệt bất kỳ
```

## Cấu trúc project

```
project2-pairs-trading-ctg-stb/
├── README.md
├── requirements.txt
├── launcher.py                     # ⭐ chạy 1 lệnh: tải data thật + build dashboard + mở trình duyệt
├── data/                            # (trống — dùng src/data_loader.py để tự tải)
├── notebooks/
│   ├── day11_cointegration_multiple_testing.ipynb
│   ├── day12_kalman_dynamic_hedge.ipynb
│   └── day13_cost_purgedcv_walkforward.ipynb
├── src/
│   ├── data_loader.py              # vnstock + fallback CafeF, rate-limit handling
│   ├── cafef_provider.py           # scraper CafeF độc lập (nguồn dự phòng)
│   ├── features.py                 # cointegration, static/dynamic hedge ratio, half-life
│   ├── strategy.py                 # z-score entry/exit logic
│   ├── backtest.py                 # cost model, purged K-Fold, walk-forward engine
│   └── risk.py                     # sharpe, drawdown, win rate, weight table
├── reports/
│   └── research_note.md            # research note đầy đủ, có Observation/Hypothesis/Evidence
├── tests/
│   ├── test_no_lookahead.py
│   ├── test_purged_kfold.py
│   └── test_cost_model.py
└── dashboard/
    ├── index.html                  # Terminal: giá + tín hiệu + tỷ trọng
    ├── backtest.html               # Kết quả backtest + risk disclosure (tính động từ JSON)
    ├── vendor_chart.js             # Chart.js nhúng sẵn — dashboard chạy offline, không cần CDN
    ├── embed_price_data.js         # do launcher.py sinh ra — KHÔNG sửa tay
    ├── embed_backtest_data.js      # do launcher.py sinh ra — KHÔNG sửa tay
    └── data/                       # JSON gốc — launcher.py ghi đè mỗi lần chạy
```
