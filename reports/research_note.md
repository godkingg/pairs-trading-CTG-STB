# Research Note — Pairs Trading CTG-STB

## 1. Câu hỏi nghiên cứu

Ngân hàng niêm yết Việt Nam có tồn tại cặp cổ phiếu cointegrated đủ ổn định để
áp dụng chiến lược mean-reversion (pairs trading), và nếu có, dynamic hedge
ratio (Kalman filter) có cải thiện được chiến lược so với static OLS hay
không — sau khi tính đủ transaction cost?

## 2. Dữ liệu

- Universe: 11 mã ngân hàng niêm yết (ACB, BID, CTG, HDB, LPB, MBB, STB, TCB,
  VCB, VIB, VPB), nguồn `vnstock` (source KBS), tần suất ngày.
- In-sample: 2024-01-02 → 2025-12-30 (498 phiên).
- Out-of-sample: 2025-12-31 → 2026-07-30 (143 phiên).

## 3. Phương pháp

```
Data (11 mã)
  ↓
Engle-Granger cointegration test — TẤT CẢ C(11,2)=55 cặp
  ↓
Multiple-testing correction (Bonferroni + BH/FDR)
  ↓
Chọn cặp p-value thấp nhất: CTG-STB
  ↓
Hedge ratio: (a) Static OLS   (b) Dynamic Kalman filter
  ↓
Spread = y − (α + β·x), ADF test kiểm tra stationary
  ↓
Half-life mean reversion (AR(1)/OU)
  ↓
Z-score entry/exit (window=20, entry=1.5σ, exit=0.5σ)
  ↓
Backtest CÓ transaction cost (0.15%) + slippage (0.08%) mỗi lần đổi vị thế
  ↓
Purged K-Fold CV (chọn entry_z/exit_z) + Walk-forward (29 window, refit 60d/test 20d)
```

## 4. Kết quả chính

### 4.1 Cointegration screening (Ngày 11)

| Metric | Giá trị |
|---|---|
| Số cặp test đồng thời | 55 |
| Cặp p-value thấp nhất | CTG-STB (p = 0.004069) |
| Significant (raw, α=0.05) | 5 / 55 |
| Significant (Bonferroni) | **0 / 55** |
| Significant (BH/FDR) | **0 / 55** |
| p-adj CTG-STB (Bonferroni / BH) | 0.2238 / 0.2238 |

**CTG-STB KHÔNG sống sót qua bất kỳ phép điều chỉnh multiple-testing nào.**
Tiếp tục nghiên cứu cặp này chỉ mang tính minh họa phương pháp, không phải
bằng chứng thống kê vững chắc về mối quan hệ cointegrated thật.

### 4.2 Static OLS vs Dynamic Kalman (Ngày 12)

| Metric | Static (OLS) | Dynamic (Kalman) |
|---|---|---|
| β (hedge ratio) | 0.4595 (cố định) | mean 0.6803, std 0.0568 (dao động 0.57–0.79) |
| ADF spread p-value | 0.000755 | 0.000000 |
| Sharpe (IS, chưa phí) | 0.873 | 3.304 |
| Sharpe (OOS, chưa phí) | 2.423 | 4.082 |

Beta động lệch khá xa khỏi beta tĩnh (CV=8.35%, "thay đổi nhẹ" theo ngưỡng đã
định), nhưng đủ lớn để nghi ngờ đây là do Kalman đang bám theo một xu hướng
dài hạn của thị trường hơn là mean-reversion thuần túy quanh 1 mức cân bằng
cố định (xem mục 5).

### 4.3 Sau transaction cost (Ngày 13)

| Segment | Sharpe gross | Sharpe net | Δ | N trades |
|---|---|---|---|---|
| Static-IS | 0.771 | **0.071** | −91% | 61 |
| Dynamic-IS | 3.307 | **1.678** | −49% | 92 |
| Static-OOS | 2.423 | **1.832** | −24% | 23 |
| Dynamic-OOS | 4.095 | **2.801** | −32% | 27 |

Giả định: phí 0.15% + slippage 0.08% mỗi lần đổi vị thế (round-trip), tính
trên notional ước lượng = giá_y + |β|·giá_x.

### 4.4 Purged K-Fold CV — chọn entry_z/exit_z

| entry_z | exit_z | mean Sharpe (CV, net phí) | std |
|---|---|---|---|
| **1.0** | **0.25** | **0.299** | 1.046 |
| 1.5 | 0.50 (chọn tay ở Ngày 11-12) | 0.196 | 1.604 |
| 1.5 | 0.75 | −0.019 | 1.612 |

Tham số "chọn tay" ban đầu (1.5 / 0.5) không phải tham số tối ưu theo CV —
chênh lệch không quá lớn, nhưng xác nhận rủi ro data snooping khi chọn tham
số bằng mắt trên cùng 1 tập dữ liệu.

### 4.5 Walk-forward (29 window, refit mỗi 60 ngày, test 20 ngày)

| Metric | Static | Dynamic (Kalman) |
|---|---|---|
| Sharpe mean ± std | −0.062 ± 3.485 | **1.626 ± 1.654** |
| % window P&L dương | 48% | **86%** |

Static không ổn định qua các regime (Sharpe đổi dấu liên tục, std rất cao).
Dynamic ổn định hơn rõ rệt về mặt thống kê mô tả, nhưng mẫu 29 window không
lớn và cùng dựa trên 1 cặp cổ phiếu duy nhất chưa qua được kiểm định ở 4.1.

## 5. Diễn giải & Rủi ro (Observation / Hypothesis / Evidence / Conclusion)

- **Observation**: Dynamic Kalman cho Sharpe cao hơn static ở mọi giai đoạn
  (IS, OOS, walk-forward), kể cả sau phí.
- **Hypothesis A (mean-reversion thật)**: Kalman theo dõi tốt hơn 1 mức cân
  bằng hedge ratio đang trôi chậm, giúp bắt tín hiệu mean-reversion chính xác
  hơn.
- **Hypothesis B (regime-following, không phải mean-reversion)**: Beta động
  dịch chuyển có hệ thống (0.46 → vùng 0.58–0.65) trong giai đoạn quan sát —
  có thể phản ánh Kalman đang "chạy theo" một xu hướng thị trường (ví dụ
  CTG outperform STB theo thời gian) chứ không phải dao động quanh 1 điểm cân
  bằng cố định như giả định của mô hình mean-reversion cổ điển.
- **Evidence hiện có KHÔNG đủ để phân biệt A và B.** Cần thêm: (i) test trên
  nhiều cặp khác đã qua được multiple-testing để xem pattern có lặp lại, (ii)
  kiểm tra beta có quay về vùng cũ hay tiếp tục trôi một chiều trong dữ liệu
  mới hơn.
- **Conclusion**: Chưa đủ cơ sở kết luận dynamic hedge ratio "tốt hơn" static
  cho pairs trading nói chung. Chỉ có thể kết luận: trên 1 cặp cổ phiếu duy
  nhất, chưa qua kiểm định cointegration chặt, dynamic Sharpe cao hơn static
  Sharpe kể cả sau phí — đây là quan sát cần replicate, không phải kết quả
  đã được validate.

## 6. Đề xuất bước tiếp theo

1. Mở rộng universe (thêm nhóm ngành khác ngoài ngân hàng) để có nhiều ứng
   viên cặp cointegrated hơn, tăng khả năng tìm được cặp sống sót qua BH/FDR.
2. Test lại toàn bộ pipeline (cointegration → multiple testing → Kalman →
   cost-aware backtest → walk-forward) trên cặp MỚI sống sót qua correction,
   thay vì tiếp tục dùng CTG-STB.
3. Nếu vẫn dùng CTG-STB cho mục đích demo/học tập, LUÔN gắn kèm cảnh báo
   "chưa qua kiểm định thống kê" ở mọi nơi hiển thị kết quả (đã áp dụng
   trong README và dashboard của project này).
