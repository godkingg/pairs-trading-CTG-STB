#!/usr/bin/env python3
"""Launcher: tải dữ liệu THẬT (vnstock, tự fallback CafeF nếu lỗi), chạy lại
toàn bộ pipeline (cointegration screening → multiple testing → static/dynamic
hedge ratio → backtest có phí → Purged K-Fold CV → walk-forward), ghi kết quả
vào dashboard/data/*.json + dashboard/embed_*.js, rồi tự mở dashboard trên
trình duyệt.

Cách dùng:
    python launcher.py                                  # mặc định: auto (vnstock, fallback CafeF)
    python launcher.py --source cafef                    # ép dùng CafeF (chậm hơn, không phụ thuộc vnstock)
    python launcher.py --pair CTG STB --end 2026-08-24    # tùy chỉnh cặp / khoảng thời gian
    python launcher.py --no-open                          # chỉ tính toán, không tự mở trình duyệt

Chạy xong, dashboard/index.html và dashboard/backtest.html sẽ hiển thị dữ
liệu THẬT (không còn dữ liệu minh họa) — banner cảnh báo trên trang Terminal
cũng tự cập nhật để phản ánh đúng nguồn dữ liệu vừa tải.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import backtest_with_cost, grid_search_entry_exit, sharpe_ratio, walk_forward
from src.data_loader import BANK_UNIVERSE, load_prices_auto, train_test_split_by_date
from src.features import (
    adf_test, dynamic_spread, fit_kalman_hedge_ratio, fit_static_hedge_ratio,
    half_life, multiple_testing_correction, screen_cointegrated_pairs, static_spread,
)
from src.strategy import generate_positions, zscore

DASHBOARD_DIR = ROOT / "dashboard"
DATA_DIR = DASHBOARD_DIR / "data"

CAPITAL_VND = 500_000_000
ENTRY_Z, EXIT_Z, WINDOW = 1.5, 0.5, 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["auto", "vnstock", "cafef"], default="auto",
                   help="Nguồn dữ liệu. auto = thử vnstock trước, fallback CafeF nếu lỗi.")
    p.add_argument("--pair", nargs=2, default=["CTG", "STB"], metavar=("SYM_Y", "SYM_X"),
                   help="Cặp cổ phiếu dùng cho trang Terminal (mặc định CTG STB).")
    p.add_argument("--universe", nargs="+", default=BANK_UNIVERSE,
                   help="Universe dùng cho cointegration screening (mặc định 11 mã ngân hàng).")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--is-cutoff", default="2025-12-31", help="Mốc chia in-sample/out-of-sample.")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--no-open", action="store_true", help="Không tự mở trình duyệt sau khi xong.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sym_y, sym_x = args.pair
    universe = sorted(set(args.universe) | {sym_y, sym_x})

    print(f"{'='*70}\n1. TẢI DỮ LIỆU (source={args.source}) — universe {len(universe)} mã\n{'='*70}")
    prices, source_used = load_prices_auto(universe, args.start, args.end, source=args.source)
    price_df = prices["close"].dropna(how="all")
    if price_df.empty:
        sys.exit("❌ Không tải được dữ liệu nào — kiểm tra kết nối mạng hoặc thử --source cafef.")
    print(f"Nguồn dữ liệu đã dùng: {source_used}")
    print(f"Khoảng thời gian: {price_df.index.min().date()} → {price_df.index.max().date()} "
          f"({len(price_df)} phiên, {price_df.shape[1]}/{len(universe)} mã có dữ liệu)")

    print(f"\n{'='*70}\n2. COINTEGRATION SCREENING + MULTIPLE TESTING\n{'='*70}")
    coint_df = screen_cointegrated_pairs(price_df)
    if coint_df.empty:
        sys.exit("❌ Không đủ dữ liệu để test cointegration (quá ít phiên chung giữa các mã).")
    corrected_df = multiple_testing_correction(coint_df)
    print(corrected_df.head(5)[["pair", "p_value", "p_adj_bonferroni", "sig_bonferroni", "sig_bh"]]
          .to_string(index=False))

    print(f"\n{'='*70}\n3. HEDGE RATIO: STATIC (OLS) vs DYNAMIC (KALMAN) — {sym_y}-{sym_x}\n{'='*70}")
    if sym_y not in price_df.columns or sym_x not in price_df.columns:
        sys.exit(f"❌ Thiếu dữ liệu cho {sym_y} hoặc {sym_x} — kiểm tra lại --pair.")

    pair_prices = price_df[[sym_y, sym_x]].dropna()
    is_df, oos_df = train_test_split_by_date(pair_prices, args.is_cutoff)
    if len(is_df) < 60:
        sys.exit(f"❌ Chỉ có {len(is_df)} phiên in-sample (<60) — không đủ để fit hedge ratio đáng tin cậy. "
                  "Thử --start sớm hơn hoặc --is-cutoff muộn hơn.")
    if len(oos_df) < 10:
        print(f"⚠️  Chỉ có {len(oos_df)} phiên out-of-sample — kết quả OOS sẽ rất nhiễu.")

    y_is, x_is = is_df[sym_y], is_df[sym_x]
    alpha_s, beta_s = fit_static_hedge_ratio(y_is, x_is)
    spread_static_is = static_spread(y_is, x_is, alpha_s, beta_s)

    state_means_is, state_covs_is = fit_kalman_hedge_ratio(y_is, x_is)
    spread_dynamic_is, beta_dyn_is = dynamic_spread(y_is, x_is, state_means_is)

    if len(oos_df) > 0:
        y_oos, x_oos = oos_df[sym_y], oos_df[sym_x]
        spread_static_oos = static_spread(y_oos, x_oos, alpha_s, beta_s)
        state_means_oos, _ = fit_kalman_hedge_ratio(
            y_oos, x_oos, init_state_mean=state_means_is[-1], init_state_cov=state_covs_is[-1]
        )
        spread_dynamic_oos, beta_dyn_oos = dynamic_spread(y_oos, x_oos, state_means_oos)
    else:
        y_oos, x_oos = y_is.iloc[:0], x_is.iloc[:0]
        spread_static_oos, spread_dynamic_oos = spread_static_is.iloc[:0], spread_dynamic_is.iloc[:0]
        beta_dyn_oos = beta_dyn_is.iloc[:0]

    hl = half_life(spread_static_is)
    adf_static = adf_test(spread_static_is)
    beta_dyn_all = pd.concat([beta_dyn_is, beta_dyn_oos])
    print(f"Static β={beta_s:.4f}, α={alpha_s:.4f} | ADF spread p-value={adf_static['p_value']:.6f} | "
          f"Half-life={f'{hl:.2f}' if hl else 'N/A'} ngày")
    print(f"Dynamic β: mean={beta_dyn_all.mean():.4f}, min={beta_dyn_all.min():.4f}, "
          f"max={beta_dyn_all.max():.4f}")

    print(f"\n{'='*70}\n4. BACKTEST CÓ PHÍ — IN-SAMPLE vs OUT-OF-SAMPLE\n{'='*70}")
    cost_summary = build_cost_summary(
        y_is, x_is, y_oos, x_oos, beta_s,
        spread_static_is, spread_static_oos, spread_dynamic_is, spread_dynamic_oos,
        beta_dyn_is, beta_dyn_oos,
    )
    with open(DATA_DIR / "cost_summary.json", "w") as f:
        json.dump(cost_summary, f, indent=2)
    for row in cost_summary:
        print(f"{row['segment']:<14} Sharpe gross={row['sharpe_gross']:.3f}  "
              f"net={row['sharpe_net']:.3f}  trades={row['n_trades']}")

    print(f"\n{'='*70}\n5. PURGED K-FOLD CV — chọn entry_z / exit_z\n{'='*70}")
    notional_is_static = y_is + abs(beta_s) * x_is
    cv_df = grid_search_entry_exit(
        spread_static_is, notional_is_static,
        entry_grid=[1.0, 1.5, 2.0, 2.5], exit_grid=[0.25, 0.5, 0.75],
    )
    cv_df.to_json(DATA_DIR / "cv_grid_search.json", orient="records", indent=2)
    if len(cv_df):
        print(cv_df.head(5).to_string(index=False))
    else:
        print("⚠️  Không đủ dữ liệu để chạy Purged K-Fold CV (in-sample quá ngắn).")

    print(f"\n{'='*70}\n6. WALK-FORWARD (refit 60 ngày / test 20 ngày)\n{'='*70}")
    if len(pair_prices) >= 80:
        wf_df = walk_forward(pair_prices, sym_y, sym_x, entry_z=ENTRY_Z, exit_z=EXIT_Z)
        wf_df["cum_pnl_static"] = wf_df["pnl_static"].cumsum()
        wf_df["cum_pnl_dynamic"] = wf_df["pnl_dynamic"].cumsum()
        wf_out = wf_df.copy()
        wf_out["window_start"] = wf_out["window_start"].astype(str)
        wf_out["window_end"] = wf_out["window_end"].astype(str)
        wf_out.to_json(DATA_DIR / "walkforward_results.json", orient="records", indent=2)
        print(f"{len(wf_df)} window | Static mean Sharpe={wf_df.sharpe_static.mean():.3f} | "
              f"Dynamic mean Sharpe={wf_df.sharpe_dynamic.mean():.3f}")
    else:
        print(f"⚠️  Chỉ có {len(pair_prices)} phiên (<80) — bỏ qua walk-forward, giữ nguyên file cũ nếu có.")

    print(f"\n{'='*70}\n7. COINTEGRATION + MODEL STATS CHO DASHBOARD\n{'='*70}")
    write_cointegration_json(corrected_df, sym_y, sym_x, hl)
    write_model_stats(beta_s, beta_dyn_all, adf_static, hl)

    print(f"\n{'='*70}\n8. XUẤT DỮ LIỆU GIÁ + TÍN HIỆU THẬT CHO TRANG TERMINAL\n{'='*70}")
    write_price_signals_json(
        pair_prices, spread_static_is, spread_static_oos, beta_s,
        sym_y, sym_x, source_used, args.start, args.end,
    )

    print(f"\n{'='*70}\n9. NHÚNG JSON VÀO embed_*.js\n{'='*70}")
    regenerate_embed_js()
    print("Đã ghi dashboard/embed_price_data.js và dashboard/embed_backtest_data.js")

    print(f"\n{'='*70}\n✅ HOÀN TẤT — dashboard đã có dữ liệu thật (nguồn: {source_used})\n{'='*70}")
    index_path = (DASHBOARD_DIR / "index.html").resolve()
    backtest_path = (DASHBOARD_DIR / "backtest.html").resolve()
    if not args.no_open:
        print(f"🌐 Đang mở: {index_path}")
        webbrowser.open(f"file://{index_path}")
    else:
        print(f"Mở thủ công:\n  {index_path}\n  {backtest_path}")


def build_cost_summary(y_is, x_is, y_oos, x_oos, beta_s,
                        spread_static_is, spread_static_oos, spread_dynamic_is, spread_dynamic_oos,
                        beta_dyn_is, beta_dyn_oos) -> list[dict]:
    notional_is_static = y_is + abs(beta_s) * x_is
    notional_is_dynamic = y_is + beta_dyn_is.abs() * x_is

    segments = [("Static-IS", spread_static_is, notional_is_static, spread_static_is.index),
                ("Dynamic-IS", spread_dynamic_is, notional_is_dynamic, spread_dynamic_is.index)]

    if len(y_oos) > 0:
        notional_oos_static = y_oos + abs(beta_s) * x_oos
        notional_oos_dynamic = y_oos + beta_dyn_oos.abs() * x_oos
        spread_static_full = pd.concat([spread_static_is.tail(WINDOW), spread_static_oos])
        spread_dynamic_full = pd.concat([spread_dynamic_is.tail(WINDOW), spread_dynamic_oos])
        notional_static_full = pd.concat([notional_is_static.tail(WINDOW), notional_oos_static])
        notional_dynamic_full = pd.concat([notional_is_dynamic.tail(WINDOW), notional_oos_dynamic])
        segments += [
            ("Static-OOS", spread_static_full, notional_static_full, spread_static_oos.index),
            ("Dynamic-OOS", spread_dynamic_full, notional_dynamic_full, spread_dynamic_oos.index),
        ]

    rows = []
    for name, spread, notional, keep_index in segments:
        bt = backtest_with_cost(spread, notional, entry_z=ENTRY_Z, exit_z=EXIT_Z, window=WINDOW)
        bt = bt.loc[bt.index.isin(keep_index)]
        n_trades = int((bt["position"].diff().fillna(0) != 0).sum()) if len(bt) else 0
        rows.append({
            "segment": name,
            "sharpe_gross": float(sharpe_ratio(bt["gross_ret"])) if len(bt) else 0.0,
            "sharpe_net": float(sharpe_ratio(bt["net_ret"])) if len(bt) else 0.0,
            "total_cost": float(bt["cost"].sum()) if len(bt) else 0.0,
            "n_trades": n_trades,
            "pnl_gross": float(bt["gross_ret"].sum()) if len(bt) else 0.0,
            "pnl_net": float(bt["net_ret"].sum()) if len(bt) else 0.0,
        })
    return rows


def write_cointegration_json(corrected_df: pd.DataFrame, sym_y: str, sym_x: str, hl: float | None) -> None:
    pair_name = f"{sym_y}-{sym_x}"
    row = corrected_df[corrected_df["pair"].isin([pair_name, f"{sym_x}-{sym_y}"])]

    payload = {
        "n_pairs_tested": int(len(corrected_df)),
        "n_significant_raw": int((corrected_df["p_value"] < 0.05).sum()),
        "n_significant_bonferroni": int(corrected_df["sig_bonferroni"].sum()),
        "n_significant_bh": int(corrected_df["sig_bh"].sum()),
        "selected_pair": pair_name,
        "half_life_days": round(hl, 2) if hl else None,
        "top_pairs": json.loads(
            corrected_df.head(5)[
                ["pair", "p_value", "p_adj_bonferroni", "sig_bonferroni", "p_adj_bh", "sig_bh"]
            ].to_json(orient="records")
        ),
    }
    with open(DATA_DIR / "cointegration_screen.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_model_stats(beta_s: float, beta_dyn_all: pd.Series, adf_static: dict, hl: float | None) -> None:
    beta_mean = float(beta_dyn_all.mean())
    payload = {
        "beta_static": round(beta_s, 4),
        "beta_dynamic_mean": round(beta_mean, 4),
        "beta_dynamic_min": round(float(beta_dyn_all.min()), 4),
        "beta_dynamic_max": round(float(beta_dyn_all.max()), 4),
        "beta_dynamic_cv": round(float(beta_dyn_all.std() / abs(beta_mean)), 4) if beta_mean != 0 else None,
        "adf_static_pvalue": round(adf_static["p_value"], 6),
        "half_life_days": round(hl, 2) if hl else None,
    }
    with open(DATA_DIR / "model_stats.json", "w") as f:
        json.dump(payload, f, indent=2)


def write_price_signals_json(pair_prices: pd.DataFrame, spread_static_is: pd.Series, spread_static_oos: pd.Series,
                              beta_s: float, sym_y: str, sym_x: str, source_used: str, start: str, end: str) -> None:
    spread_full = pd.concat([spread_static_is, spread_static_oos])
    z_full = zscore(spread_full, window=WINDOW)
    position_full = generate_positions(spread_full, entry_z=ENTRY_Z, exit_z=EXIT_Z, window=WINDOW)
    position_full = position_full.reindex(pair_prices.index).fillna(0)

    change_points = position_full[position_full.diff().fillna(position_full.iloc[0] if len(position_full) else 0) != 0]
    markers = []
    for dt, pos_val in change_points.items():
        z_val = z_full.get(dt, np.nan)
        z_val = None if pd.isna(z_val) else round(float(z_val), 2)
        mtype = "long_entry" if pos_val == 1 else ("short_entry" if pos_val == -1 else "exit")
        markers.append({"date": dt.strftime("%Y-%m-%d"), "type": mtype, "z": z_val})

    latest_date = pair_prices.index[-1]
    latest_price_y = float(pair_prices[sym_y].iloc[-1])
    latest_price_x = float(pair_prices[sym_x].iloc[-1])
    latest_pos = int(position_full.iloc[-1]) if len(position_full) else 0
    latest_z = z_full.reindex(pair_prices.index).iloc[-1] if len(pair_prices) else np.nan
    latest_z = round(float(latest_z), 2) if pd.notna(latest_z) else None

    side_y = "MUA" if latest_pos == 1 else ("BÁN" if latest_pos == -1 else "-")
    side_x = "BÁN" if latest_pos == 1 else ("MUA" if latest_pos == -1 else "-")

    weight_table = {
        "as_of": latest_date.strftime("%Y-%m-%d"),
        "pair": f"{sym_y}-{sym_x}",
        "current_position": {1: "LONG SPREAD", -1: "SHORT SPREAD", 0: "FLAT"}[latest_pos],
        "current_zscore": latest_z,
        "hedge_ratio_beta": round(beta_s, 4),
        "capital_allocated_vnd": CAPITAL_VND,
        "legs": [
            {"symbol": sym_y, "side": side_y, "price": round(latest_price_y, 2),
             "weight_pct": round(100 / (1 + abs(beta_s)), 1) if latest_pos != 0 else 0},
            {"symbol": sym_x, "side": side_x, "price": round(latest_price_x, 2),
             "weight_pct": round(100 * abs(beta_s) / (1 + abs(beta_s)), 1) if latest_pos != 0 else 0},
        ],
    }

    series = []
    for dt in pair_prices.index:
        z_val = z_full.get(dt, np.nan)
        sp_val = spread_full.get(dt, np.nan)
        series.append({
            "date": dt.strftime("%Y-%m-%d"),
            "ctg": round(float(pair_prices[sym_y].loc[dt]), 2),
            "stb": round(float(pair_prices[sym_x].loc[dt]), 2),
            "spread": None if pd.isna(sp_val) else round(float(sp_val), 3),
            "zscore": None if pd.isna(z_val) else round(float(z_val), 3),
            "position": int(position_full.loc[dt]) if dt in position_full.index else 0,
        })

    payload = {
        "meta": {
            "note": f"Dữ liệu THẬT — nguồn: {source_used}, khoảng {start} → {end}, "
                    f"tải lúc {pd.Timestamp.now():%Y-%m-%d %H:%M}.",
            "pair": f"{sym_y}-{sym_x}",
            "entry_z": ENTRY_Z, "exit_z": EXIT_Z, "window": WINDOW,
        },
        "series": series,
        "markers": markers,
        "weight_table": weight_table,
    }
    with open(DATA_DIR / "price_signals_demo.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def regenerate_embed_js() -> None:
    def load(name: str):
        path = DATA_DIR / f"{name}.json"
        with open(path) as f:
            return json.load(f)

    price_data = load("price_signals_demo")
    coint_data = load("cointegration_screen")
    with open(DASHBOARD_DIR / "embed_price_data.js", "w") as f:
        f.write("const PRICE_DATA = " + json.dumps(price_data, ensure_ascii=False) + ";\n")
        f.write("const COINT_DATA = " + json.dumps(coint_data, ensure_ascii=False) + ";\n")

    wf_data = load("walkforward_results")
    cost_data = load("cost_summary")
    coint_data2 = load("cointegration_screen")
    cv_data = load("cv_grid_search")
    model_stats = load("model_stats") if (DATA_DIR / "model_stats.json").exists() else {}
    with open(DASHBOARD_DIR / "embed_backtest_data.js", "w") as f:
        f.write("const WF_DATA = " + json.dumps(wf_data, ensure_ascii=False) + ";\n")
        f.write("const COST_DATA = " + json.dumps(cost_data, ensure_ascii=False) + ";\n")
        f.write("const COINT_DATA2 = " + json.dumps(coint_data2, ensure_ascii=False) + ";\n")
        f.write("const CV_DATA = " + json.dumps(cv_data, ensure_ascii=False) + ";\n")
        f.write("const MODEL_STATS = " + json.dumps(model_stats, ensure_ascii=False) + ";\n")


if __name__ == "__main__":
    main()
