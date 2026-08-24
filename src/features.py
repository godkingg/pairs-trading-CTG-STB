"""Feature engineering cho pairs trading: cointegration screening, hedge ratio
(static OLS + dynamic Kalman), spread, half-life, multiple-testing correction.

Nguồn logic: Ngày 11 (cointegration + multiple testing), Ngày 12 (Kalman dynamic beta).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

DEFAULT_KALMAN_DELTA = 1e-4  # tốc độ "trôi" của beta — giá trị chuẩn trong pairs-trading literature


def screen_cointegrated_pairs(price_df: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Engle-Granger cointegration test cho MỌI cặp trong universe.

    Trả về DataFrame sort theo p-value tăng dần. Quan trọng: KHÔNG tự kết luận
    "cặp p-value thấp nhất = cointegrated" — phải chạy multiple_testing_correction()
    trước khi diễn giải, vì p-value riêng lẻ bị lạc quan khi test nhiều cặp cùng lúc.
    """
    records = []
    for sym1, sym2 in combinations(price_df.columns, 2):
        pair_df = price_df[[sym1, sym2]].dropna()
        if len(pair_df) < min_obs:
            continue
        t_stat, p_value, _ = coint(pair_df[sym1], pair_df[sym2])
        records.append({
            "pair": f"{sym1}-{sym2}", "sym1": sym1, "sym2": sym2,
            "t_stat": t_stat, "p_value": p_value, "n_obs": len(pair_df),
        })
    return pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)


def multiple_testing_correction(coint_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni + Benjamini-Hochberg (FDR) correction cho bảng cointegration.

    Bắt buộc chạy bước này trước khi chọn cặp để trade — p-value "thấp nhất trong N
    phép test" có thể chỉ là false positive do data snooping.
    """
    df = coint_df.copy()
    reject_bonf, p_adj_bonf, _, _ = multipletests(df["p_value"].values, alpha=alpha, method="bonferroni")
    reject_bh, p_adj_bh, _, _ = multipletests(df["p_value"].values, alpha=alpha, method="fdr_bh")

    df["p_adj_bonferroni"] = p_adj_bonf
    df["sig_bonferroni"] = reject_bonf
    df["p_adj_bh"] = p_adj_bh
    df["sig_bh"] = reject_bh
    return df


def fit_static_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS: y = alpha + beta * x + spread. Trả về (alpha, beta)."""
    result = sm.OLS(y, sm.add_constant(x)).fit()
    return result.params.iloc[0], result.params.iloc[1]


def static_spread(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    return y - (alpha + beta * x)


def fit_kalman_hedge_ratio(
    y: pd.Series, x: pd.Series,
    init_state_mean: np.ndarray | None = None,
    init_state_cov: np.ndarray | None = None,
    delta: float = DEFAULT_KALMAN_DELTA,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman filter cho dynamic hedge ratio [beta_t, alpha_t].

    Có thể "continue" từ state trước đó (init_state_mean/cov) — dùng cho
    walk-forward hoặc chuyển từ in-sample sang out-of-sample mà không refit từ đầu.
    """
    obs_mat = np.vstack([x.values, np.ones(len(x))]).T[:, np.newaxis, :]
    trans_cov = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov,
        initial_state_mean=init_state_mean if init_state_mean is not None else [0, 0],
        initial_state_covariance=init_state_cov if init_state_cov is not None else np.ones((2, 2)),
    )
    return kf.filter(y.values)


def dynamic_spread(y: pd.Series, x: pd.Series, state_means: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Trả về (spread_t, beta_t) từ Kalman state_means."""
    beta_t = pd.Series(state_means[:, 0], index=y.index, name="beta_t")
    alpha_t = pd.Series(state_means[:, 1], index=y.index, name="alpha_t")
    return y - (alpha_t + beta_t * x), beta_t


def adf_test(series: pd.Series) -> dict:
    """ADF test, trả về dict gọn để log/so sánh."""
    stat, p_value, _, _, crit, *_ = adfuller(series)
    return {"adf_stat": stat, "p_value": p_value, "critical_values": crit, "stationary": p_value < 0.05}


def half_life(spread: pd.Series) -> float | None:
    """Half-life mean reversion qua AR(1)/Ornstein-Uhlenbeck fit trên spread.

    half_life = -ln(2) / ln(lambda), lambda là hệ số AR(1).
    Trả về None nếu lambda ngoài khoảng (0, 1) — nghĩa là spread không
    mean-reverting theo dạng OU (random walk, explosive, hoặc oscillating).
    """
    spread_lag = spread.shift(1).dropna()
    spread_curr = spread.loc[spread_lag.index]
    result = sm.OLS(spread_curr, sm.add_constant(spread_lag)).fit()
    lam = result.params.iloc[1]

    if not (0 < lam < 1):
        return None
    return -np.log(2) / np.log(lam)
