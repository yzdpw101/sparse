"""空间映射公共基础: 响应误差计算、细模型 PSLL。"""

import numpy as np

TWO_PI = 2 * np.pi


def calc_response_error(Yc_list, Yf_list, ignore_db=-45.0):
    """计算粗/细模型归一化 dB 方向图之间的误差。

    C++ calcError: e = weighted MSE + HPBW mismatch + pointing mismatch.
    简化版: 仅 MSE + HPBW，避免过度耦合导致 Broyden 不稳定。

    Args:
        Yc_list: 粗模型响应列表 [pattern_1d, ...]
        Yf_list: 细模型响应列表 (HFSS CSV dB 值)
        ignore_db: 低于此值的采样点忽略

    Returns:
        float: 总误差
    """
    total = 0.0
    for Yc, Yf in zip(Yc_list, Yf_list):
        # 归一化 dB
        yc_db = 20.0 * np.log10(np.maximum(Yc, 1e-30) / np.max(Yc))
        yf_db = Yf.copy()

        # 细模型: 10*log10 (功率→dB), 归一化
        yf_max = np.max(yf_db)
        if yf_max > 0:
            yf_db = 10.0 * np.log10(np.maximum(yf_db, 1e-30) / yf_max)

        # 仅比较有效区域
        mask = yc_db > ignore_db
        if mask.any():
            diff = yc_db[mask] - yf_db[mask]
            total += np.mean(diff * diff)

        # HPBW 差 (粗)
        peak_idx = int(np.argmax(yc_db))
        hp = -3.0
        left = np.argmin(np.abs(yc_db[:peak_idx] - hp)) if peak_idx > 0 else 0
        right = peak_idx + np.argmin(np.abs(yc_db[peak_idx:] - hp))
        hpbw_c = right - left

        peak_idx_f = int(np.argmax(yf_db))
        left_f = np.argmin(np.abs(yf_db[:peak_idx_f] - hp)) if peak_idx_f > 0 else 0
        right_f = peak_idx_f + np.argmin(np.abs(yf_db[peak_idx_f:] - hp))
        hpbw_f = right_f - left_f

        total += abs(hpbw_c - hpbw_f) * 0.5

    return total


def compute_fine_psll(Yf_list):
    """计算细模型 (HFSS) PSLL: 10*log10 归一化后取副瓣峰值。"""
    from ..analysis import _find_peaks
    worst = -np.inf
    for yf in Yf_list:
        yf_max = np.max(yf)
        if yf_max <= 0:
            continue
        db = 10.0 * np.log10(np.maximum(yf, 1e-30) / yf_max)
        _, extrema = _find_peaks(db)
        if len(extrema) >= 2:
            worst = max(worst, extrema[1])
    return float(worst)
