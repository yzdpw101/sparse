"""方向图分析函数。

独立于 Pattern 类，以 (pattern, theta) 数据对为输入，
可复用于 Pattern 计算的结果或 HFSS 全波仿真的方向图。
"""

from typing import Optional
import numpy as np


def _find_peaks(data: np.ndarray, consider_edges: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
    """找一维数组局部峰值，按值降序返回 (indices, values)。

    对应 C++ extrema1D(data, findMax=true)。
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=int), np.array([])
    if n == 1:
        return np.array([0]), np.array([data[0]])

    peak_indices = []
    for i in range(n):
        if i == 0:
            if consider_edges and data[0] >= data[1] and data[0] > data[1]:
                peak_indices.append(i)
        elif i == n - 1:
            if consider_edges and data[n - 1] >= data[n - 2] and data[n - 1] > data[n - 2]:
                peak_indices.append(i)
        else:
            if data[i] >= data[i - 1] and data[i] >= data[i + 1]:
                if data[i] > data[i - 1] or data[i] > data[i + 1]:
                    peak_indices.append(i)

    if not peak_indices:
        peak_indices = [int(np.argmax(data))]

    indices = np.array(peak_indices)
    values = data[indices]
    order = np.argsort(values)[::-1]
    return indices[order], values[order]


def find_peaks(pattern: np.ndarray,
               theta: Optional[np.ndarray] = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """查找一维方向图的所有峰值，按值降序返回。

    Args:
        pattern: 一维数组（dB 方向图或原始值均可）
        theta: 角度网格（度），长度与 pattern 相同。
               不传时返回索引而非角度。

    Returns:
        (values, coords) — 峰值值及对应坐标。
        theta 传入时为角度，否则为索引。
    """
    pattern = np.asarray(pattern)
    indices, values = _find_peaks(pattern)
    if theta is not None:
        return values, np.asarray(theta)[indices]
    return values, indices


def get_psll(pattern: np.ndarray,
             theta: Optional[np.ndarray] = None,
             mainlobe_region: Optional[tuple[float, float]] = None
             ) -> tuple[float, float]:
    """计算最高副瓣电平及其角度（或索引）。

    委托 find_peaks() 计算，取第二高峰值作为最高副瓣。

    Args:
        pattern: 一维数组，推荐传入归一化 dB 方向图
        theta: 角度网格（度），长度与 pattern 相同。
               不传时返回索引而非角度。
        mainlobe_region: (θ_start, θ_end) 主瓣角度范围（度）。
                         传入此参数时 theta 不能为 None。

    Returns:
        (psll_value, psll_coord) — 无副瓣时返回 (-inf, nan)。
        psll_coord 在 theta 传入时为角度，否则为索引。
    """
    if mainlobe_region is not None and theta is None:
        raise ValueError("使用 mainlobe_region 时必须提供 theta")

    values, coords = find_peaks(pattern, theta)

    if mainlobe_region is not None:
        t_start, t_end = mainlobe_region
        mask = (coords < t_start) | (coords > t_end)
        values = values[mask]
        coords = coords[mask]

    if len(values) < 2:
        return -np.inf, np.nan

    if theta is not None:
        return float(values[1]), float(coords[1])
    return float(values[1]), int(coords[1])
