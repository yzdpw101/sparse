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


def find_peaks(pattern: np.ndarray, theta: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """查找一维方向图的所有峰值，按值降序返回。

    Args:
        pattern: 一维数组（dB 方向图或原始值均可）
        theta: 角度网格（度），长度与 pattern 相同

    Returns:
        (values, angles) — 峰值值数组及对应角度，按值降序排列
    """
    pattern = np.asarray(pattern)
    theta = np.asarray(theta)
    indices, values = _find_peaks(pattern)
    return values, theta[indices]


def get_psll(pattern: np.ndarray, theta: np.ndarray,
             mainlobe_region: Optional[tuple[float, float]] = None
             ) -> tuple[float, float]:
    """计算最高副瓣电平及其角度。

    委托 find_peaks() 计算，取第二高峰值作为最高副瓣。

    Args:
        pattern: 一维数组，推荐传入归一化 dB 方向图
        theta: 角度网格（度），长度与 pattern 相同
        mainlobe_region: (θ_start, θ_end) 主瓣角度范围（度），可选

    Returns:
        (psll_value, psll_angle) — 无副瓣时返回 (-inf, nan)
    """
    values, angles = find_peaks(pattern, theta)

    if mainlobe_region is not None:
        t_start, t_end = mainlobe_region
        mask = (angles < t_start) | (angles > t_end)
        values = values[mask]
        angles = angles[mask]

    if len(values) < 2:
        return -np.inf, np.nan

    return float(values[1]), float(angles[1])
