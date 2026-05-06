"""方向图分析函数。

独立于 Pattern 类，以 (pattern, theta) 数据对为输入，
可复用于 Pattern 计算的结果或 HFSS 全波仿真的方向图。

支持：
  - 1D 方向图 (Nθ,)：线阵或平面阵单 φ 切片
  - 2D 方向图 (Nθ, Nφ)：平面阵全 φ 分析
"""

from typing import Optional, Union
import numpy as np


# ============================================================
#  1D 峰值搜索
# ============================================================

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


def _find_peaks_1d(pattern: np.ndarray,
                   theta: Optional[np.ndarray] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """1D 峰值搜索：找所有峰值，返回 (values, coords)。"""
    pattern = np.asarray(pattern)
    indices, values = _find_peaks(pattern)
    if theta is not None:
        return values, np.asarray(theta)[indices]
    return values, indices


def _get_psll_1d(pattern: np.ndarray,
                 theta: Optional[np.ndarray] = None,
                 mainlobe_region: Optional[tuple[float, float]] = None
                 ) -> tuple[float, Union[float, int]]:
    """1D PSLL 计算：返回 (psll_value, psll_coord)。"""
    values, coords = _find_peaks_1d(pattern, theta)

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


# ============================================================
#  2D 峰值搜索（平面阵）
# ============================================================

def _find_peaks_2d(pattern: np.ndarray,
                   theta: Optional[np.ndarray],
                   phi: Optional[np.ndarray]
                   ) -> tuple[np.ndarray, np.ndarray]:
    """2D 峰值搜索：遍历所有 φ 平面，合并全部峰值并全局排序。

    Returns:
        values: (N_total,)、coords: (N_total, 2) — 列 [θ, φ]
    """
    _, N_phi = pattern.shape
    all_values, all_coords = [], []

    for j in range(N_phi):
        indices, values = _find_peaks(pattern[:, j])
        if theta is not None:
            tc = np.asarray(theta, dtype=float)[indices]
        else:
            tc = indices.astype(float)
        if phi is not None:
            pc = np.full(len(indices), float(phi[j]))
        else:
            pc = np.full(len(indices), float(j))
        all_values.append(values)
        all_coords.append(np.column_stack([tc, pc]))

    if not all_values:
        return np.array([]), np.empty((0, 2))

    values = np.concatenate(all_values)
    coords = np.concatenate(all_coords, axis=0)
    order = np.argsort(values)[::-1]
    return values[order], coords[order]


def _get_psll_2d(pattern: np.ndarray,
                 theta: Optional[np.ndarray],
                 phi: Optional[np.ndarray],
                 mainlobe_region: Optional[tuple[float, float]]
                 ) -> tuple[np.ndarray, np.ndarray]:
    """2D PSLL：每 φ 平面一个 PSLL，返回 (pslls: (Nφ,), coords: (Nφ, 2))。"""
    _, N_phi = pattern.shape
    psll_values = np.full(N_phi, -np.inf)
    psll_coords = np.full((N_phi, 2), np.nan)

    for j in range(N_phi):
        val, coord = _get_psll_1d(pattern[:, j], theta, mainlobe_region)
        psll_values[j] = val

        valid = not np.isinf(val)
        if theta is not None:
            tc = float(coord) if valid else np.nan
        else:
            tc = float(coord) if valid else np.nan
        if phi is not None:
            pc = float(phi[j])
        else:
            pc = float(j) if valid else np.nan
        psll_coords[j] = [tc, pc]

    return psll_values, psll_coords


# ============================================================
#  公共 API（ndim 分派）
# ============================================================

def find_peaks(pattern: np.ndarray,
               theta: Optional[np.ndarray] = None,
               phi: Optional[np.ndarray] = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """查找方向图峰值，按值降序返回。

    1D 输入 (Nθ,)：
        线阵或单 φ 切片，返回 (values, coords)，coords 为角度或索引。

    2D 输入 (Nθ, Nφ)：
        平面阵方向图，遍历所有 φ 面合并峰值。
        coords 为 (N_total, 2) 数组，列为 [θ_coord, φ_coord]。

    Args:
        pattern: 1D 或 2D 方向图数组
        theta: θ 角度网格（度），shape (Nθ,)；不传则返回索引
        phi: φ 角度网格（度），shape (Nφ,)；pattern 为 2D 时可用

    Returns:
        (values, coords)
    """
    pattern = np.asarray(pattern)
    if pattern.ndim == 1:
        return _find_peaks_1d(pattern, theta)
    elif pattern.ndim == 2:
        return _find_peaks_2d(pattern, theta, phi)
    else:
        raise ValueError(f"pattern 必须是 1D 或 2D 数组, 实际 ndim={pattern.ndim}")


def get_psll(pattern: np.ndarray,
             theta: Optional[np.ndarray] = None,
             phi: Optional[np.ndarray] = None,
             mainlobe_region: Optional[tuple[float, float]] = None
             ) -> tuple:
    """计算最高副瓣电平（PSLL）。

    1D 输入 (Nθ,)：
        返回 (psll_value, psll_coord)，coord 为角度（传 theta）或索引。

    2D 输入 (Nθ, Nφ)：
        遍历所有 φ 平面，返回 (pslls: (Nφ,), coords: (Nφ, 2))。
        coords 列为 [θ_coord, φ_coord]，无副瓣的平面为 [nan, nan]。

    Args:
        pattern: 1D 或 2D 方向图数组（推荐归一化 dB）
        theta: θ 角度网格（度），shape (Nθ,)
        phi: φ 角度网格（度），shape (Nφ,)；pattern 为 2D 时可用
        mainlobe_region: (θ_start, θ_end) 主瓣角度范围（度）。
                         传入此参数时 theta 不能为 None。
                         2D 时对每个 φ 平面独立应用。

    Returns:
        1D: (psll: float, coord: float|int)
        2D: (pslls: ndarray(Nφ,), coords: ndarray(Nφ, 2))
    """
    if mainlobe_region is not None and theta is None:
        raise ValueError("使用 mainlobe_region 时必须提供 theta")

    pattern = np.asarray(pattern)
    if pattern.ndim == 1:
        return _get_psll_1d(pattern, theta, mainlobe_region)
    elif pattern.ndim == 2:
        return _get_psll_2d(pattern, theta, phi, mainlobe_region)
    else:
        raise ValueError(f"pattern 必须是 1D 或 2D 数组, 实际 ndim={pattern.ndim}")


# ============================================================
#  全平面 PSLL 便利函数
# ============================================================

def get_overall_psll(pattern: np.ndarray,
                     theta: Optional[np.ndarray] = None,
                     phi: Optional[np.ndarray] = None,
                     mainlobe_region: Optional[tuple[float, float]] = None
                     ) -> tuple[float, np.ndarray]:
    """全平面最大 PSLL —— 公式 (9) 的 f(X,Y) = max |AF(θ,φ)/FFmax|。

    1D 输入：等价于 get_psll()。
    2D 输入：调用 get_psll() 获取每 φ 平面 PSLL，返回全局最大值。

    Args:
        pattern: 1D 或 2D 方向图数组
        theta: θ 角度网格（度）
        phi: φ 角度网格（度）
        mainlobe_region: 主瓣排除区域

    Returns:
        (psll: float, coord: (2,) ndarray) — 全局最差副瓣及其 [θ, φ] 位置。
        无副瓣时返回 (-inf, [nan, nan])。
    """
    pslls, coords = get_psll(pattern, theta, phi, mainlobe_region)
    pattern = np.asarray(pattern)

    if pattern.ndim == 1:
        return (float(pslls), np.array([float(coords)]))

    # 2D: 取所有 φ 面中最差 PSLL
    valid = ~np.isinf(pslls)
    if not valid.any():
        return -np.inf, np.array([np.nan, np.nan])
    worst_idx = int(np.argmax(pslls))
    return float(pslls[worst_idx]), coords[worst_idx].copy()
