"""方向图分析函数。

独立于 Pattern 类，以 (pattern, theta) 数据对为输入，
可复用于 Pattern 计算的结果或 HFSS 全波仿真的方向图。

支持：
  - 1D 方向图 (Nθ,)：线阵或平面阵单 φ 切片
  - 2D 方向图 (Nθ, Nφ)：平面阵全 φ 分析
  - 方向性系数计算
  - 主瓣增益下降计算

mainlobe_region 约定：
  - 1D 模式: (θ_start, θ_end) — 排除该 θ 区间
  - 2D 模式: ((θ_start, θ_end), (φ_start, φ_end)) — 排除矩形区域
"""

from typing import Optional, Union, Callable
from scipy.integrate import trapezoid
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
                   theta: Optional[np.ndarray] = None,
                   mainlobe_region: Optional[tuple[float, float]] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """1D 峰值搜索：找所有峰值，可选排除主瓣区域。返回 (values, coords)。"""
    pattern = np.asarray(pattern)
    indices, values = _find_peaks(pattern)

    if theta is not None:
        coords = np.asarray(theta, dtype=float)[indices]
    else:
        coords = indices.copy()  # 保持整数类型

    # 排除主瓣区域
    if mainlobe_region is not None and len(values) > 0:
        t_start, t_end = mainlobe_region
        keep = (coords < t_start) | (coords > t_end)
        values = values[keep]
        coords = coords[keep]

    return values, coords


def _get_psll_1d(pattern: np.ndarray,
                 theta: Optional[np.ndarray] = None,
                 mainlobe_region: Optional[tuple[float, float]] = None
                 ) -> tuple[float, Union[float, int]]:
    """1D PSLL 计算：返回 (psll_value, psll_coord)。"""
    values, coords = _find_peaks_1d(pattern, theta, mainlobe_region)

    if len(values) < 2:
        return -np.inf, np.nan

    if theta is not None:
        return float(values[1]), float(coords[1])
    return float(values[1]), int(coords[1])


# ============================================================
#  2D 峰值搜索（平面阵）
# ============================================================

def _parse_mainlobe_region_2d(mainlobe_region):
    """解析 2D mainlobe_region，返回 (θ_bounds, φ_bounds)。

    1D 风格 (θ_s, θ_e) → θ_bounds=(θ_s, θ_e), φ_bounds=None
    2D 风格 ((θ_s, θ_e), (φ_s, φ_e)) → θ_bounds, φ_bounds
    """
    if mainlobe_region is None:
        return None, None
    a, b = mainlobe_region
    if isinstance(a, (int, float, np.floating)):
        return mainlobe_region, None
    return a, b


def _find_peaks_2d(pattern: np.ndarray,
                   theta: Optional[np.ndarray],
                   phi: Optional[np.ndarray],
                   mainlobe_region=None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """2D 峰值搜索：遍历所有 φ 平面，合并全部峰值并全局排序。

    Returns:
        values: (N_total,)、coords: (N_total, 2) — 列 [θ, φ]
    """
    _, N_phi = pattern.shape
    theta_bounds, phi_bounds = _parse_mainlobe_region_2d(mainlobe_region)
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

    # 排除 2D 主瓣区域
    if theta_bounds is not None and len(values) > 0:
        t0, t1 = theta_bounds
        if phi_bounds is not None:
            f0, f1 = phi_bounds
            keep = ~((coords[:, 0] >= t0) & (coords[:, 0] <= t1) &
                     (coords[:, 1] >= f0) & (coords[:, 1] <= f1))
        else:
            # 1D 风格：仅 θ 过滤，对所有 φ 面有效
            keep = (coords[:, 0] < t0) | (coords[:, 0] > t1)
        values = values[keep]
        coords = coords[keep]

    order = np.argsort(values)[::-1]
    return values[order], coords[order]


def _get_psll_2d(pattern: np.ndarray,
                 theta: Optional[np.ndarray],
                 phi: Optional[np.ndarray],
                 mainlobe_region=None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """2D PSLL：每 φ 平面一个 PSLL，返回 (pslls: (Nφ,), coords: (Nφ, 2))。"""
    _, N_phi = pattern.shape
    theta_bounds, phi_bounds = _parse_mainlobe_region_2d(mainlobe_region)

    psll_values = np.full(N_phi, -np.inf)
    psll_coords = np.full((N_phi, 2), np.nan)

    for j in range(N_phi):
        # 该 φ 平面是否需要应用 θ 过滤
        if theta_bounds is not None and phi_bounds is not None:
            phi_j = phi[j] if phi is not None else j
            if phi_bounds[0] <= phi_j <= phi_bounds[1]:
                plane_mr = theta_bounds
            else:
                plane_mr = None
        else:
            plane_mr = theta_bounds

        val, coord = _get_psll_1d(pattern[:, j], theta, plane_mr)
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
               phi: Optional[np.ndarray] = None,
               mainlobe_region=None
               ) -> tuple[np.ndarray, np.ndarray]:
    """查找方向图峰值，按值降序返回。

    1D 输入 (Nθ,)：
        线阵或单 φ 切片，返回 (values, coords)，coords 为角度或索引。
        mainlobe_region = (θ_start, θ_end) 排除该区间。

    2D 输入 (Nθ, Nφ)：
        平面阵方向图，遍历所有 φ 面合并峰值。
        coords 为 (N_total, 2) 数组，列为 [θ_coord, φ_coord]。
        mainlobe_region = ((θ_s, θ_e), (φ_s, φ_e)) 排除矩形区域，
        或仅 (θ_s, θ_e) 对所有 φ 面生效。

    Args:
        pattern: 1D 或 2D 方向图数组
        theta: θ 角度网格（度），shape (Nθ,)；不传则返回索引
        phi: φ 角度网格（度），shape (Nφ,)；pattern 为 2D 时可用
        mainlobe_region: 主瓣排除区域，1D=(θ_s,θ_e), 2D=((θ_s,θ_e),(φ_s,φ_e))

    Returns:
        (values, coords)
    """
    pattern = np.asarray(pattern)
    if pattern.ndim == 1:
        return _find_peaks_1d(pattern, theta, mainlobe_region)
    elif pattern.ndim == 2:
        return _find_peaks_2d(pattern, theta, phi, mainlobe_region)
    else:
        raise ValueError(f"pattern 必须是 1D 或 2D 数组, 实际 ndim={pattern.ndim}")


def get_psll(pattern: np.ndarray,
             theta: Optional[np.ndarray] = None,
             phi: Optional[np.ndarray] = None,
             mainlobe_region=None
             ) -> tuple:
    """计算最高副瓣电平（PSLL）。

    1D 输入 (Nθ,)：
        返回 (psll_value, psll_coord)，coord 为角度（传 theta）或索引。
        mainlobe_region = (θ_start, θ_end) 排除主瓣区域。

    2D 输入 (Nθ, Nφ)：
        遍历所有 φ 平面，返回 (pslls: (Nφ,), coords: (Nφ, 2))。
        coords 列为 [θ_coord, φ_coord]，无副瓣的平面为 [nan, nan]。
        mainlobe_region = ((θ_start, θ_end), (φ_start, φ_end)) 排除矩形区域。

    Args:
        pattern: 1D 或 2D 方向图数组（推荐归一化 dB）
        theta: θ 角度网格（度），shape (Nθ,)
        phi: φ 角度网格（度），shape (Nφ,)；pattern 为 2D 时可用
        mainlobe_region: 1D=(θ_s,θ_e), 2D=((θ_s,θ_e),(φ_s,φ_e))

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
                     mainlobe_region=None
                     ) -> tuple[float, np.ndarray]:
    """全平面最大 PSLL —— 公式 (9) 的 f(X,Y) = max |AF(θ,φ)/FFmax|。

    1D 输入：等价于 get_psll()。
    2D 输入：调用 get_psll() 获取每 φ 平面 PSLL，返回全局最大值。

    Args:
        pattern: 1D 或 2D 方向图数组
        theta: θ 角度网格（度）
        phi: φ 角度网格（度）
        mainlobe_region: 主瓣排除区域，格式同 get_psll。

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


# ============================================================
#  方向性系数
# ============================================================

def compute_directivity(pat, theta_params, phi_params,
                                field_type='field', atol=1e-9):
    """
    通用方向性系数计算，支持一维（单 phi 切面）或二维方向图。
    一维输入要求 theta 范围对称（start ≈ -stop），以便自动扩展 phi 面。
    """
    # ------------------ 生成角度向量 ------------------
    def make_angles(start, stop, step):
        vec = np.arange(start, stop + step / 2, step)
        vec = np.round(vec / atol) * atol
        return vec

    theta = make_angles(*theta_params)
    phi = make_angles(*phi_params)

    # 将 pat 转为二维，并确保维度匹配
    pat = np.asarray(pat, dtype=float)
    if pat.ndim == 1:
        # 视为 (len(theta), 1)
        if len(phi) != 1:
            raise ValueError("一维 pat 要求 phi 参数只生成一个角度（即 phi_params 长度为 1）")
        if len(theta) != len(pat):
            raise ValueError(f"一维 pat 长度 {len(pat)} 与 theta 长度 {len(theta)} 不匹配")
        pat = pat[:, np.newaxis]   # 变为二维列向量
    elif pat.ndim == 2:
        pass
    else:
        raise ValueError("pat 必须是一维或二维数组")

    # 对齐尺寸（处理浮点造成的多一个点）
    theta = theta[:pat.shape[0]]
    phi = phi[:pat.shape[1]]

    if pat.shape != (len(theta), len(phi)):
        raise ValueError(f"pat 形状 {pat.shape} 与 (theta={len(theta)}, phi={len(phi)}) 不匹配")

    # 范围检查
    if np.any(theta < -180 - atol) or np.any(theta > 180 + atol):
        raise ValueError("theta 存在超出 [-180,180] 的值")
    if np.any(phi < -atol) or np.any(phi > 360 + atol):
        raise ValueError("phi 存在超出 [0,360] 的值")

    # 360° 转为 0°
    phi = np.where(np.abs(phi - 360.0) < atol, 0.0, phi)

    # ------------------ 判断是否需要坐标转换 ------------------
    if np.all(theta >= -atol):
        # 如果是一维且 theta 全非负，无法计算有意义的方向性
        if pat.shape[1] == 1:
            raise ValueError(
                "一维方向图（单 phi 面）且 theta 无负值，无法计算全空间方向性。"
                "请提供对称的 theta 范围（如 -90 ~ 90）以自动扩展 phi 面，"
                "或直接提供完整的二维方向图。"
            )
        # 标准球坐标，直接使用
        theta_std = theta
        phi_std = phi
        pat_std = pat
    else:
        # 检查对称性（theta 有负值时必须对称）
        if abs(theta[0] + theta[-1]) > atol:
            raise ValueError("theta 有负值时，起始和终止必须互为相反数")

        # 标准 theta 网格（绝对值去重排序）
        theta_std = np.unique(np.round(np.abs(theta) / atol) * atol)
        theta_std.sort()

        # 标准 phi 网格：原 phi + 每个 phi+180° 产生的值，去重排序
        all_phi = set(phi)
        for ph in phi:
            all_phi.add(np.round((ph + 180.0) % 360.0 / atol) * atol)
        phi_std = np.sort(list(all_phi))

        # 初始化输出矩阵
        new_pat = np.full((len(theta_std), len(phi_std)), np.nan)

        # 映射数据
        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                val = pat[i, j]
                if th >= -atol:
                    t_std = abs(th)
                    p_std = ph
                else:
                    t_std = -th
                    p_std = np.round((ph + 180.0) % 360.0 / atol) * atol

                idx_t = np.argmin(np.abs(theta_std - t_std))
                idx_p = np.argmin(np.abs(phi_std - p_std))

                if np.isnan(new_pat[idx_t, idx_p]):
                    new_pat[idx_t, idx_p] = val
                else:
                    if not np.allclose(new_pat[idx_t, idx_p], val, atol=atol):
                        raise ValueError(
                            f"方向 (θ={theta_std[idx_t]}°, φ={phi_std[idx_p]}°) "
                            f"数据冲突: {new_pat[idx_t, idx_p]} vs {val}"
                        )

        # 天顶方向填充：θ=0 的整行用已有有效值填充
        zero_idx = np.where(np.abs(theta_std) < atol)[0]
        if len(zero_idx) > 0:
            idx0 = zero_idx[0]
            row = new_pat[idx0, :]
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                if not np.allclose(valid, valid[0], atol=atol):
                    raise ValueError("天顶 (θ=0°) 处不同 phi 的数据不一致")
                new_pat[idx0, :] = valid[0]

        pat_std = new_pat

    # ------------------ 方向性系数积分 ------------------
    if field_type == 'field':
        U = pat_std ** 2
    elif field_type == 'power':
        U = pat_std.copy()
    else:
        raise ValueError("field_type 必须是 'field' 或 'power'")

    U_max = np.max(U)
    theta_rad = np.deg2rad(theta_std)
    phi_rad = np.deg2rad(phi_std)

    # 闭合 phi 维度（补 360°=0° 点）
    if abs(phi_rad[-1] - 2 * np.pi) > atol:
        phi_rad = np.append(phi_rad, 2 * np.pi)
        U = np.concatenate([U, U[:, [0]]], axis=1)

    integrand = U * np.sin(theta_rad)[:, np.newaxis]
    int_theta = np.trapz(integrand, x=theta_rad, axis=0)
    P_rad = np.trapz(int_theta, x=phi_rad)

    D_linear = 4 * np.pi * U_max / P_rad
    D_dBi = 10 * np.log10(D_linear)

    return D_dBi, D_linear

def compute_gain_drop(pattern_opt_linear, pattern_ref_linear):
    """计算主瓣增益下降 (dB)。

    对比两个未归一化线性方向图的峰值增益差。

    Args:
        pattern_opt_linear: 优化后的未归一化线性方向图
        pattern_ref_linear: 参考 (均匀激励) 未归一化线性方向图

    Returns:
        (drop_db, ref_gain_db, opt_gain_db)
        drop_db: 增益下降值 (dB), 正数表示优化后降低
        ref_gain_db: 参考峰值增益 (dB, 未归一化)
        opt_gain_db: 优化后峰值增益 (dB, 未归一化)
    """
    pattern_ref_linear = np.asarray(pattern_ref_linear, dtype=float)
    pattern_opt_linear = np.asarray(pattern_opt_linear, dtype=float)

    def _peak_gain(pattern_linear):
        p2 = pattern_linear ** 2
        return float(10 * np.log10(np.max(p2) + 1e-30))

    ref_gain = _peak_gain(pattern_ref_linear)
    opt_gain = _peak_gain(pattern_opt_linear)
    drop = ref_gain - opt_gain

    return drop, ref_gain, opt_gain
