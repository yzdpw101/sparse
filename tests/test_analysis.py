"""测试方向图分析函数。"""

import numpy as np
from antopt.pattern import Pattern
from antopt.analysis import get_psll, find_peaks, get_overall_psll


def test_get_psll_broadside():
    """验证均匀 10 元半波长直线阵法向 PSLL ≈ -13.26 dB。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.to_dB(pat.linear_af(positions))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert abs(psll_val - (-13.26)) < 0.5, f"PSLL 应接近 -13.26 dB, 实际 {psll_val:.2f} dB"
    assert isinstance(psll_angle, float)
    assert abs(psll_angle) > 0, "副瓣不应在法向"


def test_get_psll_single_element():
    """单阵元无副瓣，PSLL 应返回 (-inf, nan)。"""
    pat = Pattern()
    af_db = Pattern.to_dB(pat.linear_af(np.array([0.0])))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert psll_val == -np.inf
    assert np.isnan(psll_angle)


def test_get_psll_module_level():
    """验证模块级导入正常。"""
    from antopt import get_psll
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert isinstance(psll_val, float)
    assert isinstance(psll_angle, float)
    assert psll_val < 0


def test_get_psll_with_mainlobe_region():
    """指定主瓣区域排除后计算副瓣。"""
    pat = Pattern(theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.to_dB(pat.linear_af(positions))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg, mainlobe_region=(-10, 10))
    assert psll_val < 0, "副瓣电平应为负值"
    assert isinstance(psll_angle, float)


def test_find_peaks_broadside():
    """验证 find_peaks 能找到均匀线阵的峰值。"""
    pat = Pattern(theta_deg_step=1.0)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.to_dB(pat.linear_af(positions))
    values, angles = find_peaks(af_db, pat.theta_deg)
    assert len(values) >= 2, "均匀线阵应有多个峰值"
    assert abs(values[0]) < 1e-6, "最高峰值应为 0 dB（主瓣）"


def test_find_peaks_no_theta():
    """不传 theta 时返回索引。"""
    data = np.array([0.0, -5.0, -10.0, -3.0, -8.0])
    values, indices = find_peaks(data)
    assert len(values) >= 2
    assert np.issubdtype(indices.dtype, np.integer), "不传 theta 应返回整数索引"


def test_get_psll_no_theta():
    """不传 theta 时 get_psll 返回索引。"""
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    psll_val, psll_idx = get_psll(af_db)
    assert isinstance(psll_val, float)
    assert isinstance(psll_idx, (int, np.integer)), "不传 theta 应返回整数索引"
    assert psll_val < 0


def test_get_psll_mainlobe_region_requires_theta():
    """mainlobe_region 不传 theta 时报错。"""
    data = np.random.randn(100)
    try:
        get_psll(data, mainlobe_region=(-10, 10))
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


# ============================================================
#  2D 平面阵测试
# ============================================================

def test_get_psll_2d_vs_1d_slice():
    """2D get_psll 每 φ 平面应与手动切片 1D get_psll 结果一致。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    # 4 元方阵
    x = np.array([0.0, 0.5, 0.0, 0.5])
    y = np.array([0.0, 0.0, 0.5, 0.5])
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    pslls, coords = get_psll(af_db, pat.theta_deg, pat.phi_deg)

    assert pslls.shape == (len(pat.phi_deg),)
    assert coords.shape == (len(pat.phi_deg), 2)

    # 与手动逐 φ 平面切片对比
    for j in range(len(pat.phi_deg)):
        val_1d, coord_1d = get_psll(af_db[:, j], pat.theta_deg)
        # 两者同为 -inf 时差值期望 nan, 需要额外处理
        both_inf = np.isinf(pslls[j]) and np.isinf(val_1d)
        assert both_inf or abs(pslls[j] - val_1d) < 1e-10, \
            f"φ={pat.phi_deg[j]}°: 2D={pslls[j]:.4f}, 1D={val_1d:.4f}"
        assert abs(coords[j, 0] - coord_1d) < 1e-4 \
            or (np.isnan(coords[j, 0]) and np.isnan(coord_1d))
        assert abs(coords[j, 1] - pat.phi_deg[j]) < 1e-10


def test_get_psll_2d_no_theta_phi():
    """2D get_psll 不传 theta/phi 时返回索引。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.array([0.0, 0.5, 0.0, 0.5])
    y = np.array([0.0, 0.0, 0.5, 0.5])
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    pslls, coords = get_psll(af_db)

    assert pslls.shape == (len(pat.phi_deg),)
    assert coords.shape == (len(pat.phi_deg), 2)
    assert np.issubdtype(coords.dtype, np.floating), "不传角度时坐标应为 float 索引"


def test_get_psll_2d_single_phi():
    """Nφ=1 时 shape 正确。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=0, phi_deg_end=0, phi_deg_step=1.0,
    )
    x = np.array([0.0, 0.5])
    y = np.array([0.0, 0.5])
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    pslls, coords = get_psll(af_db, pat.theta_deg, pat.phi_deg)
    assert pslls.shape == (1,)
    assert coords.shape == (1, 2)


def test_get_psll_2d_single_element():
    """2D 单阵元：所有 φ 平面均无副瓣。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=30.0,
    )
    af_db = Pattern.to_dB(pat.planar_af(np.array([0.0]), np.array([0.0])))

    pslls, coords = get_psll(af_db, pat.theta_deg, pat.phi_deg)
    assert np.all(np.isinf(pslls)), "单阵元所有 φ 平面应无副瓣"
    assert np.all(np.isnan(coords[:, 0]))
    assert np.all(~np.isnan(coords[:, 1])), "φ 坐标应保留"


def test_find_peaks_2d():
    """2D find_peaks 返回合并峰值，coords 为 (N, 2)。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.linspace(-2.25, 2.25, 10)
    y = np.zeros(10)
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    values, coords = find_peaks(af_db, pat.theta_deg, pat.phi_deg)

    assert coords.shape[1] == 2, "coords 列应为 [θ, φ]"
    assert len(values) > 0
    # 主瓣在前
    assert abs(values[0]) < 1e-6, "最高峰值应为主瓣（~0 dB）"


def test_find_peaks_2d_no_phi():
    """2D find_peaks 不传 phi 时 φ 坐标为索引。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.array([0.0, 0.5, 0.0, 0.5])
    y = np.array([0.0, 0.0, 0.5, 0.5])
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    values, coords = find_peaks(af_db, pat.theta_deg)

    assert coords.shape[1] == 2
    # φ 坐标应为浮点索引
    assert np.all(coords[:, 1] >= 0)


def test_get_overall_psll_basic():
    """全平面 PSLL 应 ≥ 各 φ 平面的 PSLL。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.linspace(-2.25, 2.25, 10)
    y = np.zeros(10)
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    psll, coord = get_overall_psll(af_db, pat.theta_deg, pat.phi_deg)

    assert isinstance(psll, float)
    assert coord.shape == (2,)
    assert psll < 0, "副瓣应为负 dB"

    # 应 ≥ 各 φ 平面的个体 PSLL（至少不能更低）
    pslls, _ = get_psll(af_db, pat.theta_deg, pat.phi_deg)
    valid = ~np.isinf(pslls)
    assert psll >= np.min(pslls[valid]), "全局 PSLL 不应低于各平面 PSLL"


def test_get_overall_psll_single_element():
    """单阵元全平面无副瓣。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=30.0,
    )
    af_db = Pattern.to_dB(pat.planar_af(np.array([0.0]), np.array([0.0])))

    psll, coord = get_overall_psll(af_db, pat.theta_deg, pat.phi_deg)
    assert psll == -np.inf
    assert np.all(np.isnan(coord))


def test_get_overall_psll_1d():
    """1D 输入时 get_overall_psll 等价于 get_psll。"""
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))

    psll1, coord1 = get_psll(af_db, pat.theta_deg)
    psll2, coord2 = get_overall_psll(af_db, pat.theta_deg)

    assert abs(psll1 - psll2) < 1e-10
    assert abs(coord2[0] - coord1) < 1e-10


def test_get_overall_psll_module_level():
    """验证模块级导入 get_overall_psll。"""
    from antopt import get_overall_psll
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    psll, coord = get_overall_psll(af_db, pat.theta_deg)
    assert isinstance(psll, float)


# ============================================================
#  mainlobe_region 测试
# ============================================================

def test_find_peaks_1d_mainlobe_region():
    """1D find_peaks 排除主瓣区域后主瓣应被移除。"""
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))

    # 不排除 → 有主瓣 0 dB
    v_all, c_all = find_peaks(af_db, pat.theta_deg)
    assert abs(v_all[0]) < 1e-6, "第一峰应为主瓣 ~0 dB"

    # 排除法向 ±15°
    v_mr, c_mr = find_peaks(af_db, pat.theta_deg, mainlobe_region=(-15, 15))
    assert v_mr[0] < 0, "排除主瓣后最高峰应为副瓣"
    # 所有剩余峰均不在排除区域内
    in_region = (c_mr >= -15) & (c_mr <= 15)
    assert not in_region.any(), "排除区域内不应有残留峰值"


def test_get_psll_1d_mainlobe_region_effect():
    """1D 排除主瓣后 PSLL 应不同（或不排除时找到的是主瓣附近小峰）。"""
    pat = Pattern(theta_deg_step=0.5)
    af_db = Pattern.to_dB(pat.linear_af(np.linspace(-2.25, 2.25, 10)))

    # 不排除主瓣 → 主瓣是最高峰，PSLL 是第一副瓣
    psll_no_mr, _ = get_psll(af_db, pat.theta_deg)
    # 排除法向 ±15° → PSLL 可能更大（如果有其他高副瓣）或更小
    psll_mr, _ = get_psll(af_db, pat.theta_deg, mainlobe_region=(-15, 15))

    assert abs(psll_no_mr - (-13.26)) < 0.5, "均匀线阵 PSLL 应在 -13.26 dB 附近"
    # 排除主瓣后 PSLL 应接近（均匀线阵第一副瓣在 ~±15° 以外）
    assert psll_mr < 0


def test_get_psll_2d_mainlobe_region_rectangular():
    """2D 矩形 mainlobe_region 排除主瓣区域。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    # 4 元方阵
    x = np.array([0.0, 0.5, -0.5, 0.0])
    y = np.array([0.0, 0.0, 0.0, 0.5])
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    # 2D 矩形主瓣区域
    mr = ((-10, 10), (-30, 30))
    pslls, coords = get_psll(af_db, pat.theta_deg, pat.phi_deg, mainlobe_region=mr)

    assert pslls.shape == (len(pat.phi_deg),)
    assert coords.shape == (len(pat.phi_deg), 2)

    # φ=0° 平面（在主瓣 φ 范围内）→ θ 主瓣区域被排除
    j_phi0 = np.where(np.abs(pat.phi_deg) < 0.1)[0][0]
    val_phi0, coord_phi0 = get_psll(
        af_db[:, j_phi0], pat.theta_deg, mainlobe_region=(-10, 10)
    )
    both_inf = np.isinf(pslls[j_phi0]) and np.isinf(val_phi0)
    assert both_inf or abs(pslls[j_phi0] - val_phi0) < 1e-10, \
        f"φ=0° 平面 PSLL 应与手动 1D(mr=(-10,10)) 一致"

    # φ=90° 平面（在主瓣 φ 范围外）→ θ 主瓣区域不应被排除
    j_phi90 = np.argmin(np.abs(pat.phi_deg - 90))
    val_90_nomr, _ = get_psll(af_db[:, j_phi90], pat.theta_deg)
    both_inf90 = np.isinf(pslls[j_phi90]) and np.isinf(val_90_nomr)
    assert both_inf90 or abs(pslls[j_phi90] - val_90_nomr) < 1e-10, \
        f"φ=90° 平面 PSLL 应与手动 1D(无 mr) 一致"


def test_find_peaks_2d_mainlobe_region():
    """2D find_peaks 排除矩形主瓣区域。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    # 10 元沿 x 轴均匀线阵（主瓣在 φ=0° 平面最窄）
    x = np.linspace(-2.25, 2.25, 10)
    y = np.zeros(10)
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    # 不排除
    v_all, c_all = find_peaks(af_db, pat.theta_deg, pat.phi_deg)
    n_all = len(v_all)

    # 排除法向 ±8° 且 φ∈[-30°,30°] 的矩形区域（覆盖主瓣）
    mr = ((-8, 8), (-30, 30))
    v_mr, c_mr = find_peaks(af_db, pat.theta_deg, pat.phi_deg, mainlobe_region=mr)

    assert len(v_mr) < n_all, "排除主瓣后峰值数应减少"
    # 确认主瓣区域内的峰值均被排除
    in_rect = (
        (c_mr[:, 0] >= -8) & (c_mr[:, 0] <= 8) &
        (c_mr[:, 1] >= -30) & (c_mr[:, 1] <= 30)
    )
    assert not in_rect.any(), "矩形区域内不应有残留峰值"
    # 排除前的主瓣 0 dB 峰值 (θ≈0,φ≈0) 已被移除
    mainlobe_mask = (c_all[:, 0] > -8) & (c_all[:, 0] < 8) & \
                    (c_all[:, 1] > -30) & (c_all[:, 1] < 30)
    assert mainlobe_mask.any(), "排除前应有主瓣"
    assert abs(v_all[mainlobe_mask][0]) < 1e-6, "排除前最高峰应为主瓣 ~0 dB"


def test_get_overall_psll_2d_mainlobe_region():
    """2D get_overall_psll + 矩形主瓣排除。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    # 4 元方阵
    x = np.array([0.0, 0.5, -0.5, 0.0])
    y = np.array([0.0, 0.0, 0.0, 0.5])
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    # 不用主瓣排除
    psll1, coord1 = get_overall_psll(af_db, pat.theta_deg, pat.phi_deg)

    # 用法向矩形主瓣排除
    mr = ((-10, 10), (-30, 30))
    psll2, coord2 = get_overall_psll(af_db, pat.theta_deg, pat.phi_deg,
                                      mainlobe_region=mr)

    # 排除主瓣后全局 PSLL 不应比排除前更低
    valid1 = not np.isinf(psll1)
    valid2 = not np.isinf(psll2)
    if valid1 and valid2:
        assert psll2 >= psll1 - 0.1, \
            f"排除主瓣后 PSLL 不应更低: 前={psll1:.2f}, 后={psll2:.2f}"
    # coord 应在排除区域外
    if valid2:
        in_rect = (-10 <= coord2[0] <= 10) and (-30 <= coord2[1] <= 30)
        assert not in_rect, f"全局 PSLL 坐标 {coord2} 不应在排除区域内"
