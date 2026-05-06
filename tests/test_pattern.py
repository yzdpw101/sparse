"""测试 Pattern 方向图计算器。"""

import numpy as np
from core.pattern import Pattern


def test_linear_af_shape():
    """验证 linear_af 输出形状。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af = pat.linear_af(positions)
    assert af.shape == (1801,), f"期望 (1801,), 实际 {af.shape}"


def test_linear_af_peak_at_broadside():
    """验证线性阵峰值在法向 (θ=0°)。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af = Pattern.to_dB(pat.linear_af(positions))
    peak_idx = np.argmax(af)
    assert abs(pat.theta_deg[peak_idx]) < 0.5, \
        f"峰值应位于 0°, 实际 {pat.theta_deg[peak_idx]:.1f}°"


def test_linear_af_scan():
    """验证波束扫描到 30° 时峰值偏移。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1,
                  theta0s_deg=30.0)
    positions = np.linspace(-2.25, 2.25, 10)
    af = Pattern.to_dB(pat.linear_af(positions))
    peak_idx = np.argmax(af)
    assert abs(pat.theta_deg[peak_idx] - 30.0) < 2.0, \
        f"峰值应位于 30° 附近, 实际 {pat.theta_deg[peak_idx]:.1f}°"


def test_linear_af_single_element():
    """单阵元退化为全向方向图。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    af = Pattern.to_dB(pat.linear_af(np.array([0.0])))
    assert np.allclose(af, 0.0, atol=1e-10), "单阵元方向图应全为 0 dB"


def test_to_dB_output_range():
    """验证 to_dB 输出最大值始终为 0 dB。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af = pat.linear_af(positions)
    af_db = Pattern.to_dB(af)
    assert abs(np.max(af_db)) < 1e-10, "归一化 dB 方向图最大值应为 0 dB"


def test_theta_grid_generation():
    """验证角度网格生成正确。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=1.0)
    assert len(pat.theta_deg) == 181
    assert abs(pat.theta_deg[0] - (-90)) < 1e-10
    assert abs(pat.theta_deg[-1] - 90) < 1e-10


def test_multi_scan_shape():
    """验证多角度扫描输出形状。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=1.0,
                  theta0s_deg=np.array([0.0, 30.0, 45.0]))
    positions = np.linspace(-2.25, 2.25, 10)
    af = pat.linear_af(positions)
    assert af.shape == (3, 181), f"期望 (3, 181), 实际 {af.shape}"


def test_properties():
    """验证属性标志位。"""
    pat1 = Pattern()
    assert pat1.array_type == "linear"
    assert not pat1.is_planar
    assert not pat1.is_multi_freq
    assert not pat1.is_multi_scan

    pat2 = Pattern(theta0s_deg=np.array([0.0, 30.0]))
    assert pat2.is_multi_scan

    pat3 = Pattern(frequenciesGHz=np.array([1.0, 2.0]))
    assert pat3.is_multi_freq


# ============================================================
#  normalize / to_dB
# ============================================================

def test_normalize_linear():
    """normalize 返回线性域归一化幅度，范围 [0, 1]。"""
    af = np.array([2.0, 1.0, 3.0], dtype=complex)
    norm = Pattern.normalize(af)
    assert np.max(norm) == 1.0
    assert np.min(norm) >= 0.0
    assert abs(norm[2] - 1.0) < 1e-12


def test_to_dB_normalized_vs_raw():
    """to_dB(normalized=True) 峰值应为 0 dB, normalized=False 不归一化。"""
    af = np.array([1.0, 0.5, 0.1], dtype=complex)
    db_norm = Pattern.to_dB(af, normalized=True)
    db_raw = Pattern.to_dB(af, normalized=False)
    assert abs(np.max(db_norm)) < 1e-10, "归一化 dB 最大值应为 0 dB"
    assert abs(db_raw[0] - 0.0) < 1e-10, "未归一化时 1.0 应为 0 dB"


def test_to_dB_default_normalized():
    """to_dB 默认启用归一化。"""
    af = np.array([0.5, 1.0, 0.25], dtype=complex)
    db_def = Pattern.to_dB(af)
    assert abs(np.max(db_def)) < 1e-10


# ============================================================
#  array_type / af()
# ============================================================

def test_array_type_linear_default():
    """默认 array_type 为 linear。"""
    pat = Pattern()
    assert pat.array_type == "linear"


def test_array_type_planar():
    """指定 array_type="planar"。"""
    pat = Pattern(array_type="planar",
                  phi_deg_start=-90, phi_deg_end=90, phi_deg_step=1.0)
    assert pat.array_type == "planar"
    assert pat.is_planar


def test_array_type_invalid():
    """非法 array_type 应报错。"""
    try:
        Pattern(array_type="circular")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_af_dispatches_to_linear_af():
    """array_type="linear" 时 af() 等价于 linear_af()。"""
    pat = Pattern(theta_deg_step=1.0)
    positions = np.linspace(-2.25, 2.25, 10)
    af1 = pat.af(positions)
    af2 = pat.linear_af(positions)
    assert np.allclose(af1, af2)


# ============================================================
#  symmetric
# ============================================================

def test_symmetric_property():
    """验证 symmetric 属性和默认值。"""
    pat1 = Pattern()
    assert not pat1.symmetric
    pat2 = Pattern(symmetric=True)
    assert pat2.symmetric


def test_linear_af_symmetric_equivalent():
    """对称阵和非对称阵输出应等价（关于原点对称的均匀阵列）。"""
    pat_asym = Pattern(theta_deg_step=1.0)
    pat_sym = Pattern(symmetric=True, theta_deg_step=1.0)

    # 非对称：10 元，关于原点对称
    full = np.linspace(-2.25, 2.25, 10)
    half = full[full > 0]  # 半边 x > 0，不含中心

    af_full = pat_asym.linear_af(full)
    af_half = pat_sym.linear_af_symmetric(half, has_center=False)

    assert np.allclose(af_full, af_half), "对称与非对称阵因子应等价"


def test_linear_af_symmetric_with_center():
    """含中心阵元的对称阵列。"""
    pat = Pattern(symmetric=True, theta_deg_step=1.0)
    # 5 元半边 (含中心 x=0 用 has_center=True)
    half = np.array([0.5, 1.0])
    af = pat.linear_af_symmetric(half, has_center=True)
    assert af.shape == (181,)


def test_af_dispatches_to_symmetric():
    """symmetric=True 时 af() 应调用 symmetric。"""
    pat = Pattern(symmetric=True, theta_deg_step=1.0)
    half = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    af = pat.af(half)
    assert af.shape == (181,)


# ============================================================
#  planar
# ============================================================

def test_planar_af_shape():
    """平面阵输出形状应为 (Nθ, Nφ)。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=1.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=2.0,
    )
    n = 4
    x = np.array([0.0, 0.5, 0.0, 0.5])
    y = np.array([0.0, 0.0, 0.5, 0.5])
    af = pat.planar_af(x, y)
    assert af.shape == (181, 91), f"期望 (181, 91), 实际 {af.shape}"


def test_planar_af_peak_at_broadside():
    """平面阵法向 (θ=0°) 应有峰值（AF 幅度 = 阵元数）。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=0.5,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=2.0,
    )
    x = np.array([0.0, 0.5, -0.5, 0.0, 0.5, -0.5])
    y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, -0.5])
    af = pat.planar_af(x, y)
    # 法向切片：θ=0° 对应 theta_deg 的中间索引
    idx_theta0 = len(pat.theta_deg) // 2
    # 法向 AF 应为阵元数（所有阵元同相）
    assert np.allclose(np.abs(af[idx_theta0, :]), len(x), atol=1e-10), \
        "法向所有 φ 的 |AF| 应等于阵元数"
    # 非零角度 AF 应小于阵元数
    assert np.max(np.abs(af[0, :])) < len(x), "偏离法向 AF 应变小"


def test_af_planar_dispatch():
    """array_type='planar' 时 af() 应调用 planar_af()。"""
    pat = Pattern(
        array_type="planar",
        theta_deg_step=1.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )
    x = np.array([0.0, 0.5])
    y = np.array([0.0, 0.5])
    af = pat.af(x, y)
    assert af.shape == (181, 37)


# ============================================================
#  planar_af_symmetric
# ============================================================

def test_planar_af_symmetric_equivalent():
    """四象限对称阵列：planar_af 和 planar_af_symmetric 应等价。"""
    pat_asym = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )
    pat_sym = Pattern(
        array_type="planar", symmetric=True,
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )

    # 第一象限的两个阵元
    q1_x = np.array([0.3, 0.7])
    q1_y = np.array([0.4, 0.6])

    # 构造完整四象限阵列（不含中心）
    full_x = np.concatenate([q1_x, -q1_x, -q1_x, q1_x])
    full_y = np.concatenate([q1_y, q1_y, -q1_y, -q1_y])

    af_full = pat_asym.planar_af(full_x, full_y)
    af_sym = pat_sym.planar_af_symmetric(q1_x, q1_y, has_center=False)

    assert np.allclose(af_full, af_sym), "对称与非对称平面阵因子应等价"


def test_planar_af_symmetric_with_center():
    """四象限对称 + 中心阵元。"""
    pat = Pattern(
        array_type="planar", symmetric=True,
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )

    q1_x = np.array([0.5, 1.0])
    q1_y = np.array([0.5, 1.0])

    # 非对称版本：4*2 + 中心 = 9 阵元
    full_x = np.concatenate([[0.0], q1_x, -q1_x, -q1_x, q1_x])
    full_y = np.concatenate([[0.0], q1_y, q1_y, -q1_y, -q1_y])

    pat_asym = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )

    af_full = pat_asym.planar_af(full_x, full_y)
    af_sym = pat.planar_af_symmetric(q1_x, q1_y, has_center=True)

    assert np.allclose(af_full, af_sym), \
        "含中心阵元的对称/非对称平面阵因子应等价"


def test_planar_af_symmetric_axis_element():
    """轴上阵元（y=0）只有 2 重对称。"""
    pat_sym = Pattern(
        array_type="planar", symmetric=True,
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )
    pat_asym = Pattern(
        array_type="planar",
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )

    # 第一象限 x 轴上阵元 (x>0, y=0)
    hx = np.array([0.5, 0.8])
    hy = np.array([0.0, 0.0])

    # 完整阵列：两侧对称 (x,0) 和 (-x,0)
    full_x = np.concatenate([hx, -hx])
    full_y = np.concatenate([hy, hy])

    af_full = pat_asym.planar_af(full_x, full_y)
    af_sym = pat_sym.planar_af_symmetric(hx, hy, has_center=False)

    assert np.allclose(af_full, af_sym), "轴上阵元 (2重对称) 应等价"


def test_planar_af_symmetric_shape():
    """对称平面阵输出形状检查。"""
    pat = Pattern(
        array_type="planar", symmetric=True,
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )
    hx = np.array([0.3, 0.7])
    hy = np.array([0.4, 0.6])
    af = pat.planar_af_symmetric(hx, hy)
    assert af.shape == (91, 37), f"期望 (91, 37), 实际 {af.shape}"


def test_af_dispatches_to_planar_symmetric():
    """symmetric=True + planar 时 af() 应调用 planar_af_symmetric()。"""
    pat = Pattern(
        array_type="planar", symmetric=True,
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )
    hx = np.array([0.3, 0.7])
    hy = np.array([0.4, 0.6])
    af = pat.af(hx, hy)
    assert af.shape == (91, 37)


# ============================================================
#  线阵 vs 平面阵 φ=0° 切片一致性
# ============================================================

def test_linear_vs_planar_phi0():
    """沿 x 轴排列的平面阵 φ=0° 切片应与线阵结果一致。"""
    positions = np.linspace(-2.25, 2.25, 10)

    pat_lin = Pattern(theta_deg_step=1.0)
    af_lin = pat_lin.linear_af(positions)

    pat_pl = Pattern(
        array_type="planar",
        theta_deg_step=1.0,
        phi_deg_start=0, phi_deg_end=0, phi_deg_step=1.0,
    )
    af_pl = pat_pl.planar_af(positions, np.zeros_like(positions))

    assert np.allclose(af_lin, af_pl[:, 0]), "线阵与平面阵 φ=0° 切片应一致"


def test_linear_vs_planar_psll():
    """线阵和平面阵(φ=0°)的 PSLL 应一致。"""
    from core.analysis import get_psll

    positions = np.linspace(-2.25, 2.25, 10)

    pat_lin = Pattern(theta_deg_step=0.1)
    af_lin_db = Pattern.to_dB(pat_lin.linear_af(positions))
    psll_lin, _ = get_psll(af_lin_db, pat_lin.theta_deg)

    pat_pl = Pattern(
        array_type="planar",
        theta_deg_step=0.1,
        phi_deg_start=0, phi_deg_end=0, phi_deg_step=1.0,
    )
    af_pl_db = Pattern.to_dB(pat_pl.planar_af(positions, np.zeros_like(positions)))
    psll_pl, _ = get_psll(af_pl_db[:, 0], pat_pl.theta_deg)

    assert abs(psll_lin - psll_pl) < 0.01, \
        f"线阵 PSLL={psll_lin:.4f}, 平面阵 PSLL={psll_pl:.4f}"


# ============================================================
#  多频 / 多角度一致性（相同参数→相同结果）
# ============================================================

def test_multi_freq_identical_linear():
    """两个相同频率的线阵结果应一致。"""
    positions = np.linspace(-2.25, 2.25, 10)
    pat = Pattern(frequenciesGHz=np.array([1.0, 1.0]), theta_deg_step=1.0)
    af = pat.linear_af(positions)
    assert af.shape[0] == 2
    assert np.allclose(af[0], af[1]), "相同频率的两组结果应一致"


def test_multi_scan_identical_linear():
    """两个相同扫描角的线阵结果应一致。"""
    positions = np.linspace(-2.25, 2.25, 10)
    pat = Pattern(theta0s_deg=np.array([0.0, 0.0]), theta_deg_step=1.0)
    af = pat.linear_af(positions)
    assert af.shape[0] == 2
    assert np.allclose(af[0], af[1]), "相同扫描角的两组结果应一致"


def test_multi_freq_identical_planar():
    """两个相同频率的平面阵结果应一致。"""
    x = np.array([0.0, 0.5, -0.5, 0.0])
    y = np.array([0.0, 0.0, 0.0, 0.5])
    pat = Pattern(
        array_type="planar",
        frequenciesGHz=np.array([1.0, 1.0]),
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    af = pat.planar_af(x, y)
    assert af.shape[0] == 2
    assert np.allclose(af[0], af[1]), "相同频率的两组平面阵结果应一致"


def test_multi_scan_identical_planar():
    """两个相同扫描角的平面阵结果应一致。"""
    x = np.array([0.0, 0.5, -0.5])
    y = np.array([0.0, 0.0, 0.0])
    pat = Pattern(
        array_type="planar",
        theta0s_deg=np.array([0.0, 0.0]),
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    af = pat.planar_af(x, y)
    assert af.shape[0] == 2
    assert np.allclose(af[0], af[1]), "相同扫描角的两组平面阵结果应一致"


def test_multi_freq_identical_symmetric():
    """两个相同频率的对称线阵结果应一致。"""
    half = np.array([0.5, 1.0, 1.5])
    pat = Pattern(symmetric=True, frequenciesGHz=np.array([1.0, 1.0]),
                  theta_deg_step=1.0)
    af = pat.linear_af_symmetric(half)
    assert af.shape[0] == 2
    assert np.allclose(af[0], af[1]), "相同频率的两组对称线阵结果应一致"


def test_multi_scan_identical_symmetric():
    """两个相同扫描角的对称线阵结果应一致。"""
    half = np.array([0.5, 1.0, 1.5])
    pat = Pattern(symmetric=True, theta0s_deg=np.array([0.0, 0.0]),
                  theta_deg_step=1.0)
    af = pat.linear_af_symmetric(half)
    assert af.shape[0] == 2
    assert np.allclose(af[0], af[1]), "相同扫描角的两组对称线阵结果应一致"


# ============================================================
#  theta0s / phi0s broadcast 配对
# ============================================================

def test_planar_theta_scalar_phi_array():
    """theta0s=标量, phi0s=数组 → broadcast 为多扫描。"""
    pat = Pattern(
        array_type="planar",
        theta0s_deg=30.0,
        phi0s_deg=np.array([0.0, 45.0, 90.0]),
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    assert pat.is_multi_scan
    x = np.array([0.0, 0.5])
    y = np.array([0.0, 0.5])
    af = pat.planar_af(x, y)
    assert af.shape[0] == 3, f"期望 3 个扫描方向, 实际 {af.shape[0]}"
    # 三个扫描方向不同
    assert not np.allclose(af[0], af[1]), "φ=0° 和 φ=45° 扫描应不同"


def test_planar_theta_array_phi_scalar():
    """theta0s=数组, phi0s=标量 → broadcast 为多扫描。"""
    pat = Pattern(
        array_type="planar",
        theta0s_deg=np.array([0.0, 30.0]),
        phi0s_deg=0.0,
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    assert pat.is_multi_scan
    x = np.array([0.0, 0.5, -0.5, 0.0])
    y = np.array([0.0, 0.0, 0.0, 0.5])
    af = pat.planar_af(x, y)
    assert af.shape[0] == 2, f"期望 2 个扫描方向, 实际 {af.shape[0]}"
    # 法向和 30° 扫描应不同
    assert not np.allclose(af[0], af[1]), "θ₀=0° 和 θ₀=30° 扫描应不同"


def test_planar_both_array_paired():
    """theta0s 和 phi0s 等长数组 → 一一配对。"""
    pat = Pattern(
        array_type="planar",
        theta0s_deg=np.array([0.0, 30.0]),
        phi0s_deg=np.array([0.0, 45.0]),
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
    )
    x = np.array([0.0, 0.5])
    y = np.array([0.0, 0.5])
    af = pat.planar_af(x, y)
    assert af.shape[0] == 2
    # 扫描方向 1: (0°,0°), 扫描方向 2: (30°,45°)
    assert not np.allclose(af[0], af[1]), "不同扫描方向应不同"


def test_planar_both_array_mismatch_error():
    """theta0s 和 phi0s 长度不等(都>1) → ValueError。"""
    try:
        Pattern(
            array_type="planar",
            theta0s_deg=np.array([0.0, 30.0, 45.0]),
            phi0s_deg=np.array([0.0, 90.0]),
            theta_deg_step=2.0,
            phi_deg_start=-90, phi_deg_end=90, phi_deg_step=10.0,
        )
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_planar_scan_angle_peak_at_scanned_direction():
    """验证平面阵扫描到 (θ₀=30°, φ₀=45°) 时峰值在该方向。"""
    pat = Pattern(
        array_type="planar",
        theta0s_deg=30.0,
        phi0s_deg=45.0,
        theta_deg_step=2.0,
        phi_deg_start=-90, phi_deg_end=90, phi_deg_step=5.0,
    )
    # 4×4 均匀面阵
    n = 4
    xx, yy = np.meshgrid(np.arange(n) * 0.5, np.arange(n) * 0.5)
    x, y = xx.ravel(), yy.ravel()
    af_db = Pattern.to_dB(pat.planar_af(x, y))

    # 峰值应在 (θ≈30°, φ≈45°)
    idx_peak = np.unravel_index(np.argmax(af_db), af_db.shape)
    theta_peak = pat.theta_deg[idx_peak[0]]
    phi_peak = pat.phi_deg[idx_peak[1]]
    assert abs(theta_peak - 30.0) < 5.0, f"峰值 θ 应在 30° 附近, 实际 {theta_peak:.1f}°"
    assert abs(phi_peak - 45.0) < 10.0, f"峰值 φ 应在 45° 附近, 实际 {phi_peak:.1f}°"
