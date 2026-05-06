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
