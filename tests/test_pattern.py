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
    af = Pattern.normalize(pat.linear_af(positions))
    peak_idx = np.argmax(af)
    assert abs(pat.theta_deg[peak_idx]) < 0.5, \
        f"峰值应位于 0°, 实际 {pat.theta_deg[peak_idx]:.1f}°"


def test_linear_af_scan():
    """验证波束扫描到 30° 时峰值偏移。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1,
                  theta0s_deg=30.0)
    positions = np.linspace(-2.25, 2.25, 10)
    af = Pattern.normalize(pat.linear_af(positions))
    peak_idx = np.argmax(af)
    assert abs(pat.theta_deg[peak_idx] - 30.0) < 2.0, \
        f"峰值应位于 30° 附近, 实际 {pat.theta_deg[peak_idx]:.1f}°"


def test_linear_af_single_element():
    """单阵元退化为全向方向图。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    af = Pattern.normalize(pat.linear_af(np.array([0.0])))
    assert np.allclose(af, 0.0, atol=1e-10), "单阵元方向图应全为 0 dB"


def test_normalize_output_range():
    """验证 normalize 输出最大值始终为 0 dB。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af = pat.linear_af(positions)
    af_norm = Pattern.normalize(af)
    assert abs(np.max(af_norm)) < 1e-10, "归一化方向图最大值应为 0 dB"


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
    assert not pat1.is_planar
    assert not pat1.is_multi_freq
    assert not pat1.is_multi_scan

    pat2 = Pattern(theta0s_deg=np.array([0.0, 30.0]))
    assert pat2.is_multi_scan

    pat3 = Pattern(frequenciesGHz=np.array([1.0, 2.0]))
    assert pat3.is_multi_freq
