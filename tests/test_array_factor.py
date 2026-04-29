"""测试阵因子计算模块。"""

import numpy as np
from sparsearray.core.geometry import UniformLinearArray
from sparsearray.core.array_factor import ArrayFactor


def test_ula_af_shape(uniform_linear_array_10, default_theta):
    """验证 AF 输出形状。"""
    af = ArrayFactor(uniform_linear_array_10)
    pattern = af.compute_af_normalized(default_theta)
    assert pattern.shape == default_theta.shape


def test_ula_peak_at_broadside(uniform_linear_array_10, default_theta):
    """验证均匀直线阵峰值在法向 (θ=0°)。"""
    af = ArrayFactor(uniform_linear_array_10)
    pattern = af.compute_af_normalized(default_theta)
    peak_idx = np.argmax(pattern)
    assert abs(default_theta[peak_idx]) < 0.5, "峰值应位于 0°"


def test_ula_msll_theoretical(uniform_linear_array_10, default_theta):
    """验证均匀直线阵 MSLL ≈ -13.26 dB。

    10 元半波长间距均匀直线阵的理论 MSLL 为 -13.26 dB。
    """
    from sparsearray.core.pattern import Pattern
    af = ArrayFactor(uniform_linear_array_10)
    pattern = af.compute_af_normalized(default_theta)
    p = Pattern(default_theta, pattern)
    msll = p.get_msll()
    assert abs(msll - (-13.26)) < 0.5, f"MSLL 应接近 -13.26 dB, 实际 {msll:.2f} dB"


def test_ula_hpbw(uniform_linear_array_10, default_theta):
    """验证 HPBW 大致正确。"""
    from sparsearray.core.pattern import Pattern
    af = ArrayFactor(uniform_linear_array_10)
    pattern = af.compute_af_normalized(default_theta)
    p = Pattern(default_theta, pattern)
    hpbw = p.get_hpbw()
    # 10 元半波长间距 ULA 的 HPBW ≈ 10.2°
    assert 8.0 < hpbw < 14.0, f"HPBW ({hpbw:.2f}°) 超出预期范围 [8°, 14°]"


def test_ula_scan(uniform_linear_array_10, default_theta):
    """验证波束扫描：扫描到 30° 时峰值偏移。"""
    af = ArrayFactor(uniform_linear_array_10)
    pattern = af.compute_af_normalized(default_theta, scan_theta=30.0)
    peak_idx = np.argmax(pattern)
    # 峰值应接近 30°
    assert abs(default_theta[peak_idx] - 30.0) < 2.0, \
        f"波束扫描峰值应位于 30° 附近, 实际 {default_theta[peak_idx]:.1f}°"


def test_single_element(uniform_linear_array_10, default_theta):
    """单阵元退化为各向同性方向图。"""
    arr = UniformLinearArray(num_elements=1, spacing=0.5, frequency=10e9)
    af = ArrayFactor(arr)
    pattern = af.compute_af_normalized(default_theta)
    # 单阵元方向图应为 0 dB（全向）
    assert np.allclose(pattern, 0.0, atol=1e-10), "单阵元方向图应全为 0 dB"
