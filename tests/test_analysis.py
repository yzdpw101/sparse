"""测试方向图分析函数。"""

import numpy as np
from core.pattern import Pattern
from core.analysis import get_psll, find_peaks


def test_get_psll_broadside():
    """验证均匀 10 元半波长直线阵法向 PSLL ≈ -13.26 dB。"""
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.normalize(pat.linear_af(positions))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert abs(psll_val - (-13.26)) < 0.5, f"PSLL 应接近 -13.26 dB, 实际 {psll_val:.2f} dB"
    assert isinstance(psll_angle, float)
    assert abs(psll_angle) > 0, "副瓣不应在法向"


def test_get_psll_single_element():
    """单阵元无副瓣，PSLL 应返回 (-inf, nan)。"""
    pat = Pattern()
    af_db = Pattern.normalize(pat.linear_af(np.array([0.0])))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert psll_val == -np.inf
    assert np.isnan(psll_angle)


def test_get_psll_module_level():
    """验证模块级导入正常。"""
    from core import get_psll
    pat = Pattern(theta_deg_step=1.0)
    af_db = Pattern.normalize(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
    assert isinstance(psll_val, float)
    assert isinstance(psll_angle, float)
    assert psll_val < 0


def test_get_psll_with_mainlobe_region():
    """指定主瓣区域排除后计算副瓣。"""
    pat = Pattern(theta_deg_step=0.1)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.normalize(pat.linear_af(positions))
    psll_val, psll_angle = get_psll(af_db, pat.theta_deg, mainlobe_region=(-10, 10))
    assert psll_val < 0, "副瓣电平应为负值"
    assert isinstance(psll_angle, float)


def test_find_peaks_broadside():
    """验证 find_peaks 能找到均匀线阵的峰值。"""
    pat = Pattern(theta_deg_step=1.0)
    positions = np.linspace(-2.25, 2.25, 10)
    af_db = Pattern.normalize(pat.linear_af(positions))
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
    af_db = Pattern.normalize(pat.linear_af(np.linspace(-2.25, 2.25, 10)))
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
