"""测试 LM 映射器。"""

import numpy as np
from antopt.mapping import LMMapper


# ── 非对称 ──

def test_asymmetric():
    """非对称: 17 元, L=9.744, dmin=0.5。"""
    mapper = LMMapper(Ne=17, L=9.744, dmin=0.5)
    assert mapper.n_vars == 17
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert len(pos) == 17
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)


def test_asymmetric_fixed_aperture():
    """非对称固定孔径: 10 元, L=4.5, dmin=0.5。"""
    mapper = LMMapper(Ne=10, L=4.5, dmin=0.5,
                      is_fixed_aperture=True)
    assert mapper.n_vars == 8
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert abs(pos[0] + 2.25) < 1e-10
    assert abs(pos[-1] - 2.25) < 1e-10
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)


def test_with_dmax():
    """有 dmax: 非对称, dmax=1.0, L 需 ≥ dmax*(Ne-1)。"""
    mapper = LMMapper(Ne=10, L=10.0, dmin=0.5, dmax=1.0)
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)
    assert np.all(np.diff(pos) <= 1.0 + 1e-10)


# ── 偶对称 ──

def test_even_symmetric():
    """偶对称: 16 元, L=10, dmin=0.5。"""
    mapper = LMMapper(Ne=16, L=10.0, dmin=0.5, is_symmetric=True)
    assert mapper.n_vars == 8
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert len(pos) == 16
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)
    assert np.allclose(pos[:8], -pos[8:][::-1])


def test_even_symmetric_fixed_aperture():
    """偶对称固定孔径: 10 元, L=4.5, dmin=0.5。"""
    mapper = LMMapper(Ne=10, L=4.5, dmin=0.5,
                      is_symmetric=True, is_fixed_aperture=True)
    assert mapper.n_vars == 4  # halfNe=5, n_fixed=1 (孔径端)
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert abs(pos[0] + 2.25) < 1e-10
    assert abs(pos[-1] - 2.25) < 1e-10
    assert np.allclose(pos[:5], -pos[5:][::-1])


# ── 奇对称 ──

def test_odd_symmetric():
    """奇对称: 17 元, n_vars = 17//2 = 8（右半侧 8 个阵元，无固定）。"""
    mapper = LMMapper(Ne=17, L=10.0, dmin=0.5, is_symmetric=True)
    assert mapper.has_center
    assert mapper.n_vars == 8
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert len(pos) == 17
    assert abs(pos[8]) < 1e-10
    assert np.allclose(pos[:8], -pos[9:][::-1])
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)  # 奇数对称：所有相邻间距 ≥ dmin


def test_odd_symmetric_fixed_aperture():
    """奇对称固定孔径: 17 元, n_vars = 17//2 - 1 = 7（右半侧 8，孔径端固定 1）。"""
    mapper = LMMapper(Ne=17, L=10.0, dmin=0.5,
                      is_symmetric=True, is_fixed_aperture=True)
    assert mapper.n_vars == 7
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert abs(pos[0] + 5.0) < 1e-10
    assert abs(pos[-1] - 5.0) < 1e-10
    assert abs(pos[8]) < 1e-10


def test_odd_symmetric_unbounded():
    """奇对称无界输入不溢出。"""
    mapper = LMMapper(Ne=15, L=8.0, dmin=0.5, is_symmetric=True)
    opt = np.full(mapper.n_vars, 50.0)
    pos = mapper.synthesize(opt)
    assert len(pos) == 15
    assert abs(pos[7]) < 1e-10
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)  # 奇数对称：所有相邻间距 ≥ dmin


# ── 通用 ──

def test_synthesize_deterministic():
    """相同输入 → 相同输出。"""
    mapper = LMMapper(Ne=20, L=10.0, dmin=0.5)
    rng = np.random.default_rng(42)
    opt = rng.normal(0, 2, mapper.n_vars)
    pos1 = mapper.synthesize(opt)
    pos2 = mapper.synthesize(opt)
    assert np.allclose(pos1, pos2)


def test_positions_within_aperture():
    """所有位置在 [-L/2, L/2] 内。"""
    mapper = LMMapper(Ne=15, L=10.0, dmin=0.5)
    rng = np.random.default_rng(123)
    for _ in range(20):
        opt = rng.normal(0, 3, mapper.n_vars)
        pos = mapper.synthesize(opt)
        assert np.all(pos >= -5.0 - 1e-10)
        assert np.all(pos <= 5.0 + 1e-10)
        assert np.all(np.diff(pos) >= 0.5 - 1e-10)


def test_unbounded_input_sigmoid_clamp():
    """极端大/小值 → sigmoid 钳位。"""
    mapper = LMMapper(Ne=10, L=5.0, dmin=0.5)
    pos_large = mapper.synthesize(np.full(mapper.n_vars, 100.0))
    pos_small = mapper.synthesize(np.full(mapper.n_vars, -100.0))
    assert np.all(np.diff(pos_large) >= 0.5 - 1e-10)
    assert np.all(np.diff(pos_small) >= 0.5 - 1e-10)
    assert np.ptp(pos_large) > np.ptp(pos_small) - 0.1


# ── 参数校验 ──

def test_validation_errors():
    """基础参数校验。"""
    try:
        LMMapper(Ne=0, L=10, dmin=0.5)
        assert False
    except ValueError:
        pass
    try:
        LMMapper(Ne=1, L=10, dmin=0.5, is_symmetric=True)
        assert False
    except ValueError:
        pass
    try:
        LMMapper(Ne=10, L=4.0, dmin=0.5)
        assert False
    except ValueError:
        pass
    try:
        mapper = LMMapper(Ne=10, L=5.0, dmin=0.5)
        mapper.synthesize(np.zeros(5))
        assert False
    except ValueError:
        pass


def test_dmax_and_fixed_aperture_rejected():
    """dmax + is_fixed_aperture 同时启用应报错。"""
    try:
        LMMapper(Ne=10, L=5.0, dmin=0.5, dmax=1.0, is_fixed_aperture=True)
        assert False
    except ValueError:
        pass


def test_dmax_and_fixed_symmetric_rejected():
    """dmax + 对称 + 固定孔径 同时启用应报错。"""
    try:
        LMMapper(Ne=10, L=10, dmin=0.5, dmax=1.0,
                 is_symmetric=True, is_fixed_aperture=True)
        assert False
    except ValueError:
        pass
