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
    spacings = np.diff(pos)
    assert np.all(spacings >= 0.5 - 1e-10)


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


# ── 奇对称 ──

def test_odd_symmetric():
    """奇对称: 17 元, n_vars = (17-1)/2 = 8。"""
    mapper = LMMapper(Ne=17, L=10.0, dmin=0.5, is_symmetric=True)
    assert mapper.has_center
    assert mapper.n_vars == 8
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert len(pos) == 17
    assert abs(pos[8]) < 1e-10  # 中心在 x=0
    assert np.allclose(pos[:8], -pos[9:][::-1])
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)


def test_odd_symmetric_unbounded():
    """奇对称无界输入不溢出。"""
    mapper = LMMapper(Ne=15, L=8.0, dmin=0.5, is_symmetric=True)
    opt = np.full(mapper.n_vars, 50.0)
    pos = mapper.synthesize(opt)
    assert len(pos) == 15
    assert abs(pos[7]) < 1e-10
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)


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
    # Ne <= 0
    try:
        LMMapper(Ne=0, L=10, dmin=0.5)
        assert False
    except ValueError:
        pass
    # 对称 Ne=1
    try:
        LMMapper(Ne=1, L=10, dmin=0.5, is_symmetric=True)
        assert False
    except ValueError:
        pass
    # dmin * (Ne-1) > L
    try:
        LMMapper(Ne=10, L=4.0, dmin=0.5)
        assert False
    except ValueError:
        pass
    # opt_vector 长度不匹配
    try:
        mapper = LMMapper(Ne=10, L=5.0, dmin=0.5)
        mapper.synthesize(np.zeros(5))
        assert False
    except ValueError:
        pass


def test_dmax_rejected():
    """dmax 已废弃。"""
    try:
        LMMapper(Ne=10, L=5.0, dmin=0.5, dmax=1.0)
        assert False
    except NotImplementedError:
        pass


def test_fixed_aperture_rejected():
    """is_fixed_aperture 已废弃。"""
    try:
        LMMapper(Ne=10, L=5.0, dmin=0.5, is_fixed_aperture=True)
        assert False
    except NotImplementedError:
        pass


def test_dmax_and_symmetric():
    """dmax + 对称 → 同样报错。"""
    try:
        LMMapper(Ne=10, L=10, dmin=0.5, dmax=1.0, is_symmetric=True)
        assert False
    except NotImplementedError:
        pass
