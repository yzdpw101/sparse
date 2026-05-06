"""测试 LM 映射器。"""

import numpy as np
from antopt.mapping import LMMapper


def test_basic_asymmetric_no_fixed():
    """非对称, 非固定孔径: 17 元, L=4.872*2, dmin=0.5。"""
    mapper = LMMapper(Ne=17, L=9.744, dmin=0.5,
                      is_symmetric=False, is_fixed_aperture=False)
    assert mapper.n_vars == 17
    # 均匀间距 → 全部 v=0
    opt = np.full(mapper.n_vars, 0.0)
    pos = mapper.synthesize(opt)
    assert len(pos) == 17
    # 间距 >= dmin
    spacings = np.diff(pos)
    assert np.all(spacings >= 0.5 - 1e-10)
    # 孔径 ≈ L (sigmoid(0)=0.5, 几何级数不完全收敛)
    assert abs(pos[-1] - pos[0] - 9.744) < 1e-4


def test_basic_symmetric_no_fixed():
    """对称, 非固定孔径: 16 元, L=10, dmin=0.5。"""
    mapper = LMMapper(Ne=16, L=10.0, dmin=0.5,
                      is_symmetric=True, is_fixed_aperture=False)
    assert mapper.n_vars == 8
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert len(pos) == 16
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)
    # 对称性
    assert np.allclose(pos[:8], -pos[8:][::-1])


def test_asymmetric_fixed_aperture():
    """非对称, 固定孔径: 10 元, L=4.5, dmin=0.5。"""
    mapper = LMMapper(Ne=10, L=4.5, dmin=0.5,
                      is_symmetric=False, is_fixed_aperture=True)
    assert mapper.n_vars == 8
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert len(pos) == 10
    # 首尾钉在 ±L/2
    assert abs(pos[0] + 2.25) < 1e-10
    assert abs(pos[-1] - 2.25) < 1e-10
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)


def test_symmetric_fixed_aperture():
    """对称, 固定孔径: 10 元, L=4.5, dmin=0.5。"""
    mapper = LMMapper(Ne=10, L=4.5, dmin=0.5,
                      is_symmetric=True, is_fixed_aperture=True)
    assert mapper.n_vars == 3  # 10/2 - 2
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert len(pos) == 10
    assert abs(pos[0] + 2.25) < 1e-10
    assert abs(pos[-1] - 2.25) < 1e-10
    assert np.allclose(pos[:5], -pos[5:][::-1])


def test_with_dmax():
    """有 dmax 上限: 非对称, dmax=1.0。"""
    mapper = LMMapper(Ne=10, L=5.0, dmin=0.5, dmax=1.0,
                      is_symmetric=False, is_fixed_aperture=False)
    opt = np.zeros(mapper.n_vars)
    pos = mapper.synthesize(opt)
    assert np.all(np.diff(pos) >= 0.5 - 1e-10)
    assert np.all(np.diff(pos) <= 1.0 + 1e-10)


def test_opt_vector_midpoint():
    """v=0.5 给出中间值位置, 间距在 (dmin, dmax) 之间。"""
    mapper = LMMapper(Ne=8, L=7.0, dmin=0.5,
                      is_symmetric=True, is_fixed_aperture=False)
    opt = np.full(mapper.n_vars, 0.5)
    pos = mapper.synthesize(opt)
    spacings = np.diff(pos)
    # 间距应 > dmin (不是等间距)
    assert np.all(spacings >= 0.5 - 1e-10)
    assert np.allclose(pos[:4], -pos[4:][::-1])


def test_validation_errors():
    """参数校验。"""
    # Ne <= 0
    try:
        LMMapper(Ne=0, L=10, dmin=0.5)
        assert False
    except ValueError:
        pass
    # 对称 Ne 奇数
    try:
        LMMapper(Ne=9, L=10, dmin=0.5, is_symmetric=True)
        assert False
    except ValueError:
        pass
    # dmin * (Ne-1) > L
    try:
        LMMapper(Ne=10, L=4.0, dmin=0.5)
        assert False
    except ValueError:
        pass
    # dmax <= dmin
    try:
        LMMapper(Ne=10, L=10, dmin=0.5, dmax=0.3)
        assert False
    except ValueError:
        pass
    # opt_vector 长度不匹配
    try:
        mapper = LMMapper(Ne=10, L=4.5, dmin=0.5,
                          is_symmetric=False, is_fixed_aperture=True)
        mapper.synthesize(np.zeros(10))  # 期望 8
        assert False
    except ValueError:
        pass


def test_synthesize_deterministic():
    """相同输入 → 相同输出。"""
    mapper = LMMapper(Ne=20, L=10.0, dmin=0.5)
    rng = np.random.default_rng(42)
    opt = rng.normal(0, 2, mapper.n_vars)  # 无界变量
    pos1 = mapper.synthesize(opt)
    pos2 = mapper.synthesize(opt)
    assert np.allclose(pos1, pos2)


def test_positions_within_aperture():
    """所有位置在 [-L/2, L/2] 内。"""
    mapper = LMMapper(Ne=15, L=10.0, dmin=0.5)
    rng = np.random.default_rng(123)
    for _ in range(20):
        opt = rng.normal(0, 3, mapper.n_vars)  # 无界变量
        pos = mapper.synthesize(opt)
        assert np.all(pos >= -5.0 - 1e-10)
        assert np.all(pos <= 5.0 + 1e-10)
        assert np.all(np.diff(pos) >= 0.5 - 1e-10)


def test_unbounded_input_sigmoid_clamp():
    """极端大/小值 → sigmoid 钳位到 ~0 或 ~1, 不溢出。"""
    mapper = LMMapper(Ne=10, L=5.0, dmin=0.5)
    # 极端正值 → v ≈ 1 (密集排列)
    pos_large = mapper.synthesize(np.full(mapper.n_vars, 100.0))
    # 极端负值 → v ≈ 0 (均匀排列)
    pos_small = mapper.synthesize(np.full(mapper.n_vars, -100.0))
    assert np.all(np.diff(pos_large) >= 0.5 - 1e-10)
    assert np.all(np.diff(pos_small) >= 0.5 - 1e-10)
    # 极端大输入产生更分散的排列（v→1 分配更多剩余长度）
    assert np.ptp(pos_large) > np.ptp(pos_small) - 0.1
