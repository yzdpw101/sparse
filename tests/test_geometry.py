"""测试阵列几何模块。"""

import numpy as np
from core.geometry import Element, LinearArray, PlanarArray, UniformLinearArray


def test_element_defaults():
    """Element 默认值测试。"""
    e = Element(x=1.0)
    assert e.x == 1.0
    assert e.y == 0.0
    assert e.z == 0.0
    assert e.amplitude == 1.0
    assert e.phase == 0.0
    assert e.active is True


def test_linear_array_basic():
    """LinearArray 基本构造测试。"""
    positions = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    arr = LinearArray(positions, frequency=10e9)
    assert arr.num_elements == 5
    assert arr.num_active == 5
    assert arr.frequency == 10e9
    assert np.allclose(arr.positions_x, positions)


def test_linear_array_with_amplitudes():
    """带幅度加权的 LinearArray 测试。"""
    pos = np.array([0.0, 0.5, 1.0])
    amps = np.array([0.5, 1.0, 0.5])
    arr = LinearArray(pos, frequency=10e9, amplitudes=amps)
    assert np.allclose(arr.amplitudes, amps)


def test_uniform_linear_array():
    """UniformLinearArray 对称性测试。"""
    ula = UniformLinearArray(num_elements=10, spacing=0.5, frequency=10e9)
    assert ula.num_elements == 10
    xs = ula.positions_x
    assert np.allclose(xs, -xs[::-1]), "均匀直线阵应关于原点对称"


def test_planar_array():
    """PlanarArray 基本构造测试。"""
    x = np.array([0.0, 0.5, 0.0, 0.5])
    y = np.array([0.0, 0.0, 0.5, 0.5])
    arr = PlanarArray(x, y, frequency=10e9)
    assert arr.num_elements == 4
    assert np.allclose(arr.positions_x, x)
    assert np.allclose(arr.positions_y, y)


def test_activate_subset():
    """activate_subset 测试。"""
    ula = UniformLinearArray(num_elements=10, spacing=0.5, frequency=10e9)
    indices = np.array([0, 2, 4, 6, 8])
    ula.activate_subset(indices)
    assert ula.num_active == 5
    assert ula.active_mask.sum() == 5


def test_set_excitations():
    """批量设置激励测试。"""
    ula = UniformLinearArray(num_elements=5, spacing=0.5, frequency=10e9)
    amps = np.array([0.1, 0.3, 0.5, 0.3, 0.1])
    phases = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    ula.set_excitations(amplitudes=amps, phases=phases)
    assert np.allclose(ula.amplitudes, amps)
    assert np.allclose(ula.phases, phases)
