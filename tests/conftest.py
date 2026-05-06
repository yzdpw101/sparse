"""共享测试 fixtures。"""

import numpy as np
import pytest


@pytest.fixture
def uniform_linear_array_10():
    """10 元均匀直线阵，半波长间距。"""
    from antopt.geometry import UniformLinearArray
    return UniformLinearArray(num_elements=10, spacing=0.5, frequency=10e9)


@pytest.fixture
def default_theta():
    """标准 θ 角度采样。"""
    return np.linspace(-90, 90, 1801)
