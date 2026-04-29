"""sparsearray - 稀疏/稀布阵列天线优化库。"""

__version__ = "0.1.0"

# 核心数据结构
from .core.geometry import Element, LinearArray, PlanarArray, UniformLinearArray
from .core.array_factor import ArrayFactor
from .core.pattern import Pattern
from .core.element_pattern import ElementPattern
