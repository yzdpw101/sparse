"""antopt.geometry — 向后兼容 shim, 实际代码在 antopt/core/geometry.py。"""

from .core.geometry import *  # noqa: F401,F403

__all__ = [
    "Element",
    "LinearArray",
    "PlanarArray",
    "UniformLinearArray",
]
