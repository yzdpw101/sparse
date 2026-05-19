"""antopt.mapping — 向后兼容 shim, 实际代码在 antopt/matrix_spacing/linear.py。"""

from .matrix_spacing.linear import *  # noqa: F401,F403

__all__ = [
    "LMMapper",
]
