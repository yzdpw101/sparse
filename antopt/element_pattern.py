"""antopt.element_pattern — 向后兼容 shim, 实际代码在 antopt/core/element_pattern.py。"""

from .core.element_pattern import *  # noqa: F401,F403

__all__ = [
    "ElementPattern",
]
