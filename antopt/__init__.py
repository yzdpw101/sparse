"""Core: 阵列天线核心计算模块。"""

from .pattern import Pattern
from .geometry import Element, LinearArray, PlanarArray, UniformLinearArray
from .element_pattern import ElementPattern
from .analysis import get_psll, find_peaks, get_overall_psll
from .mapping import LMMapper
from .optimizer import SparseArrayProblem, run_optimization
from .utils import to_json_flat, load_array_config, compute_pattern

__all__ = [
    "Pattern",
    "Element",
    "LinearArray",
    "PlanarArray",
    "UniformLinearArray",
    "ElementPattern",
    "LMMapper",
    "SparseArrayProblem",
    "run_optimization",
    "to_json_flat",
    "load_array_config",
    "compute_pattern",
    "get_psll",
    "find_peaks",
    "get_overall_psll",
]
