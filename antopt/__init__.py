"""Core: 阵列天线核心计算模块。"""

from .pattern import Pattern
from .geometry import Element, LinearArray, PlanarArray, UniformLinearArray
from .element_pattern import ElementPattern
from .analysis import get_psll, find_peaks, get_overall_psll
from .mapping import LMMapper
from .optimizer import (BeamformingProblem, SidelobeProblem, NullSteeringProblem,
                        SparseArrayProblem, minimize, CMA, DE, GWO)
from .utils import to_json_flat, load_array_config, compute_pattern
from .space_mapping import run_space_mapping, parameter_extraction

try:
    from .surrogate import KrigingSurrogate, surrogate_optimize, lhs_sample
except ImportError:
    pass  # scikit-learn 未安装时跳过

__all__ = [
    "Pattern",
    "Element",
    "LinearArray",
    "PlanarArray",
    "UniformLinearArray",
    "ElementPattern",
    "LMMapper",
    "BeamformingProblem",
    "SidelobeProblem",
    "NullSteeringProblem",
    "SparseArrayProblem",
    "minimize",
    "CMA",
    "DE",
    "GWO",
    "to_json_flat",
    "load_array_config",
    "compute_pattern",
    "get_psll",
    "find_peaks",
    "get_overall_psll",
    "run_space_mapping",
    "parameter_extraction",
    "KrigingSurrogate",
    "surrogate_optimize",
    "lhs_sample",
]
