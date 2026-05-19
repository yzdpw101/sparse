"""antopt — 稀布阵列天线优化库。"""

# 核心计算
from .core import (Element, LinearArray, PlanarArray, UniformLinearArray,
                   Pattern, ElementPattern)

# 映射器
from .matrix_spacing import LMMapper

# 优化器
from .optimizer import (BeamformingProblem, SidelobeProblem, NullSteeringProblem,
                        TripleNullProblem, SparseArrayProblem, minimize, CMA, DE, GWO)

# 空间映射
from .space_mapping import run_space_mapping, parameter_extraction

# 分析
from .analysis import (get_psll, find_peaks, get_overall_psll,
                       compute_directivity, compute_gain_drop)

# 工具
from .utils import to_json_flat, load_array_config, compute_pattern

# HFSS IO + 配置 + 画图 (新增)
from .hfss_io import read_hfss_csv, load_aep_patterns, compute_total_phases
from .config import BaseConfig, SparseConfig, ASMConfig, MSMConfig
from .plotting import plot_pattern_comparison, plot_convergence, plot_amplitude_distribution

# 代理模型 (可选依赖)
try:
    from .surrogate import KrigingSurrogate, surrogate_optimize, lhs_sample
except ImportError:
    pass  # scikit-learn 未安装时跳过

__all__ = [
    # core
    "Pattern",
    "Element",
    "LinearArray",
    "PlanarArray",
    "UniformLinearArray",
    "ElementPattern",
    # mapping
    "LMMapper",
    # optimizer
    "BeamformingProblem",
    "SidelobeProblem",
    "NullSteeringProblem",
    "TripleNullProblem",
    "SparseArrayProblem",
    "minimize",
    "CMA",
    "DE",
    "GWO",
    # space_mapping
    "run_space_mapping",
    "parameter_extraction",
    # analysis
    "get_psll",
    "find_peaks",
    "get_overall_psll",
    "compute_directivity",
    "compute_gain_drop",
    # utils
    "to_json_flat",
    "load_array_config",
    "compute_pattern",
    # hfss_io
    "read_hfss_csv",
    "load_aep_patterns",
    "compute_total_phases",
    # config
    "BaseConfig",
    "SparseConfig",
    "ASMConfig",
    "MSMConfig",
    # plotting
    "plot_pattern_comparison",
    "plot_convergence",
    "plot_amplitude_distribution",
    # surrogate (optional)
    "KrigingSurrogate",
    "surrogate_optimize",
    "lhs_sample",
]
