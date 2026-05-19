"""hfss — HFSS 全波仿真接口 (win32com, Windows-only)。"""

try:
    from .run_patch import run_patch_simulation
    from .run_aep import run_aep_simulation
    __all__ = ["run_patch_simulation", "run_aep_simulation"]
except ImportError:
    # win32com 未安装 (非 Windows 环境)
    __all__ = []
