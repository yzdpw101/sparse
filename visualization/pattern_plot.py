"""方向图绘图函数。"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_pattern_1d(ax, theta_deg: np.ndarray, af_db: np.ndarray,
                    peaks: Optional[tuple[np.ndarray, np.ndarray]] = None,
                    psll: Optional[tuple[float, float]] = None,
                    title: str = "方向图",
                    ylim: tuple[float, float] = (-50, 3)):
    """直角坐标方向图。

    Args:
        ax: matplotlib Axes
        theta_deg: 角度网格（度）
        af_db: 归一化 dB 方向图
        peaks: (values, angles) 峰值，可选
        psll: (psll_val, psll_angle) 最高副瓣，可选
        title: 图标题
        ylim: y 轴范围
    """
    ax.plot(theta_deg, af_db, linewidth=1.2)
    if peaks is not None:
        values, angles = peaks
        ax.scatter(angles, values, c='red', s=30, zorder=5, label='峰值')
    if psll is not None:
        ax.scatter([psll[1]], [psll[0]], c='orange', s=60, marker='D',
                   zorder=6, label=f'PSLL: {psll[0]:.4f} dB')
    ax.set_xlabel('θ (度)')
    ax.set_ylabel('归一化方向图 (dB)')
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    if peaks is not None or psll is not None:
        ax.legend()


def plot_pattern_polar(ax, theta_deg: np.ndarray, af_db: np.ndarray,
                       title: str = "极坐标方向图",
                       ylim: tuple[float, float] = (-50, 3)):
    """极坐标方向图（上半平面，-90° ~ 90°）。

    Args:
        ax: matplotlib PolarAxes
        theta_deg: 角度网格（度）
        af_db: 归一化 dB 方向图
        title: 图标题
        ylim: r 轴范围
    """
    theta_rad = np.deg2rad(theta_deg)
    ax.plot(theta_rad, af_db, linewidth=1.2)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title(title)
    ax.set_ylim(*ylim)
