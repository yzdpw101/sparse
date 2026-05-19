"""标准绘图工具: 方向图对比、收敛曲线、幅度分布。"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def plot_pattern_comparison(theta, curves, title="", ylim=(-40, 3),
                            error_ylim=(-10, 10), save_path=None, figsize=(10, 8)):
    """方向图 + 误差图 (上下两子图)。

    Args:
        theta: 角度数组
        curves: dict, {label: (pattern_db, color, ls)}
            pattern_db 为归一化 dB 方向图
        title: 总标题
        ylim: 方向图 y 轴范围
        error_ylim: 误差图 y 轴范围
        save_path: 保存路径 (None=不保存)
        figsize: 图片尺寸

    Returns:
        (fig, axes)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 上图: 方向图对比
    ax = axes[0]
    ref_db = None
    for label, (db, color, ls) in curves.items():
        ax.plot(theta, db, color=color, ls=ls, lw=1.0, label=label)
        if ref_db is None:
            ref_db = db  # 第一条作为参考

    ax.set_title(title)
    ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 下图: 误差 (相对于第一条曲线)
    ax = axes[1]
    if ref_db is not None and len(curves) > 1:
        second_key = list(curves.keys())[1]
        second_db = curves[second_key][0]
        mask = ref_db > -40
        error = np.zeros_like(theta)
        error[mask] = ref_db[mask] - second_db[mask]
        rms = np.sqrt(np.mean(error[mask] ** 2)) if mask.any() else 0
        ax.plot(theta, error, 'k-', lw=0.8)
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_title(f'误差 (RMS={rms:.2f} dB)')
    else:
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.set_title('误差')

    ax.set_xlabel("$\\theta$ (deg)")
    ax.set_ylabel("Error (dB)")
    ax.set_ylim(error_ylim)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_convergence(iters, values, xlabel="Iteration", ylabel="PSLL (dB)",
                     title="", save_path=None, figsize=(8, 4)):
    """收敛曲线 (单条)。

    Args:
        iters: 迭代号数组
        values: 对应值数组
        xlabel, ylabel: 坐标轴标签
        title: 标题
        save_path: 保存路径

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iters, values, "o-", lw=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_amplitude_distribution(amps, title="", save_path=None, figsize=(10, 4)):
    """幅度分布柱状图。

    Args:
        amps: 幅度数组
        title: 标题
        save_path: 保存路径

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(amps)), amps, width=0.8, color='steelblue')
    ax.set_xlabel("Element Index")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
