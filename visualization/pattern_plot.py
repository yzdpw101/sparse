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


# ============================================================
#  平面阵 2D / 3D 可视化
# ============================================================

def plot_pattern_2d(ax, theta_deg: np.ndarray, phi_deg: np.ndarray,
                    af_db: np.ndarray,
                    title: str = "2D 方向图 (dB)",
                    levels: int = 40,
                    cmap: str = 'jet'):
    """平面阵 2D 直角坐标 heatmap（θ vs φ）。

    Args:
        ax: matplotlib Axes
        theta_deg: θ 角度网格（度），shape (Nθ,)
        phi_deg: φ 角度网格（度），shape (Nφ,)
        af_db: 归一化 dB 方向图，shape (Nθ, Nφ)
        title: 图标题
        levels: 等值线层数
        cmap: 颜色映射
    """
    THETA, PHI = np.meshgrid(phi_deg, theta_deg)  # pcolormesh expects (Nθ, Nφ)
    vmin, vmax = max(np.nanmin(af_db), -60), 0
    cf = ax.pcolormesh(THETA, PHI, af_db, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    cs = ax.contour(THETA, PHI, af_db, levels=levels, colors='k',
                    linewidths=0.3, alpha=0.4)
    # 标记 -3 dB 等值线
    cs3 = ax.contour(THETA, PHI, af_db, levels=[-3], colors='white',
                     linewidths=1.0)
    ax.clabel(cs3, fmt='-3 dB', fontsize=8)
    plt.colorbar(cf, ax=ax, label='dB')
    ax.set_xlabel('φ (度)')
    ax.set_ylabel('θ (度)')
    ax.set_title(title)


def plot_pattern_3d(ax, theta_deg: np.ndarray, phi_deg: np.ndarray,
                    af_db: np.ndarray,
                    title: str = "3D 方向图",
                    db_min: float = -40):
    """平面阵 3D 曲面方向图（球坐标 r = dB 值 + offset）。

    Args:
        ax: matplotlib 3D Axes (projection='3d')
        theta_deg: θ 角度网格（度），shape (Nθ,)
        phi_deg: φ 角度网格（度），shape (Nφ,)
        af_db: 归一化 dB 方向图，shape (Nθ, Nφ)
        title: 图标题
        db_min: dB 下限，低于此值的截断
    """
    THETA, PHI = np.meshgrid(phi_deg, theta_deg)  # (Nθ, Nφ) meshgrids

    # 球坐标转笛卡尔：r = max(0, af_db - db_min)
    r = np.maximum(af_db - db_min, 0)
    x = r * np.cos(np.deg2rad(THETA)) * np.cos(np.deg2rad(PHI))  # 注意：THETA=θ, PHI=φ
    y = r * np.cos(np.deg2rad(THETA)) * np.sin(np.deg2rad(PHI))
    z = r * np.sin(np.deg2rad(THETA))

    # 用 dB 值着色
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet((af_db - db_min) / (-db_min)),
                    rstride=1, cstride=1, alpha=0.85, linewidth=0,
                    antialiased=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    # 等轴缩放
    max_range = np.array([x.max(), y.max(), z.max()]).max() * 0.6
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range * 2)


def plot_pattern_with_lobes(
    theta_deg: np.ndarray,
    af_db: np.ndarray,
    psll_val: float,
    psll_angle: float,
    target_angle: float,
    all_vals: np.ndarray,
    all_angles: np.ndarray,
    freq_ghz: float = None,
    theta0_deg: float = None,
):
    """线阵方向图，标注主瓣/副瓣/3dB 线。

    Args:
        theta_deg: θ 角度网格
        af_db: 归一化 dB 方向图
        psll_val, psll_angle: PSLL 值及角度
        target_angle: 指向角 (度)
        all_vals, all_angles: 所有峰值 (值, 角度)
        freq_ghz: 频率 (GHz), 仅用于标题
        theta0_deg: 扫描角 (度), 仅用于标题
    """
    # 主瓣: 最接近 0 dB 且最接近指向角的峰值
    near_zero = np.abs(all_vals) < 1e-6
    if near_zero.any():
        cand = np.where(near_zero)[0]
        best = cand[np.argmin(np.abs(all_angles[cand] - target_angle))]
        main_val, main_ang = all_vals[best], all_angles[best]
    else:
        main_val, main_ang = all_vals[0], all_angles[0]

    # 副瓣: 排除主瓣候选后的最高峰
    if near_zero.any():
        mask = ~near_zero
        sll_v = all_vals[mask]
        sll_a = all_angles[mask]
    else:
        sll_v, sll_a = all_vals[1:], all_angles[1:]
    sll_val = sll_v[0] if len(sll_v) > 0 else -np.inf
    sll_ang = sll_a[0] if len(sll_v) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(theta_deg, af_db, linewidth=1.0, color='steelblue')
    ax.scatter([main_ang], [main_val], c='green', s=80, marker='s', zorder=6,
               label=f'Main lobe: {main_val:.4f} dB @ {main_ang:.2f}$^\\circ$')
    if not np.isinf(sll_val):
        ax.scatter([sll_ang], [sll_val], c='red', s=80, marker='D', zorder=6,
                   label=f'Side lobe: {sll_val:.4f} dB @ {sll_ang:.2f}$^\\circ$')
    ax.axhline(y=-3, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    parts = []
    if freq_ghz is not None:
        parts.append(f"f={freq_ghz:.4g} GHz")
    if theta0_deg is not None:
        parts.append(f"$\\theta_0$={theta0_deg:.1f}$^\\circ$")
    parts.append(f"PSLL={psll_val:.2f} dB")
    ax.set_title("Radiation Pattern  (" + ", ".join(parts) + ")")
    ax.set_xlabel("$\\theta$ (deg)")
    ax.set_ylabel("Normalized Pattern (dB)")
    ax.set_ylim(-60, 3)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
