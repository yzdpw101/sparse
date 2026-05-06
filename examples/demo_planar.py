"""稀布平面阵 demo：计算并可视化 2D/3D 方向图 + PSLL 分析。

可修改下方参数直接运行。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from core.pattern import Pattern
from core.analysis import get_psll, find_peaks, get_overall_psll
from visualization import plot_pattern_1d, plot_pattern_2d, plot_pattern_3d

# ═══════════════════════════════════════════════════════════
#  调参区域
# ═══════════════════════════════════════════════════════════

N_X = 4                  # x 方向阵元数
N_Y = 4                  # y 方向阵元数
DX = 0.5                 # x 方向间距（波长单位）
DY = 0.5                 # y 方向间距（波长单位）
SCAN_THETA = 0.0         # 波束指向俯仰角（度）
SCAN_PHI = 0.0           # 波束指向方位角（度）
THETA_START = -90        # θ 起始（度）
THETA_END = 90           # θ 终止（度）
THETA_STEP = 2.0         # θ 步长（度）
PHI_START = -90          # φ 起始（度）
PHI_END = 90             # φ 终止（度）
PHI_STEP = 2.0           # φ 步长（度）

# 对称模式: True=四象限对称（只传第一象限位置）
SYMMETRIC = False

# ═══════════════════════════════════════════════════════════
#  构造阵列
# ═══════════════════════════════════════════════════════════

if SYMMETRIC:
    # 四象限对称：只传第一象限
    qx = np.arange(DX, (N_X // 2) * DX + DX / 2, DX)
    qy = np.arange(DY, (N_Y // 2) * DY + DY / 2, DY)
    QX, QY = np.meshgrid(qx, qy)
    wl_x, wl_y = QX.ravel(), QY.ravel()
    has_center = (N_X % 2 == 1) and (N_Y % 2 == 1)
    print(f"第一象限阵元数: {len(wl_x)} （对称 → {len(wl_x) * 4 + (1 if has_center else 0)} 元）")
else:
    # 非对称：全部阵元
    px = np.arange(N_X) * DX - (N_X - 1) * DX / 2
    py = np.arange(N_Y) * DY - (N_Y - 1) * DY / 2
    PX, PY = np.meshgrid(px, py)
    wl_x, wl_y = PX.ravel(), PY.ravel()
    has_center = False
    print(f"阵元数: {len(wl_x)}")

print(f"阵列孔径: {np.ptp(px):.2f} λ × {np.ptp(py):.2f} λ")
print(f"扫描方向: (theta0={SCAN_THETA} deg, phi0={SCAN_PHI} deg)")

# ═══════════════════════════════════════════════════════════
#  方向图计算
# ═══════════════════════════════════════════════════════════

pat = Pattern(
    array_type="planar",
    symmetric=SYMMETRIC,
    theta_deg_start=THETA_START,
    theta_deg_end=THETA_END,
    theta_deg_step=THETA_STEP,
    theta0s_deg=SCAN_THETA,
    phi_deg_start=PHI_START,
    phi_deg_end=PHI_END,
    phi_deg_step=PHI_STEP,
    phi0s_deg=SCAN_PHI,
)

if SYMMETRIC:
    af = pat.planar_af_symmetric(wl_x, wl_y, has_center=has_center)
else:
    af = pat.planar_af(wl_x, wl_y)
af_db = Pattern.to_dB(af)

# ═══════════════════════════════════════════════════════════
#  PSLL 分析
# ═══════════════════════════════════════════════════════════

# 主瓣排除区域（扫描角附近）
mr_theta = (SCAN_THETA - 15, SCAN_THETA + 15)
mr_phi = (SCAN_PHI - 20, SCAN_PHI + 20) if SCAN_PHI != 0 else (-20, 20)
mainlobe_region = (mr_theta, mr_phi)

# 各 φ 平面 PSLL
pslls, coord2d = get_psll(af_db, pat.theta_deg, pat.phi_deg,
                           mainlobe_region=mainlobe_region)
worst_j = int(np.nanargmax(pslls))
print(f"\n--- PSLL 分析 ---")
print(f"最差 φ 平面: φ={pat.phi_deg[worst_j]:.1f}°, PSLL={pslls[worst_j]:.4f} dB "
      f"@ θ={coord2d[worst_j, 0]:.2f}°")

# 全平面 PSLL
overall_psll, overall_coord = get_overall_psll(
    af_db, pat.theta_deg, pat.phi_deg, mainlobe_region=mainlobe_region)
print(f"全平面 PSLL: {overall_psll:.4f} dB "
      f"@ (θ={overall_coord[0]:.2f}°, φ={overall_coord[1]:.2f}°)")

# φ=SCAN_PHI 平面的 1D 切片
j_scan = np.argmin(np.abs(pat.phi_deg - SCAN_PHI))
psll_scan, theta_scan = get_psll(
    af_db[:, j_scan], pat.theta_deg, mainlobe_region=mr_theta)
print(f"φ={pat.phi_deg[j_scan]:.1f}° 平面 PSLL: {psll_scan:.4f} dB")

# φ=0° 平面
j_zero = np.argmin(np.abs(pat.phi_deg))
psll_zero, theta_zero = get_psll(
    af_db[:, j_zero], pat.theta_deg, mainlobe_region=mr_theta)
print(f"φ={pat.phi_deg[j_zero]:.1f}° 平面 PSLL: {psll_zero:.4f} dB")

# ═══════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 13))
title_prefix = f'{N_X}×{N_Y} 均匀平面阵 ({DX}λ×{DY}λ'
title_prefix += f', 扫描={SCAN_THETA}°/{SCAN_PHI}°'
title_prefix += ', 对称)' if SYMMETRIC else ')'

# ── (a) 2D heatmap ──
ax1 = fig.add_subplot(2, 3, 1)
plot_pattern_2d(ax1, pat.theta_deg, pat.phi_deg, af_db,
                title=f"2D 方向图\n{title_prefix}")

# ── (b) 3D 曲面 ──
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
plot_pattern_3d(ax2, pat.theta_deg, pat.phi_deg, af_db,
                title=f"3D 方向图\n{title_prefix}")

# ── (c) φ=0° 切片 1D ──
ax3 = fig.add_subplot(2, 3, 3)
vals, angs = find_peaks(af_db[:, j_zero], pat.theta_deg)
plot_pattern_1d(ax3, pat.theta_deg, af_db[:, j_zero],
                peaks=(vals, angs),
                psll=(psll_zero, theta_zero),
                title=f"φ=0° 切片 1D 方向图")

# ── (d) φ=SCAN_PHI° 切片 1D ──
ax4 = fig.add_subplot(2, 3, 4)
vals_s, angs_s = find_peaks(af_db[:, j_scan], pat.theta_deg)
plot_pattern_1d(ax4, pat.theta_deg, af_db[:, j_scan],
                peaks=(vals_s, angs_s),
                psll=(psll_scan, theta_scan),
                title=f"φ={SCAN_PHI}° 切片 1D 方向图")

# ── (e) 各 φ 平面 PSLL ──
ax5 = fig.add_subplot(2, 3, 5)
valid = ~np.isinf(pslls)
ax5.plot(pat.phi_deg[valid], pslls[valid], 'b.-', linewidth=1.2)
ax5.axhline(y=overall_psll, color='orange', linestyle='--',
            label=f'全平面 PSLL: {overall_psll:.2f} dB')
ax5.scatter([overall_coord[1]], [overall_psll], c='orange', s=80,
            marker='D', zorder=5)
ax5.set_xlabel('φ (度)')
ax5.set_ylabel('PSLL (dB)')
ax5.set_title('各 φ 平面 PSLL')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ── (f) 阵元布局 ──
ax6 = fig.add_subplot(2, 3, 6)
ax6.scatter(wl_x, wl_y, c='blue', s=80, marker='s', zorder=5)
if not SYMMETRIC:
    ax6.scatter([0], [0], c='red', s=30, marker='x', alpha=0.5)  # 原点标记
ax6.set_xlabel('x (λ)')
ax6.set_ylabel('y (λ)')
ax6.set_title('阵元布局')
ax6.axis('equal')
ax6.grid(True, alpha=0.3)
# 标注孔径
x_min, x_max = np.min(wl_x), np.max(wl_x)
y_min, y_max = np.min(wl_y), np.max(wl_y)
margin = 0.3
ax6.set_xlim(x_min - margin, x_max + margin)
ax6.set_ylim(y_min - margin, y_max + margin)

plt.suptitle(title_prefix, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
