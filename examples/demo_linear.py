"""稀布阵列天线 demo：计算线阵方向图并分析 PSLL。

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
from core.analysis import get_psll, find_peaks
from visualization import plot_pattern_1d, plot_pattern_polar

# ═══════════════════════════════════════════════════════════
#  调参区域
# ═══════════════════════════════════════════════════════════

N_ELEMENTS = 10          # 阵元数
SPACING = 0.5            # 阵元间距（波长单位）
SCAN_ANGLE = 0.0         # 波束指向（度），0=法向
THETA_START = -90        # 起始角度（度）
THETA_END = 90           # 终止角度（度）
THETA_STEP = 0.1         # 角度步长（度）

# 对称模式: True=对称阵列（只传半边位置），False=非对称（传全部位置）
SYMMETRIC = True

# 幅度加权 (None=均匀, 或传入长度为 N_ELEMENTS 的数组)
AMPLITUDES = None

# ═══════════════════════════════════════════════════════════
#  计算
# ═══════════════════════════════════════════════════════════

pat = Pattern(
    symmetric=SYMMETRIC,
    theta_deg_start=THETA_START,
    theta_deg_end=THETA_END,
    theta_deg_step=THETA_STEP,
    theta0s_deg=SCAN_ANGLE,
)

if SYMMETRIC:
    # 对称模式：只传半边位置（x > 0）
    half_positions = np.arange(SPACING, (N_ELEMENTS // 2) * SPACING + SPACING / 2, SPACING)
    print(f"半边阵元位置: {half_positions}")
    print(f"孔径: {2 * half_positions[-1]:.4f} λ（对称）")
    af = pat.af(half_positions, amplitudes=AMPLITUDES)
else:
    positions = np.linspace(-(N_ELEMENTS - 1) * SPACING / 2,
                            (N_ELEMENTS - 1) * SPACING / 2, N_ELEMENTS)
    print(f"阵元位置: {positions}")
    print(f"孔径: {positions[-1] - positions[0]:.4f} λ")
    af = pat.af(positions, amplitudes=AMPLITUDES)
af_db = Pattern.to_dB(af)

psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
print(f"\nPSLL: {psll_val:.4f} dB @ {psll_angle:.4f}°")

values, angles = find_peaks(af_db, pat.theta_deg)
print(f"检测到 {len(values)} 个峰值:")
for i, (v, a) in enumerate(zip(values[:8], angles[:8])):
    tag = "主瓣" if i == 0 else f"副瓣 #{i}"
    print(f"  {tag}: {v:.4f} dB @ {a:.4f}°")
if len(values) > 8:
    print(f"  ... (共 {len(values)} 个)")

# ═══════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 2, 1)
title = f'{N_ELEMENTS} 元均匀直线阵 (d={SPACING}λ, 扫描={SCAN_ANGLE}°'
title += ', 对称)' if SYMMETRIC else ')'
plot_pattern_1d(ax1, pat.theta_deg, af_db,
                peaks=(values, angles),
                psll=(psll_val, psll_angle),
                title=title)

ax2 = fig.add_subplot(1, 2, 2, projection='polar')
plot_pattern_polar(ax2, pat.theta_deg, af_db)

plt.tight_layout()
plt.show()
