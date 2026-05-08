"""计算对比：对称/非对称、多频/单频、多角度/单角度。

结论：结果一致，加速比 ≈ 对称性因子（线阵 2×、平面阵 4×），
      多频/多角度 ≈ 循环次数 × 单次。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from antopt.pattern import Pattern

# ═══════════════════════════════════════════════════════════
#  固定参数
# ═══════════════════════════════════════════════════════════

THETA_STEP = 0.1
PHI_START, PHI_END, PHI_STEP = 0, 180, 1.0
THETA0, PHI0 = 0.0, 0.0
FREQS = np.array([1.0, 2.0])
THETA0S = np.array([0.0, 30.0, 60.0])

# 线阵: 100 元, d=0.5λ₀
N_LINEAR, D_LINEAR = 100, 0.5
wl_lin_full = np.linspace(-(N_LINEAR - 1) * D_LINEAR / 2,
                           (N_LINEAR - 1) * D_LINEAR / 2, N_LINEAR)
wl_lin_half = np.arange(D_LINEAR / 2, (N_LINEAR / 2) * D_LINEAR, D_LINEAR)

# 平面阵: 10×10, dx=dy=0.5λ₀
N_PLANAR, D_PLANAR = 10, 0.5
px = np.linspace(-(N_PLANAR - 1) * D_PLANAR / 2,
                  (N_PLANAR - 1) * D_PLANAR / 2, N_PLANAR)
py = np.linspace(-(N_PLANAR - 1) * D_PLANAR / 2,
                  (N_PLANAR - 1) * D_PLANAR / 2, N_PLANAR)
PX, PY = np.meshgrid(px, py)
wl_pl_full_x, wl_pl_full_y = PX.ravel(), PY.ravel()

# 第一象限 (x>0, y>0)
qx = np.arange(D_PLANAR / 2, (N_PLANAR / 2) * D_PLANAR, D_PLANAR)
qy = np.arange(D_PLANAR / 2, (N_PLANAR / 2) * D_PLANAR, D_PLANAR)
QX, QY = np.meshgrid(qx, qy)
wl_pl_quarter_x, wl_pl_quarter_y = QX.ravel(), QY.ravel()

np.random.seed(42)
AMP_LIN = None  # 均匀
AMP_PL = None

N_WARMUP = 3
N_REPEAT = 10


def timeit(fn, *args, **kwargs):
    """运行 N_REPEAT 次取平均时间 (ms)。先 warm-up N_WARMUP 次。"""
    for _ in range(N_WARMUP):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0) / N_REPEAT * 1000


# ═══════════════════════════════════════════════════════════
#  1. 对称 vs 非对称
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("1. 对称 vs 非对称")
print("=" * 70)

# ── 线阵 ──
pat_lin = Pattern(theta_deg_step=THETA_STEP)
pat_lin_sym = Pattern(symmetric=True, theta_deg_step=THETA_STEP)

t_lin_asym = timeit(pat_lin.linear_af, wl_lin_full, AMP_LIN)
t_lin_sym = timeit(pat_lin_sym.linear_af_symmetric, wl_lin_half, False, AMP_LIN)

af_lin_asym = pat_lin.linear_af(wl_lin_full, AMP_LIN)
af_lin_sym = pat_lin_sym.linear_af_symmetric(wl_lin_half, False, AMP_LIN)
lin_sym_ok = np.allclose(af_lin_asym, af_lin_sym)

print(f"  线阵 100 元 非对称: {t_lin_asym:.2f} ms")
print(f"  线阵 100 元 对称:   {t_lin_sym:.2f} ms  (加速比 {t_lin_asym / t_lin_sym:.2f}×)")
print(f"  结果一致: {lin_sym_ok}")

# ── 平面阵 ──
pat_pl = Pattern(array_type="planar",
                 theta_deg_step=THETA_STEP,
                 phi_deg_start=PHI_START, phi_deg_end=PHI_END, phi_deg_step=PHI_STEP)
pat_pl_sym = Pattern(array_type="planar", symmetric=True,
                     theta_deg_step=THETA_STEP,
                     phi_deg_start=PHI_START, phi_deg_end=PHI_END, phi_deg_step=PHI_STEP)

t_pl_asym = timeit(pat_pl.planar_af, wl_pl_full_x, wl_pl_full_y, AMP_PL)
t_pl_sym = timeit(pat_pl_sym.planar_af_symmetric,
                  wl_pl_quarter_x, wl_pl_quarter_y, False, AMP_PL)

af_pl_asym = pat_pl.planar_af(wl_pl_full_x, wl_pl_full_y, AMP_PL)
af_pl_sym = pat_pl_sym.planar_af_symmetric(wl_pl_quarter_x, wl_pl_quarter_y, False, AMP_PL)
pl_sym_ok = np.allclose(af_pl_asym, af_pl_sym)

print(f"\n  平面阵 10×10 非对称: {t_pl_asym:.2f} ms")
print(f"  平面阵 10×10 对称:   {t_pl_sym:.2f} ms  (加速比 {t_pl_asym / t_pl_sym:.2f}×)")
print(f"  结果一致: {pl_sym_ok}")


# ═══════════════════════════════════════════════════════════
#  2. 多频 vs 单频重复
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("2. 多频 vs 单频重复  (f0=1 GHz, f1=2 GHz → k=2π, 4π)")
print("=" * 70)

# ── 线阵多频 ──
pat_lin_mf = Pattern(frequenciesGHz=FREQS, theta_deg_step=THETA_STEP)
t_lin_mf = timeit(pat_lin_mf.linear_af, wl_lin_full, AMP_LIN)
af_lin_mf = pat_lin_mf.linear_af(wl_lin_full, AMP_LIN)

# 单频重复 — 位置按 f/f₀ 缩放
t_lin_sf_total = 0
af_lin_sf_list = []
for i, fGHz in enumerate(FREQS):
    scale = fGHz / FREQS[0]
    pat_sf = Pattern(theta_deg_step=THETA_STEP)  # k=2π
    t_sf = timeit(pat_sf.linear_af, wl_lin_full * scale, AMP_LIN)
    t_lin_sf_total += t_sf
    af_lin_sf_list.append(pat_sf.linear_af(wl_lin_full * scale, AMP_LIN))

lin_mf_ok = True
for i in range(len(FREQS)):
    if not np.allclose(af_lin_mf[i], af_lin_sf_list[i]):
        lin_mf_ok = False
        break

print(f"  线阵 多频 (一次调用):       {t_lin_mf:.2f} ms")
print(f"  线阵 单频重复 ({len(FREQS)} 次, 位置缩放): {t_lin_sf_total:.2f} ms")
print(f"  结果一致: {lin_mf_ok}")

# ── 平面阵多频 ──
pat_pl_mf = Pattern(array_type="planar", frequenciesGHz=FREQS,
                    theta_deg_step=THETA_STEP,
                    phi_deg_start=PHI_START, phi_deg_end=PHI_END, phi_deg_step=PHI_STEP)
t_pl_mf = timeit(pat_pl_mf.planar_af, wl_pl_full_x, wl_pl_full_y, AMP_PL)
af_pl_mf = pat_pl_mf.planar_af(wl_pl_full_x, wl_pl_full_y, AMP_PL)

t_pl_sf_total = 0
af_pl_sf_list = []
for i, fGHz in enumerate(FREQS):
    scale = fGHz / FREQS[0]
    pat_sf = Pattern(array_type="planar",
                     theta_deg_step=THETA_STEP,
                     phi_deg_start=PHI_START, phi_deg_end=PHI_END, phi_deg_step=PHI_STEP)
    t_sf = timeit(pat_sf.planar_af, wl_pl_full_x * scale, wl_pl_full_y * scale, AMP_PL)
    t_pl_sf_total += t_sf
    af_pl_sf_list.append(pat_sf.planar_af(wl_pl_full_x * scale, wl_pl_full_y * scale, AMP_PL))

pl_mf_ok = True
for i in range(len(FREQS)):
    if not np.allclose(af_pl_mf[i], af_pl_sf_list[i]):
        pl_mf_ok = False
        break

print(f"\n  平面阵 多频 (一次调用):       {t_pl_mf:.2f} ms")
print(f"  平面阵 单频重复 ({len(FREQS)} 次, 位置缩放): {t_pl_sf_total:.2f} ms")
print(f"  结果一致: {pl_mf_ok}")


# ═══════════════════════════════════════════════════════════
#  3. 多角度 vs 单角度重复
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"3. 多角度 vs 单角度重复  (theta0 = {THETA0S} deg)")
print("=" * 70)

# ── 线阵多角度 ──
pat_lin_ma = Pattern(theta0s_deg=THETA0S, theta_deg_step=THETA_STEP)
t_lin_ma = timeit(pat_lin_ma.linear_af, wl_lin_full, AMP_LIN)
af_lin_ma = pat_lin_ma.linear_af(wl_lin_full, AMP_LIN)

t_lin_sa_total = 0
af_lin_sa_list = []
for th0 in THETA0S:
    pat_sa = Pattern(theta0s_deg=th0, theta_deg_step=THETA_STEP)
    t_sa = timeit(pat_sa.linear_af, wl_lin_full, AMP_LIN)
    t_lin_sa_total += t_sa
    af_lin_sa_list.append(pat_sa.linear_af(wl_lin_full, AMP_LIN))

lin_ma_ok = True
for i in range(len(THETA0S)):
    if not np.allclose(af_lin_ma[i], af_lin_sa_list[i]):
        lin_ma_ok = False
        break

print(f"  线阵 多角度 (一次调用):       {t_lin_ma:.2f} ms")
print(f"  线阵 单角度重复 ({len(THETA0S)} 次):        {t_lin_sa_total:.2f} ms")
print(f"  结果一致: {lin_ma_ok}")

# ── 平面阵多角度 ──
pat_pl_ma = Pattern(array_type="planar", theta0s_deg=THETA0S,
                    theta_deg_step=THETA_STEP,
                    phi_deg_start=PHI_START, phi_deg_end=PHI_END, phi_deg_step=PHI_STEP)
t_pl_ma = timeit(pat_pl_ma.planar_af, wl_pl_full_x, wl_pl_full_y, AMP_PL)
af_pl_ma = pat_pl_ma.planar_af(wl_pl_full_x, wl_pl_full_y, AMP_PL)

t_pl_sa_total = 0
af_pl_sa_list = []
for th0 in THETA0S:
    pat_sa = Pattern(array_type="planar", theta0s_deg=th0,
                     theta_deg_step=THETA_STEP,
                     phi_deg_start=PHI_START, phi_deg_end=PHI_END, phi_deg_step=PHI_STEP)
    t_sa = timeit(pat_sa.planar_af, wl_pl_full_x, wl_pl_full_y, AMP_PL)
    t_pl_sa_total += t_sa
    af_pl_sa_list.append(pat_sa.planar_af(wl_pl_full_x, wl_pl_full_y, AMP_PL))

pl_ma_ok = True
for i in range(len(THETA0S)):
    if not np.allclose(af_pl_ma[i], af_pl_sa_list[i]):
        pl_ma_ok = False
        break

print(f"\n  平面阵 多角度 (一次调用):       {t_pl_ma:.2f} ms")
print(f"  平面阵 单角度重复 ({len(THETA0S)} 次):        {t_pl_sa_total:.2f} ms")
print(f"  结果一致: {pl_ma_ok}")


# ═══════════════════════════════════════════════════════════
#  汇总
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("汇总")
print("=" * 70)
print(f"  线阵对称加速比:     {t_lin_asym / t_lin_sym:.2f}×  (理论 ~2×)")
print(f"  平面阵对称加速比:   {t_pl_asym / t_pl_sym:.2f}×   (理论 ~4×)")
print(f"  线阵多频一致性:     {lin_mf_ok}")
print(f"  平面阵多频一致性:   {pl_mf_ok}")
print(f"  线阵多角度一致性:   {lin_ma_ok}")
print(f"  平面阵多角度一致性: {pl_ma_ok}")
