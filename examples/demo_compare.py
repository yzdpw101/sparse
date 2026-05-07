"""多算法对比 demo — 同一稀布线阵问题，不同优化器 PK。

可修改下方参数直接运行。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from antopt import LMMapper, Pattern, run_optimization

# ═══════════════════════════════════════════════════════════
#  调参区域
# ═══════════════════════════════════════════════════════════

Ne = 78                   # 阵元数
L = 73                 # 孔径 (λ)
dmin = 0.5                # 最小间距 (λ)
SYMMETRIC = True         # 对称
FIXED_APERTURE = False   # 固定孔径 (已废弃，一般不开启)
THETA_STEP = 0.1          # θ 步长（度）

POP_SIZE = 50             # 种群大小
MAX_ITER = 50            # 最大迭代次数
SEED = 41                 # 随机种子
N_JOBS = -1               # 并行线程数, -1=全部CPU, 1=单线程

# 要对比的算法: (method, 显示名)
METHODS = [
    ("cma",     "pycma CMA-ES"),
    ("ngopt",   "nevergrad NGOpt"),
    ("cma_ng",  "nevergrad CMA"),
    ("de",      "nevergrad DE"),
]

# ═══════════════════════════════════════════════════════════
#  运行
# ═══════════════════════════════════════════════════════════

mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=SYMMETRIC, is_fixed_aperture=FIXED_APERTURE)
pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=THETA_STEP)

print(f"阵列: {Ne} 元, 孔径 {L}λ, dmin={dmin}λ, 对称={SYMMETRIC}")
print(f"变量维度: {mapper.n_vars}, 种群: {POP_SIZE}, 迭代: {MAX_ITER}")
print(f"{'='*70}")

results = {}
for method, name in METHODS:
    t0 = time.perf_counter()
    r = run_optimization(
        mapper, pat,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        seed=SEED,
        verbose=False,
        method=method,
        n_jobs=N_JOBS,
    )
    elapsed = time.perf_counter() - t0
    results[method] = (r, elapsed)

# ═══════════════════════════════════════════════════════════
#  汇总
# ═══════════════════════════════════════════════════════════

print(f"\n{'算法':<20s}  {'PSLL (dB)':>10s}  {'耗时':>8s}")
print(f"{'-'*42}")
for method, name in METHODS:
    r, t = results[method]
    print(f"{name:<20s}  {r['f']:>10.4f}  {t:>7.1f}s")
print(f"{'='*42}")

# 打印最优解的间距分布
best = min(results.values(), key=lambda v: v[0]['f'])[0]
pos = best["result"]["positions"]
print(f"\n最优解 ({best['method']}):")
print(f"  位置: {np.array2string(pos, precision=3, max_line_width=100)}")
print(f"  间距: {np.array2string(np.diff(pos), precision=3, max_line_width=100)}")
