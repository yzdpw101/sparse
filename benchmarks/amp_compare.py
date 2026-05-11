"""仅优化幅度: 无约束sigmoid vs 有约束[0,1] 对比。

50元对称均匀线阵, dmin=0.5λ, 幅度范围[0,1], MC=5。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time, numpy as np
from antopt import Pattern, get_psll
from antopt.optimizer import minimize

Ne, L, dmin = 50, 24.5, 0.5
MC, N_JOBS = 5, -1
MAX_ITER = 1000
AMP_LO, AMP_HI = 0.0, 1.0

# 50元对称均匀位置 (偶数, 无中心)
halfNe = Ne // 2
hpos = np.arange(0.25, halfNe * dmin, dmin)
pos = np.concatenate([-hpos[::-1], hpos])
pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)

print(f"幅度优化对比: {Ne}元对称线阵, 孔径={L}λ, dmin={dmin}λ")
print(f"MC={MC}, max_iter={MAX_ITER}")
print("=" * 55)


def run(label, bounded):
    """bounded=True → [0,1]约束无sigmoid; False → ℝ无约束+sigmoid"""
    sig = 0.5 if bounded else 1.0
    bnd = (AMP_LO, AMP_HI) if bounded else None

    def fitness(x):
        if not bounded:
            x = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))  # sigmoid → (0,1)
        amps = np.clip(x, AMP_LO, AMP_HI)
        af = pat.linear_af_symmetric(hpos, has_center=False, amplitudes=amps)
        af_db = 20 * np.log10(np.abs(af) / np.max(np.abs(af)))
        psll, _ = get_psll(af_db, pat.theta_deg)
        return float(psll)

    best_all = np.inf
    t0 = time.perf_counter()
    for seed in range(1, MC + 1):
        r = minimize(fitness, halfNe, method="cma", bounds=bnd,
                     sigma=sig, max_iter=MAX_ITER, seed=seed,
                     n_jobs=N_JOBS, verbose=200, init="chaos")
        if r["f"] < best_all: best_all = r["f"]
        print(f"  [{label}] seed={seed}: PSLL={r['f']:.4f}  best={best_all:.4f}", flush=True)
    elapsed = time.perf_counter() - t0
    return best_all, elapsed


for bounded, name in [(True, "bounded[0,1]"), (False, "unbounded+sigmoid")]:
    f, t = run(name, bounded)
    print(f"  => {name}: best={f:.4f} dB ({t:.0f}s)\n")
