"""有约束 vs 无约束 — 非对称固定孔径线阵, 78元, 波束指向30°。

对比 bounded [0,1] (去sigmoid) vs unbounded (sigmoid)。
pop_size=60, max_iter=10000, MC=5 种子。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from antopt import LMMapper, Pattern, get_psll
from antopt.optimizer import minimize

Ne, L, dmin = 78, 73.0, 0.5
POP = 60; MAX_ITER = 10000; MC = 5


def run(label, bounded):
    use_sig = not bounded
    bounds = (0.0, 1.0) if bounded else None
    mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=False,
                       is_fixed_aperture=True, use_sigmoid=use_sig)
    pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1,
                  theta0s_deg=np.array([30.0]))

    def fitness(x):
        pos = mapper.synthesize(x)
        af = pat.linear_af(pos)
        af_db = 20 * np.log10(np.abs(af) / np.max(np.abs(af)))
        psll, _ = get_psll(af_db, pat.theta_deg)
        return float(psll)

    print(f"\n--- {label} (n_vars={mapper.n_vars}, bounded={bounded}) ---")
    best_all = np.inf
    t0 = time.perf_counter()
    for seed in range(1, MC + 1):
        r = minimize(fitness, mapper.n_vars, method="cma", bounds=bounds,
                     pop_size=POP, max_iter=MAX_ITER, seed=seed, n_jobs=-1,
                     verbose=2000, init="chaos", sigma=0.5)
        f = r["f"]
        if f < best_all: best_all = f
        print(f"  seed={seed}: PSLL={f:.4f}  best={best_all:.4f}", flush=True)
    elapsed = time.perf_counter() - t0
    return best_all, elapsed


print(f"非对称固定孔径: {Ne}元 {L}λ dmin={dmin}λ, theta0=30°")
print(f"pop={POP}, max_iter={MAX_ITER}, MC={MC}")
print("=" * 55)

for bounded, name in [(True, "bounded[0,1]"), (False, "unbounded(sigmoid)")]:
    f, t = run(name, bounded)
    print(f"  => {name}: best={f:.4f} dB ({t:.0f}s)")
