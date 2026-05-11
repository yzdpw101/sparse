"""optimizer2 CMA — 132元对称固定孔径稀布线阵。

可调超参数: sigma, pop_size, max_iter 等。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from antopt import LMMapper, Pattern, get_psll
from antopt.optimizer import minimize

# ════════════ 可调参数 ════════════
Ne, L, dmin = 132, 90.5, 0.5
SIGMA = 0.5         # CMA 初始步长
MAX_ITER = 10000       # 最大迭代
POP_SIZE = False      # None=自动 (2×(4+3lnD))
SEED = 42
# ══════════════════════════════════

mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=True,
                   is_fixed_aperture=True, use_sigmoid=False)
pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=0.1)
hN = mapper._halfNe


def fitness(x):
    pos = mapper.synthesize(x)
    af = pat.linear_af_symmetric(pos[hN:], has_center=mapper.has_center)
    af_db = 20 * np.log10(np.abs(af) / np.max(np.abs(af)))
    psll, _ = get_psll(af_db, pat.theta_deg)
    return float(psll)


print(f"CMA: {Ne}元 {L}λ dmin={dmin}λ, n_vars={mapper.n_vars}")
print(f"sigma={SIGMA}, max_iter={MAX_ITER}, seed={SEED}")
print("=" * 50)

t0 = time.perf_counter()
result = minimize(
    fitness, mapper.n_vars,
    method="cma",
    bounds=(0.0, 1.0),
    sigma=SIGMA,                # ← 可调超参数
    max_iter=MAX_ITER,
    pop_size=POP_SIZE,
    n_jobs=-1,
    seed=SEED,
    verbose=True,
    init="chaos",
    stop_fitness=-30
)
elapsed = time.perf_counter() - t0

pos = mapper.synthesize(result["x"])
diffs = np.diff(pos)
print(f"\nPSLL = {result['f']:.4f} dB  ({elapsed:.0f}s)")
print(f"dmin={diffs.min():.4f}  dmax={diffs.max():.4f}  "
      f"孔径=[{pos[0]:.4f}, {pos[-1]:.4f}]")
