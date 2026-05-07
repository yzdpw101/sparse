"""幅相优化 demo — 测试 mode=[100, 110, 111, 120] 四种模式。

mode 编码:
  个位=位置: 0=导入, 1=优化
  十位=相位: 0=等相位, 1=优化, 2=导入
  百位=幅度: 0=等幅度, 1=优化, 2=导入
  例: 100=纯稀布, 110=稀布+相位, 111=全优化, 120=稀布+导入幅度
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from antopt import LMMapper, Pattern, run_optimization

# ═══════════════════════════════════════════════════════════
#  调参区域
# ═══════════════════════════════════════════════════════════

Ne, L, dmin = 17, 9.744, 0.5
SYMMETRIC = False
THETA_STEP = 0.5
POP_SIZE, MAX_ITER, SEED = 30, 100, 42
N_JOBS = -1

# 要测试的 mode
MODES = [
    (100, "纯稀布"),
    (110, "稀布 + 相位优化"),
    (111, "稀布 + 相位 + 幅度"),
    (120, "稀布 + 导入幅度"),
]

# JSON 输入文件目录
INPUT_DIR = Path(__file__).resolve().parent.parent / "input"


def load_json(filename):
    """加载 JSON 文件，返回对应数组。"""
    path = INPUT_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return np.array(data, dtype=float)


# ═══════════════════════════════════════════════════════════
#  运行
# ═══════════════════════════════════════════════════════════

mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=SYMMETRIC)
pat = Pattern(theta_deg_step=THETA_STEP)

# 导入数据（如果文件存在）
init_pos = load_json("positions.json")
init_phs = load_json("phases.json")
init_amp = load_json("amplitudes.json")

print(f"阵列: {Ne} 元, 孔径 {L}λ, dmin={dmin}λ, 对称={SYMMETRIC}")
print(f"变量基准: pos={mapper.n_vars}, Ne={Ne}")
print(f"导入数据: pos={'有' if init_pos is not None else '无'}, "
      f"phase={'有' if init_phs is not None else '无'}, "
      f"amp={'有' if init_amp is not None else '无'}")
print(f"{'='*55}")

results = {}
for mode, desc in MODES:
    t0 = time.perf_counter()
    r = run_optimization(
        mapper, pat, mode=mode,
        init_positions=init_pos, init_phases_deg=init_phs, init_amplitudes=init_amp,
        pop_size=POP_SIZE, max_iter=MAX_ITER, seed=SEED,
        verbose=False, n_jobs=N_JOBS,
    )
    t = time.perf_counter() - t0
    results[mode] = (r, t)

print(f"\n{'mode':>5s}  {'模式':<20s}  {'PSLL (dB)':>10s}  {'耗时':>7s}  {'n_vars':>6s}")
print(f"{'-'*53}")
for mode, desc in MODES:
    r, t = results[mode]
    nv = ((mode // 100) % 10 == 1) * mapper.n_vars + ((mode // 10) % 10 == 1) * Ne + (mode % 10 == 1) * Ne
    print(f"{mode:>5d}  {desc:<20s}  {r['f']:>10.4f}  {t:>6.1f}s  {nv:>6d}")

print(f"{'='*55}")

# 打印纯稀布的最优位置
best = results[100][0]
pos = best["result"]["positions"]
print(f"\n纯稀布最优: PSLL={best['f']:.4f} dB")
print(f"  位置: {np.array2string(pos, precision=3, max_line_width=100)}")
