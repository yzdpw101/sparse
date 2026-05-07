"""稀布线阵优化 demo — 最小化 PSLL。

可修改下方参数直接运行。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from antopt import LMMapper, Pattern, run_optimization

# ═══════════════════════════════════════════════════════════
#  调参区域
# ═══════════════════════════════════════════════════════════

Ne = 152                  # 阵元数
L = 98.5                # 孔径 (λ)
dmin = 0.5               # 最小间距 (λ)
SYMMETRIC = True        # 对称模式
FIXED_APERTURE = True   # 固定孔径
THETA_STEP = 0.1         # θ 步长（度）, 粗步长加快优化

POP_SIZE = 100            # CMA-ES 种群大小
MAX_ITER = 300           # 最大迭代次数
SEED = 41                # 随机种子

# ═══════════════════════════════════════════════════════════
#  优化
# ═══════════════════════════════════════════════════════════

mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=SYMMETRIC, is_fixed_aperture=FIXED_APERTURE)
pat = Pattern(theta_deg_start=-90, theta_deg_end=90, theta_deg_step=THETA_STEP)

print(f"阵元数: {Ne}, 孔径: {L}λ, dmin: {dmin}λ")
print(f"对称: {SYMMETRIC}, 变量维度: {mapper.n_vars}")
print(f"种群: {POP_SIZE}, 最大迭代: {MAX_ITER}")

result = run_optimization(
    mapper, pat,
    pop_size=POP_SIZE,
    max_iter=MAX_ITER,
    seed=SEED,
    verbose=True,
)

# ═══════════════════════════════════════════════════════════
#  结果
# ═══════════════════════════════════════════════════════════

pos = result["result"]["positions"]
print(f"\n--- 优化结果 ---")
print(f"最优适应度 (PSLL): {result['f']:.4f} dB")
print(f"最优位置 (λ):")
print(f"  {np.array2string(pos, precision=4, max_line_width=120)}")
print(f"间距: {np.array2string(np.diff(pos), precision=4, max_line_width=120)}")
print(f"孔径: {pos[-1] - pos[0]:.4f} λ")
