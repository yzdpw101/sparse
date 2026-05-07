"""稀布幅相优化主脚本 — 仿 C++ Main.cpp 流程。

用法: cd examples/稀布幅相优化 && python main.py
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from antopt import LMMapper, Pattern, run_optimization

HERE = Path(__file__).resolve().parent

# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

# ── 2. 解析参数 ──
freqs = np.array(cfg["frequenciesGHz"], dtype=float)
seed = cfg["randomSeed"]
theta_step = cfg["aspectAngle"]["thetaStepDeg"]
theta0s = np.array(cfg["aspectAngle"]["theta0sDeg"], dtype=float)
mode = cfg["mode"]
amp_bounds = tuple(cfg["amplitudeBounds"])
hpbw_target = cfg["targetHPBW"]

arr = cfg["antennaArray"]
L = arr["L_wavelength"]
dmin = arr["dmin_wavelength"]
Ne = arr["Ne"]
is_sym = arr["isSymmetryArray"]

opt = cfg["optimizer"]
method = opt["method"]
opt_params = opt.get(method, {})

# ── 3. 导入阵元配置（按需） ──
def load_array_key(filename, key):
    """从 JSON 文件加载 array_config 格式或纯数组。"""
    if not filename:
        return None
    path = HERE / filename
    if not path.exists():
        raise FileNotFoundError(f"导入文件不存在: {path}")
    with open(path) as f:
        data = json.load(f)
    # 支持 {"xCenters": [...], ...} 或纯数组
    if isinstance(data, dict) and key in data:
        data = data[key]
    return np.array(data, dtype=float) if data is not None else None

p_mode = (mode // 100) % 10
h_mode = (mode // 10) % 10
a_mode = mode % 10

imp = cfg.get("import", {})
init_pos = load_array_key(imp.get("positionsFile"), "xCenters") if p_mode in (0, 2) else None
init_phs = load_array_key(imp.get("phasesFile"), "phasesDeg") if h_mode == 2 else None
init_amp = load_array_key(imp.get("amplitudesFile"), "amplitudes") if a_mode == 2 else None

# ── 4. 构造 ──
print(f"=== 稀布幅相优化 ===")
print(f"  阵列: {Ne}元, 孔径={L}λ, dmin={dmin}λ, 对称={is_sym}")
print(f"  mode={mode} (位置={'优化' if p_mode==1 else '导入'}, "
      f"相位={['等相位','优化','导入'][h_mode]}, 幅度={['等幅度','优化','导入'][a_mode]})")
print(f"  频率: {freqs.tolist()} GHz, 扫描角: {theta0s.tolist()} deg")
print(f"  优化器: {method}, 参数: {opt_params}")
print(f"  变量维度: ", end="")

mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=is_sym)
pat = Pattern(theta_deg_step=theta_step)

n_pos = mapper.n_vars if p_mode == 1 else 0
n_phs = Ne if h_mode == 1 else 0
n_amp = Ne if a_mode == 1 else 0
print(f"{n_pos} (位置) + {n_phs} (相位) + {n_amp} (幅度) = {n_pos + n_phs + n_amp}")

# ── 5. 优化 ──
# 分离 run_optimization 参数和 SparseArrayProblem 参数
run_opts = {}
for k in ("pop_size", "max_iter", "n_jobs"):
    if k in opt_params:
        run_opts[k] = opt_params[k]
if "sigma" in opt_params:
    run_opts["sigma0"] = opt_params["sigma"]
# mode 对应 SparseArrayProblem 参数
problem_opts = dict(
    mode=mode,
    init_positions=init_pos,
    init_phases_deg=init_phs,
    init_amplitudes=init_amp,
    amplitude_bounds=amp_bounds,
    target_hpbw=hpbw_target,
)

t0 = time.perf_counter()
result = run_optimization(
    mapper, pat,
    method=method,
    seed=seed,
    **run_opts,
    **problem_opts,
)
elapsed = time.perf_counter() - t0

# ── 6. 输出 ──
res = result["result"]
print(f"\n  最优适应度: {result['f']:.4f} dB")
print(f"  耗时: {elapsed:.1f}s")
print(f"  位置: {np.array2string(res['positions'], precision=3, max_line_width=120)}")
print(f"  间距: {np.array2string(np.diff(res['positions']), precision=3, max_line_width=120)}")
print(f"  孔径: {res['positions'][-1] - res['positions'][0]:.4f} λ")

# ── 7. 保存结果 ──
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = HERE / "result" / f"{ts}.json"
out_data = {
    "settings": cfg,
    "result": {
        "bestFitness": float(result["f"]),
        "bestPositions": res["positions"].tolist(),
        "bestPhasesDeg": res["phases_deg"].tolist(),
        "bestAmplitudes": res["amplitudes"].tolist(),
        "optimizer": method,
        "elapsedSeconds": round(elapsed, 1),
    },
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)
print(f"\n  结果已保存: {out_path}")
