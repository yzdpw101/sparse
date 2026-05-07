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
from antopt.element_pattern import ElementPattern
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def _to_json_flat(obj, indent=0):
    """仿 C++ toJsonFlatArrays: 对象换行, 数组紧凑单行。"""
    sp = " " * (indent + 2)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            items.append(f'{sp}{json.dumps(k)}: {_to_json_flat(v, indent + 2)}')
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    elif isinstance(obj, list):
        return json.dumps(obj)
    else:
        return json.dumps(obj)


def _compute_pattern(pos, amps, phases, mapper, pat, fe_patterns):
    """计算方向图乘积: pattern = Fe * |AF| (实数量)。"""
    if pat.is_planar or not mapper.is_symmetric:
        af = pat.linear_af(pos, amps, phases)
    else:
        halfNe = mapper._halfNe
        af = pat.linear_af_symmetric(
            pos[halfNe:], has_center=mapper.has_center,
            amplitudes=amps[halfNe:], phases=phases[halfNe:],
        )
    af_abs = np.abs(af)
    if fe_patterns is not None:
        af_abs = af_abs * fe_patterns[0]
    return af_abs


# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

# ── 2. 解析参数 ──
freqs = np.array(cfg["frequenciesGHz"], dtype=float)
seed = cfg["randomSeed"]
asp = cfg["aspectAngle"]
theta_start = asp.get("thetaStartDeg", -90)
theta_end = asp.get("thetaEndDeg", 90)
theta_step = asp["thetaStepDeg"]
theta0s = np.array(asp["theta0sDeg"], dtype=float)
mode = cfg["mode"]
amp_bounds = tuple(cfg["amplitudeBounds"])
hpbw_target = cfg.get("targetHPBW", 100.0)
use_fe = cfg.get("useFe", False)
fe_dir = cfg.get("eGainCsvDirectory", None)
fe_deg_step = cfg.get("eGainDegStep", None)

p_mode = (mode // 100) % 10
h_mode = (mode // 10) % 10
a_mode = mode % 10

# ── 3. 导入阵元配置（按需） ──
def load_array_key(filename, key):
    """从 JSON 文件加载 array_config 格式或纯数组。支持相对/绝对路径。"""
    if not filename:
        return None
    path = Path(filename)
    if not path.is_absolute():
        path = HERE / filename
    if not path.exists():
        raise FileNotFoundError(f"导入文件不存在: {path}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and key in data:
        data = data[key]
    return np.array(data, dtype=float) if data is not None else None

imp = cfg.get("import", {})
init_pos = load_array_key(imp.get("positionsFile"), "xCenters") if p_mode in (0, 2) else None
init_phs = load_array_key(imp.get("phasesFile"), "phasesDeg") if h_mode == 2 else None
init_amp = load_array_key(imp.get("amplitudesFile"), "amplitudes") if a_mode == 2 else None

# 阵元数: 优化位置时从 antennaArray 取, 导入位置时由文件决定
optimize_pos = (p_mode == 1)
if optimize_pos:
    arr = cfg["antennaArray"]
    Ne = arr["Ne"]
    L = arr["L_wavelength"]
    dmin = arr["dmin_wavelength"]
    dmax = arr.get("dmax_wavelength")
    if dmax is None or dmax <= 0:
        dmax = None
    is_sym = arr.get("isSymmetryArray", False)
    is_fixed = arr.get("isFixedAperture", False)
else:
    Ne = len(init_pos)
    L = dmin = dmax = None
    is_sym = is_fixed = False

opt = cfg["optimizer"]
method = opt["method"]
opt_params = opt.get(method, {})
verbose = opt_params.pop("verbose", method != "cma")
stop_fitness = opt_params.pop("stopFitness", None)

# 单元方向图: 多频 CSV → Fe = sqrt(gain)
fe_patterns = None
if use_fe and fe_dir:
    fe_path = HERE / fe_dir
    if fe_path.exists():
        fe_patterns = ElementPattern.from_hfss_multi_freq(
            str(fe_path), freqs,
            np.arange(theta_start, theta_end + theta_step/2, theta_step),
            fe_deg_step or 0.01,
            is_gain=cfg.get("feIsGain", True),
            in_dB=cfg.get("feInDB", True),
            input_theta_range=cfg.get("feThetaRange", (-90.0, 90.0)),
        )
        print(f"  加载单元方向图: {len(fe_patterns)} 个频率, 每个 {len(fe_patterns[0])} 点")

# ── 4. 构造 ──
print(f"=== 稀布幅相优化 ===")
if optimize_pos:
    print(f"  阵列: {Ne}元, 孔径={L}λ, dmin={dmin}λ, 对称={is_sym}")
else:
    print(f"  阵列: {Ne}元 (导入位置), 对称={is_sym}")
print(f"  mode={mode} (位置={'优化' if p_mode==1 else '导入'}, "
      f"相位={['等相位','优化','导入'][h_mode]}, 幅度={['等幅度','优化','导入'][a_mode]})")
print(f"  频率: {freqs.tolist()} GHz, 扫描角: {theta0s.tolist()} deg")
print(f"  优化器: {method}, 参数: {opt_params}")
print(f"  变量维度: ", end="")

if optimize_pos:
    mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, dmax=dmax, is_symmetric=is_sym,
                      is_fixed_aperture=is_fixed)
    n_pos = mapper.n_vars
else:
    # 导入位置时 LM 映射器仅提供 Ne/is_symmetric 等属性
    pos_span = init_pos[-1] - init_pos[0]
    pos_dmin = min(np.diff(init_pos))
    mapper = LMMapper(Ne=Ne, L=pos_span, dmin=pos_dmin, is_symmetric=is_sym)
    n_pos = 0

pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s,
              frequenciesGHz=freqs)
n_phs = Ne if h_mode == 1 else 0
n_amp = Ne if a_mode == 1 else 0
print(f"{n_pos} (位置) + {n_phs} (相位) + {n_amp} (幅度) = {n_pos + n_phs + n_amp}")

# ── 5. 优化 ──
run_opts = {}
for k in ("pop_size", "max_iter", "n_jobs"):
    if k in opt_params:
        run_opts[k] = opt_params[k]
if "sigma" in opt_params:
    run_opts["sigma0"] = opt_params["sigma"]

problem_opts = dict(
    mode=mode,
    init_positions=init_pos,
    init_phases_deg=init_phs,
    init_amplitudes=init_amp,
    amplitude_bounds=amp_bounds,
    target_hpbw=hpbw_target,
    element_patterns=fe_patterns,
)

t0 = time.perf_counter()
result = run_optimization(
    mapper, pat,
    method=method,
    seed=seed,
    verbose=verbose,
    stop_fitness=stop_fitness,
    **run_opts,
    **problem_opts,
)
elapsed = time.perf_counter() - t0

# ── 6. 最优方向图 ──
res = result["result"]
pos = res["positions"]
amps = res["amplitudes"]
phases = np.deg2rad(res["phases_deg"])
pattern = _compute_pattern(pos, amps, phases, mapper, pat, fe_patterns)

print(f"\n  最优适应度: {result['f']:.4f} dB")
print(f"  耗时: {elapsed:.1f}s")

# 方向图: dB 归一化
theta = pat.theta_deg
af_db = Pattern.to_dB(pattern)

from antopt.analysis import find_peaks, get_psll
psll_val, psll_angle = get_psll(af_db, theta)
all_vals, all_angles = find_peaks(af_db, theta)

# 主瓣: 最接近 0 dB 且最接近指向角的峰值
target_angle = theta0s[0]
main_idx = np.argmax(af_db)  # 默认最大值为 main lobe
near_zero = np.abs(all_vals) < 1e-6
if near_zero.any():
    candidates_idx = np.where(near_zero)[0]
    main_idx_rel = candidates_idx[np.argmin(np.abs(all_angles[candidates_idx] - target_angle))]
    main_val = all_vals[main_idx_rel]
    main_ang = all_angles[main_idx_rel]
else:
    main_val = all_vals[0]
    main_ang = all_angles[0]

# 副瓣: 非主瓣的最高峰
sll_mask = np.ones(len(all_vals), dtype=bool)
if near_zero.any():
    sll_mask[candidates_idx] = False
sll_vals = all_vals[sll_mask]
sll_angles = all_angles[sll_mask]
sll_val = sll_vals[0] if len(sll_vals) > 0 else -np.inf
sll_ang = sll_angles[0] if len(sll_vals) > 0 else np.nan

# 绘图
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(theta, af_db, linewidth=1.0, color='steelblue')
# 主瓣标记 (绿色方块)
ax.scatter([main_ang], [main_val], c='green', s=80, marker='s', zorder=6,
           label=f'Main lobe: {main_val:.4f} dB @ {main_ang:.2f}$^\\circ$')
# 副瓣标记 (红色菱形)
if not np.isinf(sll_val):
    ax.scatter([sll_ang], [sll_val], c='red', s=80, marker='D', zorder=6,
               label=f'Side lobe: {sll_val:.4f} dB @ {sll_ang:.2f}$^\\circ$')
# 3dB 线
ax.axhline(y=-3, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

title = f"Radiation Pattern  (f={freqs[0]:.4g} GHz, $\\theta_0$={theta0s[0]:.1f}$^\\circ$, PSLL={psll_val:.2f} dB)"
ax.set_xlabel("$\\theta$ (deg)")
ax.set_ylabel("Normalized Pattern (dB)")
ax.set_title(title)
ax.set_ylim(-60, 3)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ── 7. 保存结果 ──
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = HERE / "result" / f"{ts}.json"
out_data = {
    "settings": cfg,
    "result": {
        "bestFitness": float(result["f"]),
        "bestPositions": pos.tolist(),
        "bestPhasesDeg": res["phases_deg"].tolist(),
        "bestAmplitudes": amps.tolist(),
        "optimizer": method,
        "elapsedSeconds": round(elapsed, 1),
    },
    "pattern": {
        "frequenciesGHz": freqs.tolist(),
        "theta0sDeg": theta0s.tolist(),
        "thetaDeg": theta.tolist(),
        "value": pattern.tolist(),
    },
}
with open(out_path, "w", encoding="utf-8") as f:
    f.write(_to_json_flat(out_data))
print(f"  结果已保存: {out_path}")
