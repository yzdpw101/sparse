"""稀布幅相优化主脚本 — 仿 C++ Main.cpp 流程。

用法: cd examples/稀布幅相优化 && python main.py
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from antopt import LMMapper, Pattern, run_optimization
from antopt.element_pattern import ElementPattern
from antopt.utils import to_json_flat, load_array_config, compute_pattern
from antopt.analysis import find_peaks, get_psll
from visualization import plot_pattern_with_lobes

# exe 兼容: __file__ 在 PyInstaller 中指向临时目录, 改用工作目录
HERE = Path(sys.argv[0]).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent

# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

# ── 2. 解析参数 ──
freqs = np.array(cfg["frequenciesGHz"], dtype=float)
asp = cfg["aspectAngle"]
theta_start = asp.get("thetaStartDeg", -90); theta_end = asp.get("thetaEndDeg", 90)
theta_step = asp["thetaStepDeg"]
theta0s = np.array(asp["theta0sDeg"], dtype=float)

optz = cfg["target"]
mode = optz["mode"]
amp_bounds = tuple(optz["amplitudeBounds"])
hpbw_target = optz.get("targetHPBW") or 180.0
use_pt_penalty = optz.get("mainLobePointingPenalty", True)

p_mode = (mode // 100) % 10; h_mode = (mode // 10) % 10; a_mode = mode % 10

# ── 3. 导入数据 ──
imp = optz.get("import", {})
init_pos = load_array_config(imp.get("positionsFile"), "xCenters", HERE) if p_mode in (0, 2) else None
init_phs = load_array_config(imp.get("phasesFile"), "phasesDeg", HERE) if h_mode == 2 else None
init_amp = load_array_config(imp.get("amplitudesFile"), "amplitudes", HERE) if a_mode == 2 else None

# 导入的绝对位置 (m) → 波长数 (λ₀ = c / f₀)
if init_pos is not None:
    lam0 = 299792458.0 / (freqs[0] * 1e9)
    init_pos = init_pos / lam0

optimize_pos = (p_mode == 1)
if optimize_pos:
    arr = cfg["antennaArray"]
    Ne, L, dmin = arr["Ne"], arr["L_wavelength"], arr["dmin_wavelength"]
    dmax = arr.get("dmax_wavelength"); dmax = None if (dmax is None or dmax <= 0) else dmax
    is_sym = arr.get("isSymmetryArray", False)
    is_fixed = arr.get("isFixedAperture", False)
else:
    Ne = len(init_pos)
    L = dmin = dmax = None; is_sym = is_fixed = False

# ── 4. 单元方向图 ──
fe_patterns = None
eg = cfg.get("ePattern", {})
if eg.get("enabled") and eg.get("csvDirectory"):
    epath = HERE / eg["csvDirectory"]
    if epath.exists():
        fe_patterns = ElementPattern.from_hfss_multi_freq(
            str(epath), freqs,
            np.arange(theta_start, theta_end + theta_step/2, theta_step),
            eg.get("degStep") or 0.01,
            is_gain=eg.get("isGain", True), in_dB=eg.get("inDB", False),
            input_theta_range=tuple(eg.get("thetaRange", (-180.0, 180.0))),
        )
        print(f"  加载单元方向图: {len(fe_patterns)} 个频率")

# ── 5. 构造 ──
opt = cfg["optimizer"]
method = opt["method"]
opt_params = opt.get(method, {}).copy()
verbose = opt_params.pop("verbose", method == "cma")
stop_fitness = opt_params.pop("stopFitness", None)

print(f"=== 稀布幅相优化 ===")
print(f"  阵列: {Ne}元" + (f", 孔径={L}λ, dmin={dmin}λ, 对称={is_sym}" if optimize_pos else " (导入位置)"))
print(f"  mode={mode}, 频率={freqs.tolist()} GHz, 指向角={theta0s.tolist()} deg, 优化器={method}")

if optimize_pos:
    mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, dmax=dmax, is_symmetric=is_sym, is_fixed_aperture=is_fixed)
    n_pos = mapper.n_vars
else:
    mapper = LMMapper(Ne=Ne, L=init_pos[-1]-init_pos[0], dmin=min(np.diff(init_pos)), is_symmetric=is_sym)
    n_pos = 0
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s, frequenciesGHz=freqs)
n_phs = Ne if h_mode == 1 else 0; n_amp = Ne if a_mode == 1 else 0
print(f"  变量维度: {n_pos}+{n_phs}+{n_amp}={n_pos+n_phs+n_amp}")

# ── 6. 优化 ──
run_opts = {k: opt_params[k] for k in ("pop_size", "max_iter", "n_jobs") if k in opt_params}
if "sigma" in opt_params:
    run_opts["sigma0"] = opt_params["sigma"]

t0 = time.perf_counter()
result = run_optimization(mapper, pat, method=method, seed=cfg["randomSeed"],
    verbose=verbose, stop_fitness=stop_fitness,
    mode=mode, init_positions=init_pos, init_phases_deg=init_phs, init_amplitudes=init_amp,
    amplitude_bounds=amp_bounds, target_hpbw=hpbw_target,
    element_patterns=fe_patterns, use_pointing_penalty=use_pt_penalty,
    **run_opts)
elapsed = time.perf_counter() - t0

# ── 7. 方向图 & 绘图 ──
res = result["result"]
pos, amps = res["positions"], res["amplitudes"]
pattern = compute_pattern(pos, amps, np.deg2rad(res["phases_deg"]), mapper, pat, fe_patterns)
af_db = Pattern.to_dB(pattern)

print(f"\n  最优适应度: {result['f']:.4f} dB, 耗时: {elapsed:.1f}s")

psll_val, psll_angle = get_psll(af_db, pat.theta_deg)
all_vals, all_angles = find_peaks(af_db, pat.theta_deg)
plot_pattern_with_lobes(pat.theta_deg, af_db, psll_val, psll_angle,
                         theta0s[0], all_vals, all_angles,
                         freq_ghz=freqs[0], theta0_deg=theta0s[0])

# ── 8. 保存结果 ──
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = HERE / "result" / f"{ts}.json"
out_data = {
    "settings": cfg,
    "result": {
        "bestFitness": float(result["f"]), "bestPositions": pos.tolist(),
        "bestPhasesDeg": res["phases_deg"].tolist(), "bestAmplitudes": amps.tolist(),
        "optimizer": method, "elapsedSeconds": round(elapsed, 1),
    },
    "pattern": {
        "frequenciesGHz": freqs.tolist(), "theta0sDeg": theta0s.tolist(),
        "thetaDeg": pat.theta_deg.tolist(), "value": pattern.tolist(),
    },
}
with open(out_path, "w", encoding="utf-8") as f:
    f.write(to_json_flat(out_data))
print(f"  结果已保存: {out_path}")
