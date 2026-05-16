"""线阵单零陷优化主脚本 — CMA 约束优化 + 断点续跑。

用法: cd examples/线阵单零陷优化 && python main.py
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
import matplotlib.pyplot as plt

from antopt import LMMapper, Pattern, minimize, NullSteeringProblem, get_psll
from antopt.analysis import _find_peaks
from antopt.element_pattern import ElementPattern
from antopt.utils import to_json_flat, load_array_config

# exe 兼容
HERE = Path(sys.argv[0]).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
CKPT_FILE = HERE / ".checkpoint.json"


# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

freqs = np.sort(np.array(cfg["frequenciesGHz"], dtype=float))
asp = cfg["aspectAngle"]
_ta = asp["thetaDeg"]  # [start, end, step]
theta_start, theta_end, theta_step = _ta[0], _ta[1], _ta[2]
theta0s = np.array(asp["theta0sDeg"], dtype=float)

# 种子校验
_raw_seed = cfg.get("randomSeed")
if _raw_seed is None:
    seed = int(np.random.randint(0, 2**31))
elif isinstance(_raw_seed, float):
    warnings.warn(f"randomSeed 为浮点数 ({_raw_seed})，已忽略，使用随机种子")
    seed = int(np.random.randint(0, 2**31))
elif isinstance(_raw_seed, int) and _raw_seed < 0:
    warnings.warn(f"randomSeed 为负整数 ({_raw_seed})，已忽略，使用随机种子")
    seed = int(np.random.randint(0, 2**31))
else:
    seed = int(_raw_seed)

# 零陷参数
ns_cfg = cfg["nullSteering"]
null_angle = ns_cfg["nullAngleDeg"]
null_target = ns_cfg["nullTargetDb"]
sll_target = ns_cfg["sllTargetDb"]
null_win_half = ns_cfg["nullWindowHalfDeg"]
null_margin = ns_cfg["nullMarginDb"]
weights = ns_cfg.get("weights", {})

# 优化模式
optz = cfg["target"]
mode = optz["mode"]
amp_bounds = tuple(optz["amplitudeBounds"])
use_half = optz.get("optimizeHalfAmpPhase", False)
p_mode = (mode // 100) % 10; h_mode = (mode // 10) % 10; a_mode = mode % 10

# ── 2. 导入数据 ──
imp = optz.get("import", {})
init_pos = load_array_config(imp.get("positionsFile"), "xCenters", HERE) if p_mode in (0, 2) else None
init_phs = load_array_config(imp.get("phasesFile"), "phasesDeg", HERE) if h_mode == 2 else None
init_amp = load_array_config(imp.get("amplitudesFile"), "amplitudes", HERE) if a_mode == 2 else None
if init_pos is not None:
    lam0 = 299792458.0 / (freqs[0] * 1e9)
    init_pos = init_pos / lam0

# ── 3. 阵列 ──
if p_mode == 1:
    arr = cfg["antennaArray"]
    Ne, L_wl, dmin_wl = arr["Ne"], arr["L_wavelength"], arr["dmin_wavelength"]
    is_sym = arr.get("isSymmetryArray", False)
    is_fixed = arr.get("isFixedAperture", False)
else:
    Ne = len(init_pos)
    L_wl = init_pos[-1] - init_pos[0]
    dmin_wl = min(np.diff(init_pos))
    is_fixed = False
    is_sym = all(abs(init_pos[i] + init_pos[Ne - 1 - i]) < 1e-6 for i in range(Ne // 2))

# ── 4. 单元方向图 ──
fe_patterns = None
eg = cfg.get("ePattern", {})
if eg.get("enabled") and eg.get("csvDirectory"):
    epath = HERE / eg["csvDirectory"]
    if epath.exists():
        _ep = eg.get("thetaDeg", [-180, 180, 0.1])
        ep_start, ep_end, ep_step = _ep[0], _ep[1], _ep[2]
        fe_patterns = ElementPattern.from_hfss_multi_freq(
            str(epath), freqs,
            np.arange(theta_start, theta_end + theta_step / 2, theta_step),
            ep_step,
            is_gain=eg.get("isGain", True), in_dB=eg.get("inDB", False),
            input_theta_range=(ep_start, ep_end),
        )
        print(f"  加载单元方向图: {len(fe_patterns)} 个频率")

# ── 5. 映射器 + 方向图 ──
mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl, is_symmetric=is_sym,
                   is_fixed_aperture=is_fixed, use_sigmoid=False)
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s, frequenciesGHz=freqs)

# ── 6. 构建适应度 ──
problem = NullSteeringProblem(
    mapper, pat, mode=mode,
    init_positions=init_pos, init_phases_deg=init_phs, init_amplitudes=init_amp,
    amplitude_bounds=amp_bounds,
    null_angle_deg=null_angle, null_target_db=null_target,
    sll_target_db=sll_target, null_window_half_deg=null_win_half,
    null_margin_db=null_margin,
    use_pointing_penalty=True, optimize_half_amp_phase=use_half,
    element_patterns=fe_patterns, weights=weights,
)

print(f"=== 线阵单零陷优化 (CMA) ===")
print(f"  阵列: {Ne}元, 孔径={L_wl:.4f}λ, dmin={dmin_wl:.4f}λ, 对称={is_sym}")
if len(freqs) > 1 or p_mode == 2:
    print(f"  频率: {freqs.tolist()} GHz")
print(f"  mode={mode} (位置={'优化' if p_mode==1 else '导入'}, "
      f"相位={'优化' if h_mode==1 else '导入' if h_mode==2 else '默认'}, "
      f"幅度={'优化' if a_mode==1 else '导入' if a_mode==2 else '默认'})")
print(f"  零陷角: {null_angle}°, 目标深度: {null_target} dB, 副瓣目标: {sll_target} dB")
print(f"  零陷窗口: ±{null_win_half}°, 谷形余量: {null_margin} dB")
print(f"  权重: pointing={weights.get('pointing', 0.5)}, "
      f"null_depth={weights.get('null_depth', 3.0)}, "
      f"valley={weights.get('valley', 1.0)}, sll={weights.get('sll', 5.0)}")
print(f"  θ ∈ [{theta_start}°, {theta_end}°], Δθ={theta_step}°, θ0={theta0s.tolist()}")
print(f"  种子: {seed}")
print(f"  n_vars={problem.n_vars} (位置={problem.n_pos}, 相位={problem.n_phase}, 幅度={problem.n_amp})")

# ── 7. 优化器参数 ──
opt_param = cfg.get("optimizer", {})
resume_enabled = opt_param.get("resume", False)

# 构建分变量类型边界
lb = np.empty(problem.n_vars); ub = np.empty(problem.n_vars)
cur = 0
if problem.n_pos > 0:
    lb[:problem.n_pos] = 0.0; ub[:problem.n_pos] = 1.0
    cur = problem.n_pos
if problem.n_phase > 0:
    lb[cur:cur+problem.n_phase] = 0.0; ub[cur:cur+problem.n_phase] = 2 * np.pi
    cur += problem.n_phase
if problem.n_amp > 0:
    lb[cur:] = amp_bounds[0]; ub[cur:] = amp_bounds[1]

t0 = time.perf_counter()
result = minimize(
    problem.fitness, problem.n_vars,
    method="cma",
    bounds=(lb, ub),
    sigma=opt_param.get("sigma", 0.5),
    pop_size=opt_param.get("pop_size") or None,
    max_iter=opt_param.get("max_iter", 500),
    seed=seed,
    n_jobs=opt_param.get("n_jobs", -1),
    verbose=opt_param.get("verbose", True),
    init="chaos",
    stop_fitness=opt_param.get("stopFitness"),
    checkpoint=str(CKPT_FILE) if resume_enabled else None,
    resume=str(CKPT_FILE) if resume_enabled and CKPT_FILE.exists() else None,
)
elapsed = time.perf_counter() - t0

# 清除断点文件
if resume_enabled and CKPT_FILE.exists():
    CKPT_FILE.unlink()

# ── 8. 结果 ──
res = problem.get_result(result["x"])
pos, amps, phases_deg = res["positions"], res["amplitudes"], res["phases_deg"]
theta_range = f"θ ∈ [{theta_start}°, {theta_end}°], Δθ = {theta_step}°"

print(f"\n  最优适应度: {result['f']:.4f}, 耗时: {elapsed:.1f}s")

# ── 9. 计算最终方向图 (与 fitness() AF 路径一致) ──
phases_rad = np.deg2rad(phases_deg)
if mapper.is_symmetric and use_half:
    offset = 1 if mapper.has_center else 0
    half_pos = pos[mapper._halfNe + offset:]
    half_amp = amps[mapper._halfNe + offset:]
    half_phs = phases_rad[mapper._halfNe + offset:]
    kwargs = dict(has_center=mapper.has_center, amplitudes=half_amp, phases=half_phs)
    if mapper.has_center:
        kwargs["center_amplitude"] = amps[mapper._halfNe]
        kwargs["center_phase"] = phases_rad[mapper._halfNe]
    af = pat.linear_af_symmetric(half_pos, **kwargs)
else:
    af = pat.linear_af(pos, amps, phases_rad)
if fe_patterns is not None:
    af_mag = np.abs(af)
    if af.ndim == 1:
        af = af_mag * fe_patterns[0]
    else:
        for i in range(len(fe_patterns)):
            af[i] = af_mag[i] * fe_patterns[i]
af_db = 20 * np.log10(np.maximum(np.abs(af), 1e-30) / np.max(np.abs(af)))
af_db_flat = af_db.reshape(-1, af_db.shape[-1]) if af_db.ndim > 1 else af_db[None, :]

n_freq, n_scan = len(freqs), len(theta0s)
multi_freq = n_freq > 1
multi_scan = n_scan > 1

# ── 10. 保存图片 + CSV ──
import csv as _csv
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir = out_dir / "patterns"; csv_dir.mkdir(parents=True, exist_ok=True)
per_dir = fig_dir / "per_pattern"; per_dir.mkdir(exist_ok=True)

worst_psll = -np.inf
worst_null = -np.inf
for sub in range(af_db_flat.shape[0]):
    si = sub % n_scan; fi = sub // n_scan if n_scan > 0 else 0
    f_ghz = freqs[fi % n_freq]; t0_scan = theta0s[si]
    psll_v, _ = get_psll(af_db_flat[sub], pat.theta_deg)
    if psll_v > worst_psll: worst_psll = psll_v

    # 零陷深度
    null_idx = int(np.argmin(np.abs(pat.theta_deg - null_angle)))
    hw = int(round(null_win_half / abs(pat.theta_deg[1] - pat.theta_deg[0])))
    null_win_slice = slice(max(0, null_idx - hw), min(len(pat.theta_deg), null_idx + hw + 1))
    null_depth = np.max(af_db_flat[sub][null_win_slice])
    if null_depth > worst_null: worst_null = null_depth

    if multi_freq:
        label = f"f = {f_ghz:.4g} GHz,  $\\theta_0$ = {t0_scan:.1f}°"
    else:
        label = f"$\\theta_0$ = {t0_scan:.1f}°"
    if multi_freq:
        fname = f"f{f_ghz:.4g}GHz_theta{t0_scan:.1f}.png"
    else:
        fname = f"theta{t0_scan:.1f}.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pat.theta_deg, af_db_flat[sub], linewidth=1.0, label=label)
    pk_idx, pk_val = _find_peaks(af_db_flat[sub])
    if len(pk_val) > 1:
        sl_theta, sl_db = pat.theta_deg[pk_idx[1]], pk_val[1]
        ax.plot(sl_theta, sl_db, "ro", markersize=4,
                label=f"SLL: {sl_theta:.2f}°, {sl_db:.2f} dB")
    ax.axvline(x=null_angle, color='g', linestyle='--', alpha=0.5,
               label=f"Null target: {null_angle}°")
    ax.plot(null_angle, null_depth, "gs", markersize=6,
            label=f"Null: {null_depth:.2f} dB")
    ax.set_title(f"{label}\nPSLL = {psll_v:.2f} dB, Null = {null_depth:.2f} dB\n{theta_range}")
    ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(-80, 3); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(per_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    csv_name = fname.replace(".png", ".csv")
    with open(csv_dir / csv_name, "w", newline="", encoding="utf-8") as cf:
        w = _csv.writer(cf)
        w.writerow(["theta_deg", "pattern_dB"])
        for j in range(len(pat.theta_deg)):
            w.writerow([pat.theta_deg[j], af_db_flat[sub, j]])

# ── 11. 保存 JSON ──
cfg_save = dict(cfg)
cfg_save["randomSeed"] = seed
out_data = {
    "settings": cfg_save,
    "result": {
        "bestFitness": float(result["f"]),
        "bestPSLL": worst_psll,
        "worstNullDepth": worst_null,
        "bestX": result["x"].tolist(),
        "bestPositions": pos.tolist(),
        "bestPhasesDeg": phases_deg.tolist(),
        "bestAmplitudes": amps.tolist(),
        "optimizer": "cma", "elapsedSeconds": round(elapsed, 1),
    },
}
with open(out_dir / "optResult.json", "w", encoding="utf-8") as f:
    f.write(to_json_flat(out_data))
print(f"  结果已保存: {out_dir}")
