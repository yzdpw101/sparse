"""稀布幅相优化 — Kriging 代理模型 + CMA-ES。

用法: cd examples/线阵稀布幅相优化代理模型 && python main.py

流程:
  1. LHS 采样 → AF 计算 PSLL → 训练 Kriging
  2. CMA-ES 在 Kriging 上搜索 → EI 填充采样
  3. 重训练, 重复直到收敛
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

from antopt import LMMapper, Pattern, SparseArrayProblem, get_psll
from antopt.surrogate import surrogate_optimize
from antopt.analysis import _find_peaks
from antopt.element_pattern import ElementPattern
from antopt.utils import to_json_flat, load_array_config

HERE = Path(sys.argv[0]).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent


# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

freqs = np.sort(np.array(cfg["frequenciesGHz"], dtype=float))
asp = cfg["aspectAngle"]
_ta = asp["thetaDeg"]
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

optz = cfg["target"]
mode = optz["mode"]
amp_bounds = tuple(optz["amplitudeBounds"])
hpbw_target = optz.get("targetHPBW") or 180.0
use_pt_penalty = optz.get("mainLobePointingPenalty", True)
optimize_half = optz.get("optimizeHalfAmpPhase", False)
p_mode = (mode // 100) % 10; h_mode = (mode // 10) % 10; a_mode = mode % 10

# ── 2. 导入数据 ──
imp = optz.get("import", {})
init_pos = load_array_config(imp.get("positionsFile"), "xCenters", HERE) if p_mode in (0, 2) and imp.get("positionsFile") else None
init_phs = load_array_config(imp.get("phasesFile"), "phasesDeg", HERE) if h_mode == 2 and imp.get("phasesFile") else None
init_amp = load_array_config(imp.get("amplitudesFile"), "amplitudes", HERE) if a_mode == 2 and imp.get("amplitudesFile") else None
if init_pos is not None:
    lam0 = 299792458.0 / (freqs[0] * 1e9)
    init_pos = init_pos / lam0

optimize_pos = (p_mode == 1)

# ── 3. 阵列 ──
if optimize_pos:
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
        _ep = eg.get("thetaDeg", [-180, 180, 0.01])
        ep_start, ep_end, ep_step = _ep[0], _ep[1], _ep[2]
        fe_patterns = ElementPattern.from_hfss_multi_freq(
            str(epath), freqs,
            np.arange(theta_start, theta_end + theta_step/2, theta_step),
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
problem = SparseArrayProblem(mapper, pat, mode=mode,
    init_positions=init_pos, init_phases_deg=init_phs, init_amplitudes=init_amp,
    amplitude_bounds=amp_bounds, target_hpbw=hpbw_target,
    element_patterns=fe_patterns, use_pointing_penalty=use_pt_penalty,
    optimize_half_amp_phase=optimize_half)

print(f"=== 稀布幅相优化 (Kriging 代理) ===")
print(f"  阵列: {Ne}元, 孔径={L_wl:.4f}λ, dmin={dmin_wl:.4f}λ, 对称={is_sym}")
if len(freqs) > 1 or p_mode == 2:
    print(f"  频率: {freqs.tolist()} GHz")
print(f"  mode={mode} (位置={'优化' if p_mode==1 else '导入' if p_mode==2 else '默认'}, "
      f"相位={'优化' if h_mode==1 else '导入' if h_mode==2 else '默认'}, "
      f"幅度={'优化' if a_mode==1 else '导入' if a_mode==2 else '默认'})")
print(f"  θ ∈ [{theta_start}°, {theta_end}°], Δθ={theta_step}°, θ0={theta0s.tolist()}")
print(f"  种子: {seed}")
print(f"  n_vars={problem.n_vars} (位置={problem.n_pos}, 相位={problem.n_phase}, 幅度={problem.n_amp})")

# ── 7. 代理模型优化 ──
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

surr_cfg = cfg.get("surrogate", {})
t0 = time.perf_counter()
result = surrogate_optimize(
    problem.fitness, problem.n_vars, bounds=(lb, ub),
    n_initial=surr_cfg.get("n_initial"),
    n_infill=surr_cfg.get("n_infill", 5),
    max_iter=surr_cfg.get("max_iter", 20),
    patience=surr_cfg.get("patience", 3),
    acquisition=surr_cfg.get("acquisition", "ei"),
    kappa=surr_cfg.get("kappa", 2.0),
    method=surr_cfg.get("method", "cma"),
    sigma=surr_cfg.get("sigma", 0.3),
    seed=seed,
    verbose=cfg.get("optimizer", {}).get("verbose", True),
)
elapsed = time.perf_counter() - t0

# ── 8. 结果 ──
res = problem.get_result(result["x"])
pos, amps = res["positions"], res["amplitudes"]
phases_deg = res["phases_deg"]

print(f"\n  最优 PSLL: {result['f']:.4f} dB, 总评估: {result['n_evals']} 次, 耗时: {elapsed:.1f}s")

# ── 9. 保存图片 + CSV ──
import csv as _csv
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir = out_dir / "patterns"; csv_dir.mkdir(parents=True, exist_ok=True)
n_freq, n_scan = len(freqs), len(theta0s)
per_dir = fig_dir / "per_pattern"; per_dir.mkdir(exist_ok=True)

# 收敛历史图
history = result["history"]
fig_h, ax_h = plt.subplots(figsize=(8, 4))
ax_h.plot([h["n_evals"] for h in history], [h["best_f"] for h in history],
          "o-", markersize=3, linewidth=1)
ax_h.set_xlabel("函数评估次数"); ax_h.set_ylabel("最优 PSLL (dB)")
ax_h.set_title("代理模型优化收敛曲线"); ax_h.grid(True, alpha=0.3)
fig_h.savefig(fig_dir / "convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig_h)

# 计算方向图
phases_rad = np.deg2rad(phases_deg)
if mapper.is_symmetric and problem.optimize_half_amp_phase:
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
# 与 fitness() 一致: pattern = Fe × |AF|
if fe_patterns is not None:
    af_mag = np.abs(af)
    if af.ndim == 1:
        af = af_mag * fe_patterns[0]
    else:
        for i in range(len(fe_patterns)):
            af[i] = af_mag[i] * fe_patterns[i]
af_db = 20 * np.log10(np.maximum(np.abs(af), 1e-30) / np.max(np.abs(af)))
af_db_flat = af_db.reshape(-1, af_db.shape[-1]) if af_db.ndim > 1 else af_db[None, :]

multi_freq = n_freq > 1
multi_scan = n_scan > 1
if multi_scan: (fig_dir / "by_freq").mkdir(exist_ok=True)
if multi_freq: (fig_dir / "by_angle").mkdir(exist_ok=True)

worst_psll = -np.inf
for sub in range(af_db_flat.shape[0]):
    si = sub % n_scan; fi = sub // n_scan if n_scan > 0 else 0
    f_ghz = freqs[fi % n_freq]; t0_ang = theta0s[si]
    psll_v, _ = get_psll(af_db_flat[sub], pat.theta_deg)
    if psll_v > worst_psll: worst_psll = psll_v
    if multi_freq:
        label = f"f = {f_ghz:.4g} GHz,  $\\theta_0$ = {t0_ang:.1f}°"
        fname = f"f{f_ghz:.4g}GHz_theta{t0_ang:.1f}.png"
    else:
        label = f"$\\theta_0$ = {t0_ang:.1f}°"
        fname = f"theta{t0_ang:.1f}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pat.theta_deg, af_db_flat[sub], linewidth=1.0, label=label)
    pk_idx, pk_val = _find_peaks(af_db_flat[sub])
    if len(pk_val) > 1:
        sl_theta, sl_db = pat.theta_deg[pk_idx[1]], pk_val[1]
        ax.plot(sl_theta, sl_db, "ro", markersize=4,
                label=f"SLL: {sl_theta:.2f}°, {sl_db:.2f} dB")
    theta_range = f"θ ∈ [{theta_start}°, {theta_end}°], Δθ = {theta_step}°"
    ax.set_title(f"{label},  PSLL = {psll_v:.2f} dB\n{theta_range}")
    ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(per_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    csv_name = fname.replace(".png", ".csv")
    with open(csv_dir / csv_name, "w", newline="", encoding="utf-8") as cf:
        w = _csv.writer(cf)
        w.writerow(["theta_deg", "pattern_dB"])
        for j in range(len(pat.theta_deg)):
            w.writerow([pat.theta_deg[j], af_db_flat[sub, j]])

# ── 10. 保存 JSON ──
cfg_save = dict(cfg)
cfg_save["randomSeed"] = seed
out_data = {
    "settings": cfg_save,
    "result": {
        "bestFitness": float(result["f"]), "bestPSLL": worst_psll,
        "bestX": result["x"].tolist(),
        "bestPositions": pos.tolist(),
        "bestPhasesDeg": phases_deg.tolist(), "bestAmplitudes": amps.tolist(),
        "optimizer": "kriging_cma", "elapsedSeconds": round(elapsed, 1),
        "nEvals": result["n_evals"],
    },
}
with open(out_dir / "optResult.json", "w", encoding="utf-8") as f:
    f.write(to_json_flat(out_data))
print(f"  结果已保存: {out_dir}")
