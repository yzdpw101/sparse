"""线阵稀布幅相优化 — 基于 antopt 新库结构的精简版。

用法: cd examples/线阵稀布幅相优化 && python main2.py
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

from antopt import (LMMapper, Pattern, SidelobeProblem, NullSteeringProblem,
                    TripleNullProblem, minimize, get_psll, load_array_config, to_json_flat)
from antopt.config import SparseConfig
from antopt.analysis import _find_peaks
from antopt.element_pattern import ElementPattern
from antopt.plotting import plot_pattern_comparison

HERE = Path(__file__).resolve().parent


# ============================================================
# 1. 加载配置
# ============================================================
cfg = SparseConfig.from_json(HERE / "Config.json")

freqs = np.sort(np.array(cfg.frequenciesGHz, dtype=float))
theta0s = np.array(cfg.theta0s, dtype=float)
theta = np.arange(cfg.theta_start, cfg.theta_end + cfg.theta_step / 2, cfg.theta_step)

# 种子
seed = cfg.randomSeed if cfg.randomSeed is not None else int(np.random.randint(0, 2**31))

print(f"=== 线阵稀布幅相优化 ===")
print(f"  频率: {freqs} GHz")
print(f"  θ ∈ [{cfg.theta_start}°, {cfg.theta_end}°], Δθ={cfg.theta_step}°, θ0={theta0s}")

# ============================================================
# 2. 阵列位置
# ============================================================
optimize_pos = (cfg.position_source == "optimize")

if optimize_pos:
    Ne = cfg.position_source  # placeholder, 从 raw JSON 取
    # 从原始 JSON 获取优化参数 (Config.json 里有 Ne, L_wavelength 等)
    with open(HERE / "Config.json", encoding="utf-8") as f:
        raw = json.load(f)
    pos_raw = raw["position"]
    Ne = pos_raw["Ne"]
    L_wl = pos_raw["L_wavelength"]
    dmin_wl = pos_raw["dmin_wavelength"]
    is_sym = pos_raw.get("symmetric", False)
    is_fixed = pos_raw.get("fixedAperture", False)
    init_pos = None
else:
    init_pos = load_array_config(cfg.position_source, "xCenters", HERE)
    lam0 = 299792458.0 / (freqs[0] * 1e9)
    init_pos = init_pos / lam0
    Ne = len(init_pos)
    L_wl = init_pos[-1] - init_pos[0]
    dmin_wl = float(np.min(np.diff(init_pos)))
    is_fixed = False
    is_sym = all(abs(init_pos[i] + init_pos[Ne - 1 - i]) < 1e-6 for i in range(Ne // 2))

print(f"  阵列: {Ne}元, 孔径={L_wl:.4f}λ, dmin={dmin_wl:.4f}λ, 对称={is_sym}")

# ============================================================
# 3. 单元方向图
# ============================================================
fe_patterns = None
if cfg.ep_enabled and cfg.ep_csv_dir:
    epath = Path(cfg.ep_csv_dir)
    if epath.exists():
        _is_aep = cfg.ep_aep_mode
        if not _is_aep:
            _csv_files = [f for f in epath.iterdir() if f.name.startswith("aep_elem_") and f.suffix == ".csv"]
            _is_aep = len(list(_csv_files)) > 0

        if _is_aep:
            fe_patterns = ElementPattern.from_hfss_aep(
                str(epath), freqs, theta, num_elements=Ne,
                oriDegStep=theta[1]-theta[0], is_gain=cfg.ep_is_gain,
                in_dB=cfg.ep_in_db, input_theta_range=cfg.ep_theta_range)
            print(f"  加载 AEP 方向图: {len(fe_patterns[0])} 个单元")
        else:
            # CSV 原始步长 0.01°, 目标步长由 theta 决定
            fe_patterns = ElementPattern.from_hfss_multi_freq(
                str(epath), freqs, theta, 0.01,
                is_gain=cfg.ep_is_gain, in_dB=cfg.ep_in_db,
                input_theta_range=cfg.ep_theta_range)
            print(f"  加载单元方向图: {len(fe_patterns)} 个频率")

# ============================================================
# 4. 映射器 + 问题构建
# ============================================================
mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl, is_symmetric=is_sym,
                   is_fixed_aperture=is_fixed, use_sigmoid=False)
pat = Pattern(theta_deg_start=cfg.theta_start, theta_deg_end=cfg.theta_end,
              theta_deg_step=cfg.theta_step, theta0s_deg=theta0s, frequenciesGHz=freqs)

# 激励参数
amp_source = cfg.amplitude_source
amp_bounds = cfg.amplitude_bounds
phs_source = cfg.phase_source
common_kw = dict(
    position_source=cfg.position_source if optimize_pos else init_pos,
    amplitude_source=amp_source,
    amplitude_bounds=amp_bounds,
    amplitude_in_db=cfg.amplitude_in_db,
    amplitude_step=cfg.amplitude_step,
    phase_step_deg=cfg.phase_step,
    phase_source=phs_source,
    optimize_half_amp=cfg.amplitude_optimize_half,
    optimize_half_phase=cfg.phase_optimize_half,
    init_positions=init_pos,
    element_patterns=fe_patterns,
    use_pointing_penalty=True,
)

task = cfg.target_type
if task == "sidelobe":
    problem = SidelobeProblem(mapper, pat,
        target_psll=cfg.target_psll,
        target_hpbw=cfg.target_hpbw or 180.0,
        gain_drop_limit_db=cfg.target_gain_drop_limit_db,
        directivity_limit_dbi=cfg.target_directivity_limit_dbi,
        **common_kw)
    stop_fitness = cfg.target_psll
elif task == "null":
    # null 任务需要额外参数, 从原始 JSON 取
    with open(HERE / "Config.json", encoding="utf-8") as f:
        raw = json.load(f)
    tgt = raw["target"]
    problem = NullSteeringProblem(mapper, pat,
        null_angle_deg=tgt["nullAngleDeg"],
        null_target_db=tgt["nullTargetDb"],
        sll_target_db=tgt["sllTargetDb"],
        null_window_half_deg=tgt["nullWindowHalfDeg"],
        null_margin_db=tgt["nullMarginDb"],
        weights=tgt.get("weights"),
        **common_kw)
    stop_fitness = None
elif task == "triple_null":
    with open(HERE / "Config.json", encoding="utf-8") as f:
        raw = json.load(f)
    tgt = raw["target"]
    problem = TripleNullProblem(mapper, pat,
        null_angles_deg=tgt["nullAnglesDeg"],
        null_target_db=tgt["nullTargetDb"],
        sll_target_db=tgt["sllTargetDb"],
        null_window_half_deg=tgt["nullWindowHalfDeg"],
        null_margin_db=tgt["nullMarginDb"],
        weights=tgt.get("weights"),
        **common_kw)
    stop_fitness = None
else:
    raise ValueError(f"未知任务类型: {task}")

print(f"  任务: {task}, n_vars={problem.n_vars} "
      f"(pos={problem.n_pos}, phase={problem.n_phase}, amp={problem.n_amp})")

# ============================================================
# 5. 优化
# ============================================================
# 边界
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

print(f"\n--- CMA-ES 优化 ---")
t0 = time.perf_counter()
result = minimize(
    problem.fitness, problem.n_vars,
    method="cma", bounds=(lb, ub),
    sigma=cfg.opt_sigma,
    pop_size=cfg.opt_pop_size,
    max_iter=cfg.opt_max_iter,
    seed=seed, n_jobs=cfg.opt_n_jobs, verbose=cfg.opt_verbose,
    stop_fitness=stop_fitness or cfg.opt_stop_fitness,
)
elapsed = time.perf_counter() - t0

res = problem.get_result(result["x"])
pos, amps, phases_deg = res["positions"], res["amplitudes"], res["phases_deg"]

print(f"  最优 fitness: {result['f']:.4f}, 耗时: {elapsed:.1f}s")

# ============================================================
# 6. 方向图 + 结果
# ============================================================
af = problem._compute_af(pos, amps, np.deg2rad(phases_deg))
af_abs = np.abs(af)
af_db = 20 * np.log10(np.maximum(af_abs, 1e-30) / np.max(af_abs))
af_db_flat = af_db.reshape(-1, af_db.shape[-1]) if af_db.ndim > 1 else af_db[None, :]

if problem._gd_abs is not None or problem._gd_rel is not None:
    af_abs_flat = af_abs.reshape(-1, af_abs.shape[-1]) if af_abs.ndim > 1 else af_abs[None, :]
    cur_gain = float(20 * np.log10(np.max(af_abs_flat[0]) + 1e-30))
    drop = problem._ref_main_gain_db - cur_gain
    limit_str = f"abs={problem._gd_abs}" if problem._gd_abs else f"rel={problem._gd_rel}"
    print(f"  主瓣增益: {cur_gain:.2f} dB (参考: {problem._ref_main_gain_db:.2f} dB, 下降: {drop:.2f} dB, 限制: {limit_str})")

if problem._dir_abs is not None or problem._dir_rel is not None:
    from antopt import compute_directivity
    af_abs_flat = af_abs.reshape(-1, af_abs.shape[-1]) if af_abs.ndim > 1 else af_abs[None, :]
    t_step = float(abs(problem.pattern.theta_deg[1] - problem.pattern.theta_deg[0]))
    d_val, _ = compute_directivity(af_abs_flat[0],
        [float(problem.pattern.theta_deg[0]), float(problem.pattern.theta_deg[-1]), t_step],
        [0, 0.5, 1])
    if problem._dir_rel is not None:
        limit_str = f"ref={problem._ref_directivity_dbi:.2f} - {problem._dir_rel:.1f} = {problem._ref_directivity_dbi - problem._dir_rel:.2f} dBi"
    else:
        limit_str = f"abs={problem._dir_abs:.1f} dBi"
    print(f"  方向性系数: {d_val:.2f} dBi (限制: {limit_str})")

n_freq, n_scan = len(freqs), len(theta0s)
worst_psll = -np.inf

# 保存
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir = out_dir / "patterns"; csv_dir.mkdir(parents=True, exist_ok=True)

import csv as _csv

for sub in range(af_db_flat.shape[0]):
    si = sub % n_scan; fi = sub // n_scan if n_scan > 0 else 0
    f_ghz = freqs[fi % n_freq]; t0_deg = theta0s[si]
    psll_v, _ = get_psll(af_db_flat[sub], theta)
    if psll_v > worst_psll:
        worst_psll = psll_v

    label = f"f={f_ghz:.4g}GHz, $\\theta_0$={t0_deg:.1f}°"
    fname = f"f{f_ghz:.4g}GHz_theta{t0_deg:.1f}"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, af_db_flat[sub], lw=1.0, label=label)

    # 标注副瓣
    pk_idx, pk_val = _find_peaks(af_db_flat[sub])
    sll_mask = pk_val < -3
    if np.any(sll_mask):
        sll_db = np.max(pk_val[sll_mask])
        sll_theta = theta[pk_idx[sll_mask][np.argmax(pk_val[sll_mask])]]
        ax.plot(sll_theta, sll_db, "ro", markersize=4,
                label=f"SLL: {sll_theta:.2f}°, {sll_db:.2f} dB")

    ax.set_title(f"{label},  PSLL = {psll_v:.2f} dB")
    ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(fig_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV
    with open(csv_dir / f"{fname}.csv", "w", newline="", encoding="utf-8") as cf:
        w = _csv.writer(cf)
        w.writerow(["theta_deg", "pattern_dB"])
        for j in range(len(theta)):
            w.writerow([theta[j], af_db_flat[sub, j]])

print(f"\n  PSLL: {worst_psll:.2f} dB")

# ============================================================
# 7. 保存 JSON + 幅度
# ============================================================
# 位置转回米 (mapper 输出是波长)
lam0_save = 299792458.0 / (freqs[0] * 1e9)
pos_m = pos * lam0_save

out_data = {
    "settings": {**json.load(open(HERE / "Config.json", encoding="utf-8")), "randomSeed": seed},
    "result": {
        "bestFitness": float(result["f"]), "bestPSLL": worst_psll,
        "bestPositions": pos_m.tolist(), "bestPhasesDeg": phases_deg.tolist(),
        "bestAmplitudes": res.get("amplitudes_dB", amps).tolist(),
        "elapsedSeconds": round(elapsed, 1),
    },
}
with open(out_dir / "optResult.json", "w", encoding="utf-8") as f:
    f.write(to_json_flat(out_data))

print(f"  结果已保存: {out_dir}")
