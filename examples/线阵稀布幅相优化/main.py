"""稀布幅相优化主脚本 — CMA 约束优化 + 断点续跑。

用法: cd examples/线阵稀布幅相优化 && python main.py
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

from antopt import LMMapper, Pattern, minimize, SparseArrayProblem, get_psll
from antopt.element_pattern import ElementPattern
from antopt.utils import to_json_flat, load_array_config

# exe 兼容
HERE = Path(sys.argv[0]).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
CKPT_FILE = HERE / ".checkpoint.json"


# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

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

# ── 2. 导入数据 ──
imp = optz.get("import", {})
init_pos = load_array_config(imp.get("positionsFile"), "xCenters", HERE) if p_mode in (0, 2) else None
init_phs = load_array_config(imp.get("phasesFile"), "phasesDeg", HERE) if h_mode == 2 else None
init_amp = load_array_config(imp.get("amplitudesFile"), "amplitudes", HERE) if a_mode == 2 else None
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
    is_sym = is_fixed = False

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

# ── 5. 映射器 + 方向图 ──
mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl, is_symmetric=is_sym,
                   is_fixed_aperture=is_fixed, use_sigmoid=False)
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s, frequenciesGHz=freqs)

print(f"=== 稀布幅相优化 (CMA) ===")
print(f"  阵列: {Ne}元, 孔径={L_wl}λ, dmin={dmin_wl}λ, 对称={is_sym}")
print(f"  mode={mode}, theta0={theta0s.tolist()}, n_vars={mapper.n_vars}")

# ── 6. 构建适应度 ──
problem = SparseArrayProblem(mapper, pat, mode=mode,
    init_positions=init_pos, init_phases_deg=init_phs, init_amplitudes=init_amp,
    amplitude_bounds=amp_bounds, target_hpbw=hpbw_target,
    element_patterns=fe_patterns, use_pointing_penalty=use_pt_penalty)

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
    seed=cfg.get("randomSeed", 42),
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
pos, amps = res["positions"], res["amplitudes"]
phases_deg = res["phases_deg"]
theta_range = f"θ ∈ [{theta_start}°, {theta_end}°], Δθ = {theta_step}°"

print(f"\n  最优 PSLL: {result['f']:.4f} dB, 耗时: {elapsed:.1f}s")

# ── 9. 保存图片 + CSV ──
import csv as _csv
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir = out_dir / "patterns"; csv_dir.mkdir(parents=True, exist_ok=True)
n_freq, n_scan = len(freqs), len(theta0s)
per_dir = fig_dir / "per_pattern"; per_dir.mkdir(exist_ok=True)

# 计算方向图 (对称 / 非对称) — 带幅度和相位
phases_rad = np.deg2rad(phases_deg)
if mapper.is_symmetric:
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
af_db = 20 * np.log10(np.abs(af) / np.max(np.abs(af)))
af_db_flat = af_db.reshape(-1, af_db.shape[-1]) if af_db.ndim > 1 else af_db[None, :]

multi_freq = n_freq > 1
multi_scan = n_scan > 1
if multi_scan: (fig_dir / "by_freq").mkdir(exist_ok=True)
if multi_freq: (fig_dir / "by_angle").mkdir(exist_ok=True)

worst_psll = -np.inf
for sub in range(af_db_flat.shape[0]):
    si = sub % n_scan; fi = sub // n_scan if n_scan > 0 else 0
    f_ghz = freqs[fi % n_freq]; t0 = theta0s[si]
    psll_v, _ = get_psll(af_db_flat[sub], pat.theta_deg)
    if psll_v > worst_psll: worst_psll = psll_v
    # 标题: 单频省略频率
    if multi_freq:
        label = f"f = {f_ghz:.4g} GHz,  $\\theta_0$ = {t0:.1f}°"
    else:
        label = f"$\\theta_0$ = {t0:.1f}°"
    # 文件名: 单频省略频率前缀
    if multi_freq:
        fname = f"f{f_ghz:.4g}GHz_theta{t0:.1f}.png"
    else:
        fname = f"theta{t0:.1f}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pat.theta_deg, af_db_flat[sub], linewidth=1.0)
    ax.set_title(f"{label},  PSLL = {psll_v:.2f} dB\n{theta_range}")
    ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3)
    fig.savefig(per_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 保存 CSV
    csv_name = fname.replace(".png", ".csv")
    with open(csv_dir / csv_name, "w", newline="", encoding="utf-8") as cf:
        w = _csv.writer(cf)
        w.writerow(["theta_deg", "pattern_dB"])
        for j in range(len(pat.theta_deg)):
            w.writerow([pat.theta_deg[j], af_db_flat[sub, j]])

# by_freq: 单频多角度 → 一张图, 多频多角度 → 每频率一张
if multi_scan:
    for fi, f_ghz in enumerate(freqs):
        fig, ax = plt.subplots(figsize=(10, 5))
        for si, t0 in enumerate(theta0s):
            ax.plot(pat.theta_deg, af_db_flat[fi * n_scan + si], linewidth=1.0,
                    label=f"$\\theta_0$ = {t0:.1f}°")
        if multi_freq:
            t = f"f = {f_ghz:.4g} GHz\n{theta_range}"
            fn = f"f{f_ghz:.4g}GHz.png"
        else:
            t = f"$\\theta_0$ scan\n{theta_range}"
            fn = "all_angles.png"
        ax.set_title(t); ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
        ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.savefig(fig_dir / "by_freq" / fn, dpi=150, bbox_inches="tight")
        plt.close(fig)

# by_angle: 多频率角度 → 每角度一张, 单角度多频 → 一张
if multi_freq:
    for si, t0 in enumerate(theta0s):
        fig, ax = plt.subplots(figsize=(10, 5))
        for fi, f_ghz in enumerate(freqs):
            ax.plot(pat.theta_deg, af_db_flat[fi * n_scan + si], linewidth=1.0,
                    label=f"f = {f_ghz:.4g} GHz")
        if multi_scan:
            t = f"$\\theta_0$ = {t0:.1f}°\n{theta_range}"
            fn = f"theta{t0:.1f}deg.png"
        else:
            t = f"$\\theta_0$ = {t0:.1f}°\n{theta_range}"
            fn = "all_freqs.png"
        ax.set_title(t); ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
        ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.savefig(fig_dir / "by_angle" / fn, dpi=150, bbox_inches="tight")
        plt.close(fig)

# ── 10. 保存 JSON ──
out_data = {
    "settings": cfg,
    "result": {
        "bestFitness": float(result["f"]), "bestPSLL": worst_psll,
        "bestX": result["x"].tolist(),
        "bestPositions": pos.tolist(),
        "bestPhasesDeg": phases_deg.tolist(), "bestAmplitudes": amps.tolist(),
        "optimizer": "cma", "elapsedSeconds": round(elapsed, 1),
    },
}
with open(out_dir / "optResult.json", "w", encoding="utf-8") as f:
    f.write(to_json_flat(out_data))
print(f"  结果已保存: {out_dir}")
