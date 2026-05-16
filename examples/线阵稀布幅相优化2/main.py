"""稀布幅相优化主脚本 — CMA 约束优化 + 断点续跑。

用法: cd examples/线阵稀布幅相优化2 && python main.py
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

from antopt import LMMapper, Pattern, minimize, SidelobeProblem, NullSteeringProblem, get_psll
from antopt.analysis import _find_peaks
from antopt.element_pattern import ElementPattern
from antopt.utils import to_json_flat, load_array_config

# exe 兼容
HERE = Path(sys.argv[0]).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
CKPT_FILE = HERE / ".checkpoint.json"


def _resolve_source(source, file_path, key):
    """解析 source 字段 → (source_value, init_data)。"""
    if source is True or source == "optimize":
        return True, None
    if source is False or source == "default":
        return "default", None
    # 字符串 = 文件路径
    if isinstance(source, str):
        return "file", load_array_config(source, key, HERE)
    raise ValueError(f"{source!r} 不是有效的 source 值")


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

# ── 2. 解析 position / amplitude / phase source ──
pos_cfg = cfg["position"]
amp_cfg = cfg["amplitude"]
phs_cfg = cfg["phase"]

# 位置
pos_source = pos_cfg["source"]
init_pos = None
if isinstance(pos_source, str) and pos_source != "optimize":
    init_pos = load_array_config(pos_source, "xCenters", HERE)
    lam0 = 299792458.0 / (freqs[0] * 1e9)
    init_pos = init_pos / lam0

# 幅度
amp_source_raw = amp_cfg["source"]
amp_source, init_amp = _resolve_source(amp_source_raw, amp_cfg.get("file"), "amplitudes")
amp_bounds = tuple(amp_cfg.get("bounds", [0.0, 1.0]))
amp_half = amp_cfg.get("optimizeHalf", False)

# 相位
phs_source_raw = phs_cfg["source"]
phs_source, init_phs = _resolve_source(phs_source_raw, phs_cfg.get("file"), "phasesDeg")
phs_half = phs_cfg.get("optimizeHalf", False)

# 对称半优化: 幅度或相位任一启用即可
optimize_half = amp_half or phs_half

# ── 3. 阵列 ──
optimize_pos = (pos_source is True)
if optimize_pos:
    Ne = pos_cfg["Ne"]
    L_wl = pos_cfg["L_wavelength"]
    dmin_wl = pos_cfg["dmin_wavelength"]
    is_sym = pos_cfg.get("symmetric", False)
    is_fixed = pos_cfg.get("fixedAperture", False)
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
    epath = Path(eg["csvDirectory"])
    if not epath.is_absolute():
        epath = HERE / epath
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
tgt = cfg["target"]
task = tgt["type"]
common_kw = dict(
    position_source=pos_source,
    amplitude_source=amp_source,
    amplitude_bounds=amp_bounds,
    phase_source=phs_source,
    optimize_half_amp_phase=optimize_half,
    init_positions=init_pos,
    init_phases_deg=init_phs,
    init_amplitudes=init_amp,
    element_patterns=fe_patterns,
    use_pointing_penalty=True,
)

if task == "sidelobe":
    problem = SidelobeProblem(mapper, pat,
        target_psll=tgt.get("targetPSLL"),
        target_hpbw=tgt.get("targetHPBW") or 180.0,
        mainlobe_region=None,
        **common_kw)
    stop_fitness = tgt.get("targetPSLL")
elif task == "null":
    problem = NullSteeringProblem(mapper, pat,
        null_angle_deg=tgt["nullAngleDeg"],
        null_target_db=tgt["nullTargetDb"],
        sll_target_db=tgt["sllTargetDb"],
        null_window_half_deg=tgt["nullWindowHalfDeg"],
        null_margin_db=tgt["nullMarginDb"],
        weights=tgt.get("weights"),
        **common_kw)
    stop_fitness = None
else:
    raise ValueError(f"未知任务类型: {task}")

print(f"=== 稀布幅相优化 (CMA) ===")
print(f"  任务: {task}")
print(f"  阵列: {Ne}元, 孔径={L_wl:.4f}λ, dmin={dmin_wl:.4f}λ, 对称={is_sym}")
if len(freqs) > 1 or not optimize_pos:
    print(f"  频率: {freqs.tolist()} GHz")
print(f"  位置={'优化' if optimize_pos else '导入'}, "
      f"幅度={amp_cfg['source']}, 相位={phs_cfg['source']}")
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
    stop_fitness=stop_fitness or opt_param.get("stopFitness"),
    checkpoint=str(CKPT_FILE) if resume_enabled else None,
    resume=str(CKPT_FILE) if resume_enabled and CKPT_FILE.exists() else None,
)
elapsed = time.perf_counter() - t0

if resume_enabled and CKPT_FILE.exists():
    CKPT_FILE.unlink()

# ── 8. 结果 ──
res = problem.get_result(result["x"])
pos, amps = res["positions"], res["amplitudes"]
phases_deg = res["phases_deg"]
theta_range = f"θ ∈ [{theta_start}°, {theta_end}°], Δθ = {theta_step}°"

print(f"\n  最优适应度: {result['f']:.4f}, 耗时: {elapsed:.1f}s")

# ── 9. 保存图片 + CSV ──
import csv as _csv
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir = out_dir / "patterns"; csv_dir.mkdir(parents=True, exist_ok=True)
n_freq, n_scan = len(freqs), len(theta0s)
per_dir = fig_dir / "per_pattern"; per_dir.mkdir(exist_ok=True)

# 计算方向图 (复用 problem 的 _compute_af)
af = problem._compute_af(pos, amps, np.deg2rad(phases_deg))
af_db = 20 * np.log10(np.maximum(np.abs(af), 1e-30) / np.max(np.abs(af)))
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
    if multi_freq:
        label = f"f = {f_ghz:.4g} GHz,  $\\theta_0$ = {t0:.1f}°"
    else:
        label = f"$\\theta_0$ = {t0:.1f}°"
    if multi_freq:
        fname = f"f{f_ghz:.4g}GHz_theta{t0:.1f}.png"
    else:
        fname = f"theta{t0:.1f}.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pat.theta_deg, af_db_flat[sub], linewidth=1.0, label=label)

    if task == "null":
        # 零陷任务标注
        null_ang = tgt["nullAngleDeg"]
        null_win = tgt["nullWindowHalfDeg"]
        # 1. 最大副瓣 (排除主瓣内峰值, 幅度 > -3 dB 的峰为主瓣)
        pk_idx, pk_val = _find_peaks(af_db_flat[sub])
        sll_mask = pk_val < -3
        if np.any(sll_mask):
            sll_db = np.max(pk_val[sll_mask])
            sll_theta = pat.theta_deg[pk_idx[sll_mask][np.argmax(pk_val[sll_mask])]]
        else:
            sll_db = np.max(pk_val) if len(pk_val) > 0 else -np.inf
            sll_theta = pat.theta_deg[pk_idx[0]] if len(pk_idx) > 0 else 0
        ax.plot(sll_theta, sll_db, "ro", markersize=4,
                label=f"SLL: {sll_theta:.2f}°, {sll_db:.2f} dB")
        # 2. 零陷中心竖线
        ax.axvline(null_ang, color="r", linestyle="--", linewidth=0.8, alpha=0.7,
                   label=f"Null @ {null_ang:.1f}°")
        # 3. 零陷窗口两侧竖线
        win_lo, win_hi = null_ang - null_win, null_ang + null_win
        ax.axvline(win_lo, color="orange", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.axvline(win_hi, color="orange", linestyle=":", linewidth=0.7, alpha=0.7,
                   label=f"Window ±{null_win:.1f}°")
        # 4. 窗口内最高值
        win_mask = (pat.theta_deg >= win_lo) & (pat.theta_deg <= win_hi)
        wmax_db = -np.inf
        if np.any(win_mask):
            win_db = af_db_flat[sub, win_mask]
            win_theta = pat.theta_deg[win_mask]
            wmax_idx = np.argmax(win_db)
            wmax_db = win_db[wmax_idx]
            ax.plot(win_theta[wmax_idx], wmax_db, "s", color="orange", markersize=5,
                    label=f"In-window max: {wmax_db:.1f} dB")
        title = f"{label},  Null: {wmax_db:.1f} dB @ {null_ang:.1f}°±{null_win:.1f}°,  PSLL = {psll_v:.2f} dB\n{theta_range}"
    else:
        # 副瓣任务: 标注最高副瓣 (排除主瓣)
        pk_idx, pk_val = _find_peaks(af_db_flat[sub])
        sll_mask = pk_val < -3
        if np.any(sll_mask):
            sll_db = np.max(pk_val[sll_mask])
            sll_theta = pat.theta_deg[pk_idx[sll_mask][np.argmax(pk_val[sll_mask])]]
            ax.plot(sll_theta, sll_db, "ro", markersize=4,
                    label=f"SLL: {sll_theta:.2f}°, {sll_db:.2f} dB")
        title = f"{label},  PSLL = {psll_v:.2f} dB\n{theta_range}"

    ax.set_title(title)
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

if multi_scan:
    for fi, f_ghz in enumerate(freqs):
        fig, ax = plt.subplots(figsize=(10, 5))
        for si, t0 in enumerate(theta0s):
            ax.plot(pat.theta_deg, af_db_flat[fi * n_scan + si], linewidth=1.0,
                    label=f"$\\theta_0$ = {t0:.1f}°")
        if task == "null":
            null_ang = tgt["nullAngleDeg"]
            null_win = tgt["nullWindowHalfDeg"]
            ax.axvline(null_ang, color="r", linestyle="--", linewidth=0.8, alpha=0.7,
                       label=f"Null @ {null_ang:.1f}°")
            win_lo, win_hi = null_ang - null_win, null_ang + null_win
            ax.axvline(win_lo, color="orange", linestyle=":", linewidth=0.7, alpha=0.7)
            ax.axvline(win_hi, color="orange", linestyle=":", linewidth=0.7, alpha=0.7,
                       label=f"Window ±{null_win:.1f}°")
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

if multi_freq:
    for si, t0 in enumerate(theta0s):
        fig, ax = plt.subplots(figsize=(10, 5))
        for fi, f_ghz in enumerate(freqs):
            ax.plot(pat.theta_deg, af_db_flat[fi * n_scan + si], linewidth=1.0,
                    label=f"f = {f_ghz:.4g} GHz")
        if task == "null":
            null_ang = tgt["nullAngleDeg"]
            null_win = tgt["nullWindowHalfDeg"]
            ax.axvline(null_ang, color="r", linestyle="--", linewidth=0.8, alpha=0.7,
                       label=f"Null @ {null_ang:.1f}°")
            win_lo, win_hi = null_ang - null_win, null_ang + null_win
            ax.axvline(win_lo, color="orange", linestyle=":", linewidth=0.7, alpha=0.7)
            ax.axvline(win_hi, color="orange", linestyle=":", linewidth=0.7, alpha=0.7,
                       label=f"Window ±{null_win:.1f}°")
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
cfg_save = dict(cfg)
cfg_save["randomSeed"] = seed
out_data = {
    "settings": cfg_save,
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

# 保存优化幅度为 txt (无表头, 一列)
np.savetxt(out_dir / "amplitudes.txt", amps, fmt="%.6f")
print(f"  结果已保存: {out_dir}")
