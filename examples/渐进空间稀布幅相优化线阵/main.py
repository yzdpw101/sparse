"""渐进空间映射 (ASM) 稀布幅相优化线阵。

粗模型 (AF×AEP) + 细模型 (HFSS) 迭代对齐。
用法: python main.py
"""
import sys, os, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from antopt import (LMMapper, Pattern, SidelobeProblem, minimize, get_psll,
                    load_array_config, to_json_flat)
from antopt.config import ASMConfig
from antopt.hfss_io import read_hfss_csv_db, compute_total_phases, load_aep_patterns
from antopt.plotting import plot_pattern_comparison, plot_convergence
from antopt.space_mapping.asm import parameter_extraction, compute_fine_psll
from antopt.space_mapping.base import calc_response_error
from hfss import run_patch_simulation

HERE = Path(__file__).resolve().parent
lam0 = 299792458.0 / (2.45e9)  # 会在下面被实际频率覆盖

# ============================================================
# 1. 加载配置
# ============================================================
cfg = ASMConfig.from_json(HERE / "Config.json")

freqs = np.sort(np.array(cfg.frequenciesGHz, dtype=float))
theta0s = np.array(cfg.theta0s, dtype=float)
theta = np.arange(cfg.theta_start, cfg.theta_end + cfg.theta_step / 2, cfg.theta_step)
lam0 = 299792458.0 / (freqs[0] * 1e9)
k = 2 * np.pi / lam0
seed = cfg.randomSeed if cfg.randomSeed is not None else int(np.random.randint(0, 2**31))

print(f"=== ASM 稀布幅相优化 ===")
print(f"  频率: {freqs} GHz, θ0={theta0s}")

# ============================================================
# 2. 阵列位置
# ============================================================
init_pos = None
if cfg.position_source and cfg.position_source != "optimize":
    init_pos = load_array_config(cfg.position_source, "xCenters", HERE) / lam0

Ne = len(init_pos)
L_wl = init_pos[-1] - init_pos[0]
dmin_wl = float(np.min(np.diff(init_pos)))
pos_wl = init_pos
print(f"  阵列: {Ne}元, 孔径={L_wl:.2f}λ, dmin={dmin_wl:.2f}λ")

# ============================================================
# 3. 加载 AEP
# ============================================================
fe_patterns = None
if cfg.ep_enabled and cfg.ep_csv_dir:
    epath = Path(cfg.ep_csv_dir)
    if not epath.is_absolute():
        epath = HERE / epath
    if epath.exists():
        fe_patterns = load_aep_patterns(
            epath, freqs, theta, Ne,
            is_gain=cfg.ep_is_gain,
            in_dB=cfg.ep_in_db,
            input_theta_range=cfg.ep_theta_range,
        )
        print(f"  AEP: {len(fe_patterns[0])} 个单元方向图")

# ============================================================
# 4. 映射器 + 问题
# ============================================================
mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl, is_symmetric=False,
                   is_fixed_aperture=True, use_sigmoid=False)
pat = Pattern(theta_deg_start=cfg.theta_start, theta_deg_end=cfg.theta_end,
              theta_deg_step=cfg.theta_step, theta0s_deg=theta0s, frequenciesGHz=freqs)

problem = SidelobeProblem(
    mapper, pat,
    position_source=pos_wl,
    amplitude_source=cfg.amplitude_source,
    amplitude_bounds=cfg.amplitude_bounds,
    phase_source=cfg.phase_source,
    element_patterns=fe_patterns,
    use_pointing_penalty=True,
)
print(f"  n_vars={problem.n_vars} (pos={problem.n_pos}, phase={problem.n_phase}, amp={problem.n_amp})")

# 安全 fitness
_orig_fitness = problem.fitness
def _safe_fitness(x):
    f = _orig_fitness(x)
    return f if np.isfinite(f) else 100.0
problem.fitness = _safe_fitness

# ============================================================
# 5. Step 1: 粗模型优化 (CMA-ES)
# ============================================================
print(f"\n--- Step 1: CMA-ES 粗模型优化 ---")

lb = np.zeros(problem.n_vars); ub = np.ones(problem.n_vars)
cur = 0
if problem.n_pos > 0:
    lb[:problem.n_pos] = 0.0; ub[:problem.n_pos] = 1.0; cur = problem.n_pos
if problem.n_phase > 0:
    lb[cur:cur+problem.n_phase] = 0.0; ub[cur:cur+problem.n_phase] = 2 * np.pi; cur += problem.n_phase
if problem.n_amp > 0:
    lb[cur:] = cfg.amplitude_bounds[0]; ub[cur:] = cfg.amplitude_bounds[1]

t0 = time.perf_counter()
result = minimize(
    problem.fitness, problem.n_vars,
    method="cma", bounds=(lb, ub),
    sigma=cfg.opt_sigma, max_iter=cfg.opt_max_iter,
    seed=seed, n_jobs=cfg.opt_n_jobs, verbose=cfg.opt_verbose,
)
elapsed = time.perf_counter() - t0

Xc_star = result["x"]
res = problem.get_result(Xc_star)
pos, amps, phases_deg = res["positions"], res["amplitudes"], res["phases_deg"]

# 粗模型方向图
yc_db = 20 * np.log10(np.maximum(np.abs(problem._compute_af(pos, amps, np.deg2rad(phases_deg))), 1e-30))
yc_norm = yc_db - np.max(yc_db)
coarse_psll, _ = get_psll(yc_norm, theta)

print(f"  粗模型 fitness: {result['f']:.4f}, PSLL: {coarse_psll:.2f} dB, 耗时: {elapsed:.1f}s")

# ============================================================
# 6. 输出目录
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / f"asm_{Ne}elem_{ts}"
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
csv_dir = out_dir / "patterns"; csv_dir.mkdir(exist_ok=True)

# ============================================================
# 7. Step 2: HFSS 初始细模型仿真
# ============================================================
print(f"\n--- Step 2: HFSS 初始细模型仿真 ---")

hfss_dir = str(out_dir / "hfss")

# HFSS 需要 total phases = 残余 + steering
total_phases_deg = compute_total_phases(pos_wl, phases_deg, theta0s, lam0)

t1 = time.perf_counter()
run_patch_simulation(
    x_centers=(pos_wl * lam0).tolist(),
    magnitudes=amps.tolist(),
    phases_deg=total_phases_deg.tolist(),
    frequency_ghz=freqs[0],
    results_dir=hfss_dir,
    array_width_wl=0.0,
    array_length_wl=L_wl,
    array_margin_wl=0.5,
    airbox_margin_wl=0.25,
    results_phi_sections=[0],
    theta_start=cfg.theta_start, theta_stop=cfg.theta_end, theta_step=cfg.theta_step,
    phi_start=0, phi_stop=0, phi_step=1.0,
    close_after=False,
)
hfss_elapsed = time.perf_counter() - t1
print(f"  HFSS 耗时: {hfss_elapsed:.1f}s")

yf_db = read_hfss_csv_db(hfss_dir)
fine_psll, _ = get_psll(yf_db, theta)
print(f"  细模型 PSLL: {fine_psll:.2f} dB")

# ============================================================
# 8. 粗细对比
# ============================================================
print(f"\n--- Step 3: 粗细对比 ---")
print(f"  粗: {coarse_psll:.2f} dB,  细: {fine_psll:.2f} dB,  差: {abs(fine_psll - coarse_psll):.2f} dB")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, yc_norm, 'b-', lw=1.0, label=f'Coarse  PSLL={coarse_psll:.1f}dB')
ax.plot(theta, yf_db, 'r--', lw=1.0, label=f'Fine (HFSS)  PSLL={fine_psll:.1f}dB')
ax.set_title(f'{Ne}元 ASM: 粗 vs 细')
ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend()
fig.savefig(fig_dir / "coarse_vs_fine.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ============================================================
# 9. Step 4: ASM 迭代
# ============================================================
print(f"\n--- Step 4: ASM 迭代 (最多 {cfg.asm_max_iterations} 轮) ---")

asm_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
hfss_asm_dir = str(out_dir / "hfss_asm" / asm_ts)
os.makedirs(hfss_asm_dir, exist_ok=True)

# 辅助函数
def _coarse_db(x_full):
    """全变量 → 归一化 dB 方向图"""
    p, ph, am = problem._decode(x_full)
    af = problem._compute_af(p, am, ph)
    af_abs = np.abs(af)
    if af.ndim > 1: af_abs = af_abs.reshape(-1, af_abs.shape[-1])
    else: af_abs = af_abs[None, :]
    return 20.0 * np.log10(np.maximum(af_abs, 1e-30) / np.max(af_abs))[0]

def _hfss_eval(x_full):
    """全变量 → HFSS 细模型 dB 方向图"""
    p, ph, am = problem._decode(x_full)
    total_ph = compute_total_phases(p, np.rad2deg(ph), theta0s, lam0)
    arr_len = p[-1] - p[0]
    run_patch_simulation(
        x_centers=(p * lam0).tolist(),
        magnitudes=am.tolist(),
        phases_deg=total_ph.tolist(),
        frequency_ghz=freqs[0],
        results_dir=hfss_asm_dir,
        array_width_wl=0.0,
        array_length_wl=arr_len,
        array_margin_wl=0.5,
        airbox_margin_wl=0.25,
        results_phi_sections=[0],
        theta_start=cfg.theta_start, theta_stop=cfg.theta_end, theta_step=cfg.theta_step,
        phi_start=0, phi_stop=0, phi_step=1.0,
        close_after=False,
    )
    return read_hfss_csv_db(hfss_asm_dir)

def _response_error(yc_db, yf_db, ignore_db=-30.0):
    mask = yc_db > ignore_db
    if not mask.any(): return 0.0
    diff = yc_db[mask] - yf_db[mask]
    return float(np.mean(diff * diff))

def pe_wrapper(yf_db, x0=None):
    """PE: CMA-ES 找全变量使粗模型匹配细模型"""
    def pe_obj(x):
        return _response_error(_coarse_db(x), yf_db)
    r = minimize(pe_obj, problem.n_vars,
        method="cma", bounds=(lb, ub),
        x0=x0 if x0 is not None else Xc_star, sigma=cfg.asm_pe_sigma,
        max_iter=cfg.asm_pe_max_iter, seed=seed, verbose=False,
        stop_fitness=cfg.asm_pe_stop_fitness)
    return r["x"]

# 初始 PE
print(f"  初始 PE...")
xe_init = pe_wrapper(yf_db, x0=Xc_star)
f = xe_init - Xc_star
print(f"  初始残差: {np.linalg.norm(f):.4f}")

B = np.eye(problem.n_vars)
Xf = Xc_star.copy()
asm_hist = [{"iter": 0, "PSLL": fine_psll, "res_norm": float(np.linalg.norm(f))}]

for it in range(1, cfg.asm_max_iterations + 1):
    print(f"\n  --- ASM iter {it}/{cfg.asm_max_iterations} ---")

    # Broyden 步
    try:
        h = np.linalg.solve(B, -f)
    except np.linalg.LinAlgError:
        h = -np.linalg.lstsq(B, f, rcond=None)[0]

    h_norm = np.linalg.norm(h)
    if h_norm > 0.5:
        h = h * 0.5 / h_norm
        print(f"  步长截断: {h_norm:.2f} -> 0.5")

    Xf_new = Xf + h
    Xf_new = np.clip(Xf_new, lb, ub)

    # HFSS
    print(f"  HFSS 仿真中...")
    t_hfss = time.perf_counter()
    yf_db_k = _hfss_eval(Xf_new)
    psll_k, _ = get_psll(yf_db_k, theta)
    print(f"  HFSS 耗时: {time.perf_counter() - t_hfss:.1f}s, PSLL: {psll_k:.2f} dB")

    # 本代对比图
    yc_k = _coarse_db(Xf_new)
    coarse_psll_k, _ = get_psll(yc_k, theta)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, yc_k, "b-", lw=1.0, label=f"Coarse {coarse_psll_k:.1f}dB")
    ax.plot(theta, yf_db_k, "r--", lw=1.0, label=f"Fine {psll_k:.1f}dB")
    ax.set_title(f"ASM iter {it}")
    ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(fig_dir / f"asm_iter{it:02d}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # PE + Broyden
    print(f"  PE...", end="", flush=True)
    t_pe = time.perf_counter()
    xe_k = pe_wrapper(yf_db_k, x0=Xf_new)
    f_new = xe_k - Xc_star
    print(f" done ({time.perf_counter() - t_pe:.1f}s), |f|={np.linalg.norm(f_new):.4f}")

    s = h; y_broy = f_new - f
    ds = np.dot(s, s)
    if ds > 1e-16:
        B += np.outer(y_broy - B @ s, s) / ds

    f = f_new; Xf = Xf_new
    res_norm = float(np.linalg.norm(f))
    asm_hist.append({"iter": it, "PSLL": psll_k, "res_norm": res_norm})
    print(f"  PSLL={psll_k:.2f} dB, res_norm={res_norm:.4f}")

    if psll_k <= cfg.asm_target_fine_psll:
        print(f"  收敛 (PSLL <= {cfg.asm_target_fine_psll} dB)"); break
    if res_norm < cfg.asm_target_res_norm:
        print(f"  收敛 (res_norm < {cfg.asm_target_res_norm})"); break

# ============================================================
# 10. 结果汇总
# ============================================================
print(f"\n{'='*50}")
print(f"  初始 PSLL: {asm_hist[0]['PSLL']:.2f} dB")
print(f"  最终 PSLL: {asm_hist[-1]['PSLL']:.2f} dB")
print(f"  ASM 迭代: {len(asm_hist) - 1} 轮")

# 收敛图
plot_convergence(
    [h["iter"] for h in asm_hist],
    [h["PSLL"] for h in asm_hist],
    xlabel="ASM Iteration", ylabel="Fine PSLL (dB)",
    title=f"{Ne}元 ASM Convergence",
    save_path=fig_dir / "asm_convergence.png",
)
plt.close("all")

# 幅度分布
from antopt.plotting import plot_amplitude_distribution
plot_amplitude_distribution(amps, title=f"优化幅度分布 (PSLL: {asm_hist[-1]['PSLL']:.2f} dB)",
                           save_path=fig_dir / "amplitude_distribution.png")
plt.close("all")

# JSON
import antopt.utils as _utils
asm_data = {
    "settings": {"Ne": Ne, "freq_ghz": freqs[0], "d_wl": dmin_wl,
                 "arr_len_wl": L_wl, "method": "ASM", "n_vars": problem.n_vars},
    "coarse_psll": coarse_psll,
    "initial_fine_psll": asm_hist[0]["PSLL"],
    "final_fine_psll": asm_hist[-1]["PSLL"],
    "asm_history": asm_hist,
    "amplitudes": amps.tolist(),
}
with open(out_dir / "asmResult.json", "w", encoding="utf-8") as f:
    f.write(_utils.to_json_flat(asm_data))

print(f"\n  结果已保存: {out_dir}")
