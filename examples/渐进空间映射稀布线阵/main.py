"""渐进空间映射稀布线阵 — 粗模型(方向图乘积) + 细模型(HFSS) 迭代优化。

用法: cd examples/渐进空间映射稀布线阵 && python main.py
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = Path(__file__).resolve().parent

from antopt import LMMapper, Pattern, run_optimization
from antopt.element_pattern import ElementPattern
from antopt.utils import to_json_flat
from antopt.space_mapping import run_space_mapping
from script.Run_Patch import run_patch_simulation

# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

freq_ghz = cfg["frequencyGHz"]
seed = cfg["randomSeed"]
asp = cfg["aspectAngle"]
theta_start = asp["thetaStartDeg"]; theta_end = asp["thetaEndDeg"]
theta_step = asp["thetaStepDeg"]
theta0s = np.array(asp["theta0sDeg"], dtype=float)

optz = cfg["target"]
mode = optz["mode"]
amp_bounds = tuple(optz["amplitudeBounds"])
hpbw_target = optz.get("targetHPBW") or 180.0
use_pt_penalty = optz.get("mainLobePointingPenalty", True)

p_mode = (mode // 100) % 10

# ── 2. 天线阵列 ──
arr = cfg["antennaArray"]
Ne, L, dmin = arr["Ne"], arr["L_wavelength"], arr["dmin_wavelength"]
is_sym = arr.get("isSymmetryArray", False)

optimize_pos = (p_mode == 1)
if not optimize_pos:
    raise ValueError("空间映射需要优化位置 (mode 百位=1)")

# ── 3. 单元方向图 ──
fe_patterns = None
eg = cfg.get("ePattern", {})
if eg.get("enabled") and eg.get("csvDirectory"):
    epath = HERE / eg["csvDirectory"]
    if epath.exists():
        fe_patterns = ElementPattern.from_hfss_multi_freq(
            str(epath), np.array([freq_ghz]),
            np.arange(theta_start, theta_end + theta_step/2, theta_step),
            eg.get("degStep") or 0.01,
            is_gain=eg.get("isGain", True), in_dB=eg.get("inDB", False),
            input_theta_range=tuple(eg.get("thetaRange", (-180.0, 180.0))),
        )
        print(f"  加载单元方向图: {len(fe_patterns)} 个频率")

# ── 4. 构造粗模型 ──
coarse_opt_cfg = cfg["coarse"]["optimizer"]
coarse_method = coarse_opt_cfg.get("method", "cma")
coarse_cfg = coarse_opt_cfg.get(coarse_method, coarse_opt_cfg.get("cma", {}))

use_sigmoid = (coarse_method != "cma_cpp")
mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=is_sym, use_sigmoid=use_sigmoid)
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s,
              frequenciesGHz=np.array([freq_ghz]))

print(f"=== 渐进空间映射稀布线阵 ===")
print(f"  阵列: {Ne}元, 孔径={L}λ, dmin={dmin}λ, 对称={is_sym}")
print(f"  频率: {freq_ghz} GHz, 指向角: {theta0s.tolist()} deg")
print(f"  变量维度: {mapper.n_vars}, 优化器={coarse_method}")

# ── 5. 粗模型优化 (getBestXc) ──
print(f"\n--- Step 1: 粗模型优化 ---")
t0 = time.perf_counter()
coarse_result = run_optimization(
    mapper, pat, method=coarse_method, mode=mode, seed=seed,
    target_hpbw=hpbw_target, use_pointing_penalty=use_pt_penalty,
    element_patterns=fe_patterns, amplitude_bounds=amp_bounds,
    pop_size=coarse_cfg.get("pop_size"), max_iter=coarse_cfg.get("max_iter", 1000),
    sigma0=coarse_cfg.get("sigma", 1.0), n_jobs=coarse_cfg.get("n_jobs", 0),
    verbose=coarse_cfg.get("verbose", True),
    verbose_interval=coarse_cfg.get("verboseInterval"),
    stop_fitness=coarse_cfg.get("stopFitness"),
)
Xc_star = coarse_result["x"]
print(f"  粗模型最优 PSLL: {coarse_result['f']:.4f} dB")
print(f"  耗时: {time.perf_counter() - t0:.1f}s")

# ── 6. 空间映射迭代 ──
pe_opt_cfg = cfg["pe"]["optimizer"]
pe_method = pe_opt_cfg.get("method", "cma")
pe_cfg = pe_opt_cfg.get(pe_method, pe_opt_cfg.get("cma", {}))
asm_cfg = cfg["asm"]

# HFSS 仿真函数 — 参数写死在 main.py 不暴露到 Config
def hfss_func(Xf_var):
    """Xf_var (ℝ^D) → 细模型方向图列表 [Yf]"""
    pos = mapper.synthesize(Xf_var)
    lam0 = 299792458.0 / (freq_ghz * 1e9)
    x_m = pos * lam0  # 波长 → 米
    results_dir = str(HERE / "result")

    # 波束指向相位: C++ phasesRad = -wavenumber * wlpos * lambda * sin(theta0)
    scan_phase_deg = -360.0 * pos * np.sin(np.deg2rad(theta0s[0]))

    _ = run_patch_simulation(
        x_centers=x_m.tolist(),
        phases_deg=scan_phase_deg.tolist(),
        frequency_ghz=freq_ghz,
        results_dir=results_dir,
        array_length_wl=cfg["antennaArray"]["L_wavelength"],
        results_phi_sections=[0],
        theta_start=theta_start, theta_stop=theta_end, theta_step=theta_step,
        phi_start=0, phi_stop=0, phi_step=1.0,
        close_after=True,  # 仿真完关闭释放 COM
    )

    # 读取 HFSS CSV (theta=-180..180, 第二列为 GainTotal)
    import csv, os
    csv_path = os.path.join(results_dir, "(GainTotal).csv")
    yf_full = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过标题
        for row in reader:
            if len(row) >= 2:
                try:
                    yf_full.append(float(row[1]))
                except ValueError:
                    continue
    return [np.array(yf_full)]  # HFSS 已按 theta 范围导出, 无需切片

print(f"\n--- Step 2: 空间映射迭代 ---")
t0 = time.perf_counter()
try:
    asm_result = run_space_mapping(
        Xc_star, mapper, pat, fe_patterns, hfss_func,
        max_asm_iter=asm_cfg["maxIterations"],
        target_obj=asm_cfg["targetFineObj"],
        target_res_norm=asm_cfg["targetResNorm"],
        pop_size=pe_cfg["pop_size"],
        pe_max_iter=pe_cfg["max_iter"],
        sigma=pe_cfg["sigma"],
        seed=seed,
        verbose=True,
    )
except Exception as e:
    print(f"\n  ASM 异常: {e}")
    # 兜底: 只保存粗模型响应
    pos = mapper.synthesize(Xc_star)
    if mapper.is_symmetric:
        hN = mapper._halfNe
        offset = 1 if mapper.has_center else 0
        af = pat.linear_af_symmetric(pos[hN + offset:], has_center=mapper.has_center)
    else:
        af = pat.linear_af(pos)
    af_abs = np.abs(af)
    yc0 = 20.0 * np.log10(np.maximum(af_abs, 1e-30) / np.max(af_abs))
    asm_result = {"Xf": Xc_star, "history": [], "yc_star_db": yc0.tolist()}
elapsed = time.perf_counter() - t0

if asm_result["history"]:
    best = min(asm_result["history"], key=lambda h: h["PSLL"])
    print(f"\n=== 空间映射完成 ===")
    print(f"  最优 PSLL: {best['PSLL']:.2f} dB (iter {best['iter']})")
else:
    best = {"PSLL": float("nan"), "iter": 0}

# ── 7. 保存结果 ──
import matplotlib.pyplot as plt
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
out_dir.mkdir(parents=True, exist_ok=True)

fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
theta = pat.theta_deg
yc_star = np.array(asm_result["yc_star_db"])
# 始终保存粗模型最优方向图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, yc_star, 'b-', linewidth=1.0)
ax.set_title(f"Coarse Optimum (PSLL={coarse_result['f']:.2f} dB)")
ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3)
fig.savefig(fig_dir / "coarse_optimum.png", dpi=150, bbox_inches="tight")
plt.close(fig)
# 每代对比图
for h in asm_result["history"]:
    k = h["iter"]
    yc_xe = np.array(h["yc_db"])   # PE 响应
    yf = np.array(h["yf_db"])      # HFSS 响应
    has_pe = (k >= 1)              # 第一代无 PE
    ncols = 3 if has_pe else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4.5))
    if not has_pe:
        axes = [axes[0], axes[1]]  # 2-panel 统一索引
    # Panel 1: 粗模型最优
    axes[0].plot(theta, yc_star, 'b-', linewidth=1.0)
    axes[0].set_title(f"Coarse Optimum (Xc*)")
    axes[0].set_ylim(-60, 3); axes[0].grid(True, alpha=0.3)
    # Panel 2: PE (仅 iter≥1)
    if has_pe:
        axes[1].plot(theta, yc_xe, 'g-', linewidth=1.0)
        axes[1].set_title(f"PE (Xe)")
        axes[1].set_ylim(-60, 3); axes[1].grid(True, alpha=0.3)
    # Panel 3 (or 2): HFSS
    ax_fine = axes[-1]
    ax_fine.plot(theta, yf, 'r-', linewidth=1.0)
    ax_fine.set_title(f"HFSS Fine (PSLL={h['PSLL']:.2f} dB)")
    ax_fine.set_ylim(-60, 3); ax_fine.grid(True, alpha=0.3)
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_xlabel("$\\theta$ (deg)")
        ax.set_ylabel("Norm. Pattern (dB)")
    fig.suptitle(f"ASM Iter {k}", fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / f"iter{k:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
print(f"  方向图已保存: {fig_dir}")

# ── 8. 保存 JSON ──
out_data = {
    "settings": cfg,
    "coarse": {
        "bestFitness": float(coarse_result["f"]),
        "bestPositions": coarse_result["result"]["positions"].tolist(),
    },
    "asm": {
        "bestFitness": best["PSLL"],
        "history": [{k: float(v) if isinstance(v, (np.floating, float))
                      else v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in h.items()} for h in asm_result["history"]],
    },
}

with open(out_dir / "optResult.json", "w", encoding="utf-8") as f:
    f.write(to_json_flat(out_data))
print(f"\n  结果已保存: {out_dir}")
