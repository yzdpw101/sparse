"""50元 AEP + 流形空间映射 (MSM) 测试

配置: 50元, d=0.5λ, fc=2.0GHz, 位置固定, 仅优化相位
流程:
  1. CMA-ES 优化 AEP 粗模型 (50维相位)
  2. HFSS 细模型仿真
  3. MSM 迭代 (替代模型 + CMA-ES + Broyden S 更新)

用法:
  1. 先生成 AEP: python test_aep_50elem.py
  2. 再跑 MSM: python test_msm_50elem.py
"""
import sys, json, time, csv as _csv, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from antopt import LMMapper, Pattern, SidelobeProblem, minimize, get_psll
from antopt.element_pattern import ElementPattern
from antopt.utils import load_array_config
from antopt.space_mapping import run_manifold_mapping
from antopt.analysis import _find_peaks
from hfss import run_patch_simulation

HERE = Path(__file__).resolve().parent

# ============================================================
# 1. 参数
# ============================================================
freq_ghz = 2.45
lam0 = 299792458.0 / (freq_ghz * 1e9)
theta_start, theta_end, theta_step = -90, 90, 0.1
theta0s = np.array([0.0])
seed = 42
theta = np.arange(theta_start, theta_end + theta_step / 2, theta_step)

# 50元阵位置
x_m = np.array(load_array_config(
    "input/array_config/uniform_50_0p5wl_2.45Ghz.json", "xCenters", HERE))
Ne = len(x_m)
pos_wl = x_m / lam0
L_wl = pos_wl[-1] - pos_wl[0]
dmin_wl = float(np.min(np.diff(pos_wl)))
arr_len_wl = 2 * np.max(np.abs(pos_wl))

print(f"=== {Ne}元 AEP + MSM 测试 ===")
print(f"  {Ne}元, {freq_ghz} GHz, d={dmin_wl:.2f}λ, 仅优化相位")
print(f"  阵列长度: {arr_len_wl:.2f}λ")

# ============================================================
# 2. 加载 AEP 方向图
# ============================================================
aep_csv_dir = HERE / "result"
aep_dirs = sorted(aep_csv_dir.glob("aep_*elem_*"))
aep_path = None
for d in reversed(aep_dirs):
    csvs = list((d / "aep_csv").glob("aep_elem_*.csv"))
    if len(csvs) == Ne:
        aep_path = d / "aep_csv"
        break

if aep_path is None:
    print(f"错误: 未找到 {Ne} 元 AEP, 请先运行 test_aep_{Ne}elem.py")
    sys.exit(1)

print(f"  AEP 目录: {aep_path}")

aep_data = ElementPattern.from_hfss_aep(
    str(aep_path), np.array([freq_ghz]), theta, num_elements=Ne,
    oriDegStep=theta_step, is_gain=True, in_dB=False,
    input_theta_range=(theta_start, theta_end))
print(f"  AEP: {len(aep_data[0])} 个单元方向图加载完成")

# ============================================================
# 3. 构建 AEP 粗模型问题
# ============================================================
mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl, is_symmetric=False,
                   is_fixed_aperture=True, use_sigmoid=False)
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s,
              frequenciesGHz=np.array([freq_ghz]))

problem = SidelobeProblem(
    mapper, pat,
    position_source=pos_wl,
    amplitude_source="default",
    amplitude_bounds=(0.0, 1.0),
    phase_source=True,
    element_patterns=aep_data,
)

print(f"  n_vars={problem.n_vars} (phase={problem.n_phase})")

_orig_fitness = problem.fitness
def _safe_fitness(x):
    f = _orig_fitness(x)
    return f if np.isfinite(f) else 100.0
problem.fitness = _safe_fitness

lb = np.zeros(problem.n_vars)
ub = np.ones(problem.n_vars) * 2 * np.pi

# ============================================================
# 4. Step 1: AEP 粗模型优化
# ============================================================
print(f"\n--- Step 1: AEP 粗模型优化 ({Ne}维 CMA-ES) ---")

t0 = time.perf_counter()
result = minimize(
    problem.fitness, problem.n_vars,
    method="cma", bounds=(lb, ub),
    sigma=0.5, max_iter=300, seed=seed, n_jobs=-1, verbose=True,
)
elapsed = time.perf_counter() - t0

Xc_star = result["x"]
res = problem.get_result(Xc_star)
pos, amps = res["positions"], res["amplitudes"]
phases_deg = res["phases_deg"]

print(f"\n  AEP 粗模型最优 fitness: {result['f']:.4f}, 耗时 {elapsed:.1f}s")

yc_db = 20 * np.log10(np.maximum(np.abs(problem._compute_af(pos, amps, np.deg2rad(phases_deg))), 1e-30))
yc_max = np.max(yc_db)
yc_norm = yc_db - yc_max
coarse_psll, coarse_theta = get_psll(yc_norm, theta)
print(f"  粗模型 PSLL: {coarse_psll:.2f} dB @ θ={coarse_theta:.1f}°")

# ============================================================
# 5. Step 2: HFSS 细模型仿真
# ============================================================
print(f"\n--- Step 2: HFSS 细模型仿真 ({Ne}元, 预计较慢) ---")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / f"msm_{Ne}elem_{ts}"
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
hfss_results_dir = str(out_dir / "hfss")

t1 = time.perf_counter()
run_patch_simulation(
    x_centers=(pos * lam0).tolist(),
    magnitudes=amps.tolist(),
    phases_deg=phases_deg.tolist(),
    frequency_ghz=freq_ghz,
    results_dir=hfss_results_dir,
    array_length_wl=arr_len_wl,
    array_margin_wl=0.5,
    airbox_margin_wl=0.25,
    results_phi_sections=[0],
    theta_start=theta_start, theta_stop=theta_end, theta_step=theta_step,
    phi_start=0, phi_stop=0, phi_step=1.0,
    close_after=False,
)
hfss_elapsed = time.perf_counter() - t1
print(f"  HFSS 细模型耗时: {hfss_elapsed:.1f}s ({hfss_elapsed/60:.1f} min)")

hfss_csvs = [f for f in os.listdir(hfss_results_dir) if f.endswith(".csv")]
yf_linear = []
with open(os.path.join(hfss_results_dir, hfss_csvs[0]), "r") as f:
    reader = _csv.reader(f); next(reader, None)
    for row in reader:
        if len(row) >= 2:
            try: yf_linear.append(float(row[1]))
            except: continue
yf_linear = np.array(yf_linear)
yf_db = 10.0 * np.log10(np.maximum(yf_linear, 1e-30) / np.max(yf_linear))
fine_psll, fine_theta = get_psll(yf_db, theta)
print(f"  细模型 PSLL: {fine_psll:.2f} dB @ θ={fine_theta:.1f}°")

# ============================================================
# 6. 粗细对比
# ============================================================
print(f"\n--- Step 3: 粗细对比 ---")
print(f"  粗模型 PSLL: {coarse_psll:.2f} dB,  细模型 PSLL: {fine_psll:.2f} dB")
print(f"  差异: {abs(fine_psll - coarse_psll):.2f} dB")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, yc_norm, 'b-', lw=1.0, label='AEP 粗模型')
ax.plot(theta, yf_db, 'r--', lw=1.0, label='HFSS 细模型')
ax.set_title(f'{Ne}元 AEP 粗 vs HFSS 细  (PSLL: {coarse_psll:.2f} / {fine_psll:.2f} dB)')
ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend()
fig.savefig(fig_dir / "coarse_vs_fine.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ============================================================
# 7. Step 4: MSM 迭代
# ============================================================
print(f"\n--- Step 4: MSM 迭代 (流形空间映射, 最多5轮) ---")

IGNORE_DB = -45.0

def coarse_model_func(x):
    """粗模型: 相位 → 归一化 dB 方向图"""
    af = problem._compute_af(pos, amps, x)
    af_abs = np.abs(af)
    yc = 20.0 * np.log10(np.maximum(af_abs, 1e-30) / np.max(af_abs))
    return np.maximum(yc, IGNORE_DB)

hfss_asm_dir = str(out_dir / "hfss_msm")
os.makedirs(hfss_asm_dir, exist_ok=True)

_call_count = [0]

def hfss_callback(Xf):
    """MSM HFSS 回调"""
    _call_count[0] += 1
    phases = np.rad2deg(Xf) % 360

    yc_x = coarse_model_func(Xf)
    _, extrema = _find_peaks(yc_x)
    c_psll = float(extrema[1]) if len(extrema) >= 2 else -np.inf
    print(f"    [HFSS call #{_call_count[0]}] coarse PSLL={c_psll:.2f} dB")

    t_asm = time.perf_counter()
    run_patch_simulation(
        x_centers=(pos * lam0).tolist(),
        magnitudes=amps.tolist(),
        phases_deg=phases.tolist(),
        frequency_ghz=freq_ghz,
        results_dir=hfss_asm_dir,
        array_length_wl=arr_len_wl,
        array_margin_wl=0.5,
        airbox_margin_wl=0.25,
        results_phi_sections=[0],
        theta_start=theta_start, theta_stop=theta_end, theta_step=theta_step,
        phi_start=0, phi_stop=0, phi_step=1.0,
        close_after=False,
    )
    print(f"    HFSS 耗时: {time.perf_counter() - t_asm:.1f}s")

    csvs = sorted([f for f in os.listdir(hfss_asm_dir) if f.endswith(".csv")])
    yf_l = []
    with open(os.path.join(hfss_asm_dir, csvs[-1])) as fc:
        reader = _csv.reader(fc); next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try: yf_l.append(float(row[1]))
                except: continue
    return [np.array(yf_l)]

# 每代保存粗细对比图
def on_iter_end_cb(iter_k, Xf_new, yc_db, yf_db_iter, psll_fine):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, yc_db, 'b-', lw=1.0, label='AEP 粗模型')
    ax.plot(theta, yf_db_iter, 'r--', lw=1.0, label='HFSS 细模型')
    ax.set_title(f'MSM iter {iter_k}  (Fine PSLL: {psll_fine:.2f} dB)')
    ax.set_xlabel("$\\theta$ (deg)"); ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(-60, 3); ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(fig_dir / f"msm_iter{iter_k:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

msm_result = run_manifold_mapping(
    Xc_star, coarse_model_func, problem.n_vars,
    hfss_callback,
    max_iter=5,
    target_fine_psll=-30.0,
    cma_iter=300,
    sigma=0.3,
    seed=seed,
    verbose=True,
    on_iter_end=on_iter_end_cb,
)

# ============================================================
# 8. 结果汇总
# ============================================================
print(f"\n{'='*50}")
print(f"=== MSM 结果 ===")
print(f"  初始细模型 PSLL: {msm_result['history'][0]['PSLL']:.2f} dB")
print(f"  最终细模型 PSLL: {msm_result['history'][-1]['PSLL']:.2f} dB")
print(f"  MSM 迭代次数: {len(msm_result['history']) - 1}")

iters = [h["iter"] for h in msm_result["history"]]
pss = [h["PSLL"] for h in msm_result["history"]]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(iters, pss, "o-", lw=1.0)
ax.set_xlabel("MSM Iteration"); ax.set_ylabel("Fine PSLL (dB)")
ax.set_title(f"{Ne}元 MSM Convergence (AEP)"); ax.grid(True, alpha=0.3)
fig.savefig(fig_dir / "msm_convergence.png", dpi=150, bbox_inches="tight"); plt.close(fig)

import antopt.utils as _utils
msm_data = {
    "settings": {"Ne": Ne, "freq_ghz": freq_ghz, "d_wl": dmin_wl,
                 "arr_len_wl": arr_len_wl, "method": "manifold_mapping"},
    "coarse_psll": coarse_psll,
    "initial_fine_psll": fine_psll,
    "msm_history": msm_result["history"],
    "final_fine_psll": msm_result["history"][-1]["PSLL"],
}
with open(out_dir / "msmResult.json", "w", encoding="utf-8") as f:
    f.write(_utils.to_json_flat(msm_data))

print(f"\n  全部结果已保存: {out_dir}")
