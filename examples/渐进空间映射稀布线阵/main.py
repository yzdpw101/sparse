"""渐进空间映射稀布线阵 — 粗模型(方向图乘积) + 细模型(HFSS) 迭代优化。

用法: cd examples/渐进空间映射稀布线阵 && python main.py
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "script"))

import warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = Path(__file__).resolve().parent

from antopt import LMMapper, Pattern, run_optimization
from antopt.element_pattern import ElementPattern
from antopt.utils import load_array_config, compute_pattern, to_json_flat
from antopt.space_mapping import run_space_mapping
from Run_Patch import run_patch_simulation

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
mapper = LMMapper(Ne=Ne, L=L, dmin=dmin, is_symmetric=is_sym)
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s,
              frequenciesGHz=np.array([freq_ghz]))

print(f"=== 渐进空间映射稀布线阵 ===")
print(f"  阵列: {Ne}元, 孔径={L}λ, dmin={dmin}λ, 对称={is_sym}")
print(f"  频率: {freq_ghz} GHz, 指向角: {theta0s.tolist()} deg")
print(f"  变量维度: {mapper.n_vars}")

# ── 5. 粗模型优化 (getBestXc) ──
cma_cfg = cfg["coarse"]["optimizer"]["cma"]
print(f"\n--- Step 1: 粗模型优化 ---")
t0 = time.perf_counter()
coarse_result = run_optimization(
    mapper, pat, method="cma", mode=mode, seed=seed,
    target_hpbw=hpbw_target, use_pointing_penalty=use_pt_penalty,
    element_patterns=fe_patterns, amplitude_bounds=amp_bounds,
    pop_size=cma_cfg["pop_size"], max_iter=cma_cfg["max_iter"],
    sigma0=cma_cfg["sigma"], n_jobs=cma_cfg.get("n_jobs", 0),
    verbose=cma_cfg.get("verbose", True),
)
Xc_star = coarse_result["x"]
print(f"  粗模型最优 PSLL: {coarse_result['f']:.4f} dB")
print(f"  耗时: {time.perf_counter() - t0:.1f}s")

# ── 6. 空间映射迭代 ──
pe_cfg = cfg["pe"]["optimizer"]["cma"]
asm_cfg = cfg["asm"]
hfss_cfg = cfg["hfss"]

# HFSS 仿真函数
def hfss_func(Xf_var):
    """Xf_var (ℝ^D) → 细模型方向图列表 [Yf]"""
    pos = mapper.synthesize(Xf_var)
    lam0 = 299792458.0 / (freq_ghz * 1e9)
    x_m = pos * lam0  # 波长 → 米

    results_dir = str(HERE / hfss_cfg.get("resultsDir", "result"))
    oDes, oProj, oDesk = run_patch_simulation(
        x_centers=x_m.tolist(),
        frequency_ghz=freq_ghz,
        results_dir=results_dir,
        array_length_wl=cfg["antennaArray"]["L_wavelength"],
        run_simulation=hfss_cfg.get("runSimulation", True),
        close_after=hfss_cfg.get("closeAfter", False),
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
    yf_arr = np.array(yf_full)
    # 切片到粗模型 theta 范围
    idx_start = int(round((theta_start - (-180.0)) / theta_step))
    idx_end = idx_start + len(pat.theta_deg)
    yf_arr = yf_arr[idx_start:idx_end]
    return [yf_arr]

print(f"\n--- Step 2: 空间映射迭代 ---")
t0 = time.perf_counter()
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
elapsed = time.perf_counter() - t0

# ── 7. 结果 ──
best = min(asm_result["history"], key=lambda h: h["PSLL"])
print(f"\n=== 空间映射完成 ===")
print(f"  最优 PSLL: {best['PSLL']:.2f} dB (iter {best['iter']})")
print(f"  粗模型 PSLL: {coarse_result['f']:.2f} dB")
print(f"  总耗时: {elapsed:.1f}s")

# ── 8. 保存 ──
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
out_dir.mkdir(parents=True, exist_ok=True)

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
