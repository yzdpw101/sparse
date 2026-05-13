"""渐进空间映射稀布幅相优化线阵

粗模型(方向图乘积) + 细模型(HFSS) 迭代优化。
所有 ASM 逻辑内联，不依赖 antopt/space_mapping.py。

用法: cd examples/渐进空间映射稀布幅相优化线阵 && python main.py
"""
import sys, json, time, os, csv
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
from antopt.analysis import _find_peaks
from antopt.utils import to_json_flat
from script.Run_Patch import run_patch_simulation

HERE = Path(__file__).resolve().parent


# ============================================================
#  1. 加载配置
# ============================================================
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

freqs = np.sort(np.array(cfg["frequenciesGHz"], dtype=float))  # 升序，最小在前
asp = cfg["aspectAngle"]
_ta = asp["thetaDeg"]  # [start, end, step]
theta_start, theta_end, theta_step = _ta[0], _ta[1], _ta[2]
theta0s = np.array(asp["theta0sDeg"], dtype=float)

# 种子校验: null→随机, 负整数→警告+随机, 浮点→警告+随机, 正整数→使用
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

tgt = cfg["target"]
mode = tgt["mode"]
amp_bounds = tuple(tgt["amplitudeBounds"])
hpbw_target = tgt.get("targetHPBW") or 180.0
use_pt_penalty = tgt.get("mainLobePointingPenalty", True)
sym_half = tgt.get("optimizeHalfAmpPhase", False)

p_mode = (mode // 100) % 10
h_mode = (mode // 10) % 10
a_mode = mode % 10

# ============================================================
#  2. 天线阵列参数
# ============================================================
arr = cfg["antennaArray"]
Ne = arr["Ne"]
L_wl = arr["L_wavelength"]
dmin_wl = arr["dmin_wavelength"]
is_sym = arr.get("isSymmetryArray", False)
is_fixed = arr.get("isFixedAperture", False)

# 导入位置 (mode 百位=2)
init_positions = None
init_phases_deg = None
init_amplitudes = None
if p_mode == 2:
    pos_file = HERE / arr["positionsFile"]
    with open(pos_file, encoding="utf-8") as f:
        pos_data = json.load(f)
    init_positions = np.array(pos_data["xCenters"])
    init_phases_deg = np.array(pos_data.get("phasesDeg", [0]*Ne))
    init_amplitudes = np.array(pos_data.get("amplitudes", [1]*Ne))
    print(f"  导入位置: {len(init_positions)} 元, 范围 [{init_positions.min():.3f}, {init_positions.max():.3f}] λ")

# ============================================================
#  3. 单元方向图
# ============================================================
fe_patterns = None
eg = cfg.get("ePattern", {})
if eg.get("enabled") and eg.get("csvDirectory"):
    epath = HERE / eg["csvDirectory"]
    if epath.exists():
        _ep = eg.get("thetaDeg", [-180, 180, 0.01])
        ep_start, ep_end, ep_step = _ep[0], _ep[1], _ep[2]
        fe_patterns = ElementPattern.from_hfss_multi_freq(
            str(epath), freqs,
            np.arange(theta_start, theta_end + theta_step / 2, theta_step),
            ep_step,
            is_gain=eg.get("isGain", True),
            in_dB=eg.get("inDB", False),
            input_theta_range=(ep_start, ep_end),
        )
        print(f"  加载单元方向图: {len(fe_patterns)} 个频率")

# ============================================================
#  4. 构造粗模型组件
# ============================================================
mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl,
                  is_symmetric=is_sym, is_fixed_aperture=is_fixed,
                  use_sigmoid=False)
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s,
              frequenciesGHz=freqs)

print(f"=== 渐进空间映射稀布幅相优化线阵 ===")
print(f"  阵列: {Ne}元, 孔径={L_wl:.4f}λ, dmin={dmin_wl:.4f}λ, 对称={is_sym}, 固定孔径={is_fixed}")
print(f"  频率: {freqs.tolist()} GHz, 指向角: {theta0s.tolist()} deg")
print(f"  种子: {seed if seed is not None else '随机'}")
print(f"  模式: {mode} (位置={'优化' if p_mode==1 else '导入' if p_mode==2 else '默认'}, "
      f"相位={'优化' if h_mode==1 else '导入' if h_mode==2 else '默认'}, "
      f"幅度={'优化' if a_mode==1 else '导入' if a_mode==2 else '默认'})")

# ============================================================
#  5. 粗模型优化 — Step 1
# ============================================================
print(f"\n--- Step 1: 粗模型优化 (mode={mode}) ---")

problem = SparseArrayProblem(
    mapper, pat, mode=mode,
    init_positions=init_positions,
    init_phases_deg=init_phases_deg,
    init_amplitudes=init_amplitudes,
    amplitude_bounds=amp_bounds,
    target_hpbw=hpbw_target,
    element_patterns=fe_patterns,
    use_pointing_penalty=use_pt_penalty,
    optimize_half_amp_phase=sym_half,
)

print(f"  变量维度: {problem.n_vars} (位置={problem.n_pos}, 相位={problem.n_phase}, 幅度={problem.n_amp})")

# 构造 bounds
lb = np.zeros(problem.n_vars)
ub = np.ones(problem.n_vars)
cur = 0
if problem.n_pos > 0:
    lb[cur:cur+problem.n_pos] = 0.0
    ub[cur:cur+problem.n_pos] = 1.0
    cur += problem.n_pos
if problem.n_phase > 0:
    lb[cur:cur+problem.n_phase] = 0.0
    ub[cur:cur+problem.n_phase] = 2 * np.pi
    cur += problem.n_phase
if problem.n_amp > 0:
    lb[cur:] = amp_bounds[0]
    ub[cur:] = amp_bounds[1]

opt_cfg = cfg["optimizer"]
CKPT_FILE = HERE / ".checkpoint.json"
resume_enabled = opt_cfg.get("resume", False)

t0 = time.perf_counter()
result = minimize(
    problem.fitness, problem.n_vars,
    method=opt_cfg.get("method", "cma"),
    bounds=(lb, ub),
    sigma=opt_cfg.get("sigma", 0.5),
    pop_size=opt_cfg.get("pop_size") or None,
    max_iter=opt_cfg.get("max_iter", 1000),
    seed=seed,
    n_jobs=opt_cfg.get("n_jobs", -1),
    verbose=opt_cfg.get("verbose", True),
    init="chaos",
    stop_fitness=opt_cfg.get("stopFitness"),
    checkpoint=str(CKPT_FILE) if resume_enabled else None,
    resume=str(CKPT_FILE) if resume_enabled and CKPT_FILE.exists() else None,
)
elapsed = time.perf_counter() - t0

if resume_enabled and CKPT_FILE.exists():
    CKPT_FILE.unlink()

print(f"\n  粗模型最优 PSLL: {result['f']:.4f} dB")
print(f"  耗时: {elapsed:.1f}s")

# ============================================================
#  6. 解码结果
# ============================================================
res = problem.get_result(result["x"])
Xc_star = result["x"]  # 粗模型最优变量向量
pos = res["positions"]           # Ne 个位置 (λ)
amps = res["amplitudes"]         # Ne 个幅度
phases_deg = res["phases_deg"]   # Ne 个相位 (度)

print(f"\n  位置 (λ): {pos[:5].round(3)} ... {pos[-5:].round(3)}")
print(f"  相位 (°): {phases_deg[:5].round(1)} ... {phases_deg[-5:].round(1)}")
print(f"  幅度: {amps[:5].round(3)} ... {amps[-5:].round(3)}")

# ============================================================
#  7. 计算粗模型方向图
# ============================================================
def compute_coarse_pattern(x_var):
    """从优化变量计算粗模型归一化 dB 方向图。"""
    # 位置
    if p_mode == 1:
        p = mapper.synthesize(x_var[:problem.n_pos])
    else:
        p = init_positions
    # 相位
    if h_mode == 1:
        raw = x_var[problem.n_pos:problem.n_pos + problem.n_phase]
        if problem.optimize_half_amp_phase:
            ph = problem._expand_half(raw)
        else:
            ph = raw.copy()
    elif h_mode == 2:
        ph = np.deg2rad(init_phases_deg)
    else:
        ph = np.zeros(Ne)
    # 幅度
    if a_mode == 1:
        raw = x_var[-problem.n_amp:]
        if problem.optimize_half_amp_phase:
            a = problem._expand_half(raw)
        else:
            a = raw.copy()
    elif a_mode == 2:
        a = init_amplitudes
    else:
        a = np.ones(Ne)

    # 计算 AF (与 fitness() 路径一致)
    if mapper.is_symmetric and problem.optimize_half_amp_phase:
        hN = mapper._halfNe
        offset = 1 if mapper.has_center else 0
        kw = dict(has_center=mapper.has_center,
                  amplitudes=a[hN + offset:], phases=ph[hN + offset:])
        if mapper.has_center:
            kw["center_amplitude"] = a[hN]
            kw["center_phase"] = ph[hN]
        af = pat.linear_af_symmetric(p[hN + offset:], **kw)
    else:
        af = pat.linear_af(p, a, ph)
    af_abs = np.abs(af)
    if fe_patterns is not None:
        af_abs = af_abs * fe_patterns[0]
    return 20.0 * np.log10(np.maximum(af_abs, 1e-30) / np.max(af_abs))


yc_star_db = compute_coarse_pattern(Xc_star)
coarse_psll, coarse_psll_theta = get_psll(yc_star_db, pat.theta_deg)
print(f"  粗模型方向图 PSLL: {coarse_psll:.2f} dB @ θ={coarse_psll_theta:.1f}°")

# ============================================================
#  8. 保存结果
# ============================================================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / ts
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir = out_dir / "patterns"
csv_dir.mkdir(parents=True, exist_ok=True)

theta = pat.theta_deg

# 粗模型方向图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, yc_star_db, 'b-', linewidth=1.0)
ax.set_title(f"Coarse Optimum (PSLL={coarse_psll:.2f} dB)")
ax.set_xlabel("$\\theta$ (deg)")
ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3)
ax.grid(True, alpha=0.3)
fig.savefig(fig_dir / "coarse_optimum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# JSON 结果
cfg_save = dict(cfg)
cfg_save["randomSeed"] = seed  # 替换为实际使用的种子值
out_data = {
    "settings": cfg_save,
    "coarse": {
        "bestFitness": float(result["f"]),
        "bestX": result["x"].tolist(),
        "bestPositions": pos.tolist(),
        "bestAmplitudes": amps.tolist(),
        "bestPhases_deg": phases_deg.tolist(),
    },
    "elapsed_seconds": elapsed,
}
with open(out_dir / "optResult.json", "w", encoding="utf-8") as f:
    f.write(to_json_flat(out_data))

print(f"\n  结果已保存: {out_dir}")

# ============================================================
#  Step 2: HFSS 细模型仿真
# ============================================================
print(f"\n--- Step 2: HFSS 细模型仿真 ---")

lam0 = 299792458.0 / (freqs[0] * 1e9)  # 波长 (m)
x_m = pos * lam0  # 波长 → 米

hfss_results_dir = str(out_dir / "hfss")
run_patch_simulation(
    x_centers=x_m.tolist(),
    phases_deg=phases_deg.tolist(),  # 优化后的相位
    frequency_ghz=freqs[0],
    results_dir=hfss_results_dir,
    array_length_wl=L_wl,
    results_phi_sections=[0],
    theta_start=theta_start, theta_stop=theta_end, theta_step=theta_step,
    phi_start=0, phi_stop=0, phi_step=1.0,
    close_after=False,
)

# 读取 HFSS CSV (GainTotal, 线性功率)
csv_files = [f for f in os.listdir(hfss_results_dir) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"HFSS 导出 CSV 未找到: {hfss_results_dir}")
csv_path = os.path.join(hfss_results_dir, csv_files[0])
print(f"  读取: {csv_path}")
yf_linear = []
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    next(reader, None)  # 跳过标题行
    for row in reader:
        if len(row) >= 2:
            try:
                yf_linear.append(float(row[1]))
            except ValueError:
                continue
yf_linear = np.array(yf_linear)
# GainTotal 是功率量 → 10*log10 (不是 20*log10)
yf_db = 10.0 * np.log10(np.maximum(yf_linear, 1e-30) / np.max(yf_linear))
fine_psll, fine_psll_theta = get_psll(yf_db, pat.theta_deg)
print(f"  细模型 PSLL: {fine_psll:.2f} dB @ θ={fine_psll_theta:.1f}°")

# ============================================================
#  Step 3: 粗细对比
# ============================================================
print(f"\n--- Step 3: 粗细对比 ---")
print(f"  粗模型 PSLL: {coarse_psll:.2f} dB,  细模型 PSLL: {fine_psll:.2f} dB")
print(f"  差异: {abs(fine_psll - coarse_psll):.2f} dB")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, yc_star_db, 'b-', linewidth=1.0, label='Coarse (pattern product)')
ax.plot(theta, yf_db, 'r--', linewidth=1.0, label='Fine (HFSS)')
ax.set_title(f'Coarse vs Fine  (PSLL: {coarse_psll:.2f} / {fine_psll:.2f} dB)')
ax.set_xlabel("$\\theta$ (deg)")
ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3)
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(fig_dir / "coarse_vs_fine.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  对比图已保存: {fig_dir / 'coarse_vs_fine.png'}")

# ============================================================
#  Step 4: ASM 迭代
# ============================================================
print(f"\n--- Step 4: ASM 迭代 ---")

asm_cfg = cfg["asm"]
pe_cfg = asm_cfg["pe"]


def hfss_eval(Xc_var):
    """粗模型变量 → HFSS 细模型归一化 dB 方向图。

    Xc_var: 粗模型变量 (5D if optimize_half_amp_phase else 10D)
    返回: 归一化 dB 方向图 (与 pat.theta_deg 同尺寸)
    """
    # 位置 (从 mapper 合成)
    p = problem.mapper.synthesize(Xc_var[:problem.n_pos]) if problem.n_pos > 0 else init_positions
    # 相位: 粗模型变量 → 全阵元
    if problem.optimize_half_amp_phase:
        raw = Xc_var[problem.n_pos:problem.n_pos + problem.n_phase]
        ph_full = problem._expand_half(raw)
    else:
        ph_full = Xc_var[problem.n_pos:problem.n_pos + problem.n_phase]
    phases_deg_full = np.rad2deg(ph_full)

    # 调用 HFSS
    x_m = (p * lam0).tolist()
    run_patch_simulation(
        x_centers=x_m, phases_deg=phases_deg_full.tolist(),
        frequency_ghz=freqs[0],
        results_dir=str(hfss_asm_dir),
        array_length_wl=L_wl, results_phi_sections=[0],
        theta_start=theta_start, theta_stop=theta_end, theta_step=theta_step,
        phi_start=0, phi_stop=0, phi_step=1.0, close_after=False,
    )

    # 读取 CSV
    csv_files = [f for f in os.listdir(hfss_asm_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"HFSS CSV 未找到: {hfss_asm_dir}")
    yf_linear = []
    with open(os.path.join(hfss_asm_dir, csv_files[0]), "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try: yf_linear.append(float(row[1]))
                except ValueError: continue
    yf_linear = np.array(yf_linear)
    yf_max = np.max(yf_linear)
    if yf_max <= 0:
        return np.full_like(yf_linear, -100.0)
    return 10.0 * np.log10(np.maximum(yf_linear, 1e-30) / yf_max)


def response_error(yc_db, yf_db, main_lobe_width=10.0):
    """粗细方向图误差: 主瓣区域加权 + 忽略粗模型 < -30 dB 的点。

    Args:
        yc_db: 粗模型归一化 dB 方向图
        yf_db: 细模型归一化 dB 方向图
        main_lobe_width: 主瓣区域半宽度 (度), 默认 ±10°
    """
    # 排除粗模型 < -30 dB 的点
    mask = yc_db > -30.0
    if not mask.any():
        return 0.0

    # 主瓣区域加权 (±main_lobe_width 度内权重更大)
    theta = pat.theta_deg
    main_lobe_mask = np.abs(theta) <= main_lobe_width
    weights = np.where(main_lobe_mask & mask, 5.0, 1.0)  # 主瓣区域 5 倍权重
    weights = weights[mask]

    diff = yc_db[mask] - yf_db[mask]
    return float(np.sum(weights * diff * diff) / np.sum(weights))


def _sigmoid(x):
    """数值稳定的 sigmoid。"""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _logit(x):
    """sigmoid 反函数, x ∈ (0,1) → ℝ。"""
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1.0 - x))


def _normalize(x, lb, ub):
    """实际值 → [0,1] 归一化。"""
    return (x - lb) / (ub - lb + 1e-30)


def _denormalize(x_norm, lb, ub):
    """[0,1] 归一化 → 实际值。"""
    return x_norm * (ub - lb) + lb


def parameter_extraction(yf_db, x0_unbounded=None):
    """PE: 无约束优化 + sigmoid 映射, 找粗模型变量匹配细模型响应。

    优化变量 y ∈ ℝ^n → sigmoid → [0,1] → 反归一化 → 实际值
    x0_unbounded: 无约束空间初始点 (默认随机)
    返回: 实际值 (与 lb/ub 对齐)
    """
    def pe_obj(y):
        x_norm = _sigmoid(y)                          # (0,1)
        x_actual = _denormalize(x_norm, lb, ub)       # 实际值
        return response_error(compute_coarse_pattern(x_actual), yf_db)
    pe_result = minimize(
        pe_obj, problem.n_vars,
        method=pe_cfg.get("method", "cma"),
        bounds=None,  # 无约束
        x0=x0_unbounded,
        sigma=pe_cfg.get("sigma", 1.0),
        pop_size=pe_cfg.get("pop_size") or None,
        max_iter=pe_cfg.get("max_iter", 500),
        n_jobs=pe_cfg.get("n_jobs", -1),
        verbose=pe_cfg.get("verbose", True),
        stop_fitness=pe_cfg.get("stopFitness", 0.001),
    )
    x_norm = _sigmoid(pe_result["x"])
    return _denormalize(x_norm, lb, ub)  # 返回实际值


# ASM 迭代目录
asm_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
hfss_asm_dir = str(out_dir / "hfss_asm" / asm_ts)
os.makedirs(hfss_asm_dir, exist_ok=True)

# 初始细模型评估 (Xc_star → HFSS)
print(f"\n  初始细模型评估...")
yf_db_init = hfss_eval(Xc_star)
fine_psll_init, _ = get_psll(yf_db_init, pat.theta_deg)
print(f"  初始细模型 PSLL: {fine_psll_init:.2f} dB")

# ASM 在归一化 [0,1] → 无约束空间 ℝ^n 中工作
# Xc_star [lb,ub] → 归一化 [0,1] → logit → 无约束
Xc_norm = _normalize(Xc_star, lb, ub)
Xc_unbounded = _logit(Xc_norm)

# PE 初始 (返回实际值, 以粗模型最优解为初始点)
print(f"  PE (参数提取)...")
Xe = parameter_extraction(yf_db_init, x0_unbounded=Xc_unbounded)

Xe_norm = _normalize(Xe, lb, ub)
Xe_unbounded = _logit(Xe_norm)
f = Xe_unbounded - Xc_unbounded  # 残差 (无约束空间)
print(f"  初始残差范数: {np.linalg.norm(f):.4f}")

# Broyden Jacobian 初始化
n_asm = problem.n_vars
B = np.eye(n_asm)
Xf = Xc_unbounded.copy()
asm_hist = [{"iter": 0, "PSLL": fine_psll_init, "res_norm": float("nan")}]

for k in range(1, asm_cfg["maxIterations"] + 1):
    print(f"\n  --- ASM iter {k} ---")
    # Broyden 步 (无约束空间)
    try:
        h = np.linalg.solve(B, -f)
    except np.linalg.LinAlgError:
        h = -np.linalg.lstsq(B, f, rcond=None)[0]
    Xf_new = Xf + h

    # 无约束 → sigmoid [0,1] → 反归一化实际值 → HFSS 评估
    Xf_actual = _denormalize(_sigmoid(Xf_new), lb, ub)
    yf_db = hfss_eval(Xf_actual)
    psll, _ = get_psll(yf_db, pat.theta_deg)

    # 计算当前粗模型方向图
    yc_db = compute_coarse_pattern(Xf_actual)
    coarse_psll_k, _ = get_psll(yc_db, pat.theta_deg)

    # 保存本代粗细对比图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, yc_db, "b-", linewidth=1.0, label="Coarse")
    ax.plot(theta, yf_db, "r--", linewidth=1.0, label="Fine (HFSS)")
    ax.set_title(f"ASM iter {k}  (Coarse: {coarse_psll_k:.2f} / Fine: {psll:.2f} dB)")
    ax.set_xlabel("$\\theta$ (deg)")
    ax.set_ylabel("Norm. Pattern (dB)")
    ax.set_ylim(-60, 3)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(fig_dir / f"asm_iter{k:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 保存本代方向图数据 CSV
    with open(csv_dir / f"asm_iter{k:02d}.csv", "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["theta_deg", "coarse_dB", "fine_dB"])
        for j in range(len(theta)):
            w.writerow([theta[j], yc_db[j], yf_db[j]])

    # PE: 以当前点为初始 → 返回实际值 → 归一化 [0,1] → logit → 无约束
    Xe_actual = parameter_extraction(yf_db, x0_unbounded=Xf_new)
    Xe_new = _logit(_normalize(Xe_actual, lb, ub))
    f_new = Xe_new - Xc_unbounded

    # Broyden 更新 (无约束空间)
    s = h; y = f_new - f
    ds = np.dot(s, s)
    if ds > 1e-16:
        B += np.outer(y - B @ s, s) / ds

    f = f_new; Xf = Xf_new
    res_norm = float(np.linalg.norm(f))
    asm_hist.append({"iter": k, "PSLL": psll, "res_norm": res_norm})
    print(f"  PSLL={psll:.2f} dB, res_norm={res_norm:.4f}")

    if psll <= asm_cfg["targetFinePSLL"] or res_norm < asm_cfg["targetResNorm"]:
        print(f"  ASM 收敛!")
        break

# ASM 结果
print(f"\n  ASM 最终细模型 PSLL: {asm_hist[-1]['PSLL']:.2f} dB")
print(f"  ASM 迭代次数: {len(asm_hist) - 1}")

# ASM 历史图
iters = [h["iter"] for h in asm_hist]
pss = [h["PSLL"] for h in asm_hist]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(iters, pss, "o-", linewidth=1.0)
ax.set_xlabel("ASM Iteration")
ax.set_ylabel("Fine PSLL (dB)")
ax.set_title("ASM Convergence")
ax.grid(True, alpha=0.3)
fig.savefig(fig_dir / "asm_convergence.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ASM 最终方向图对比
yf_db_final = hfss_eval(_denormalize(_sigmoid(Xf), lb, ub))
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, yc_star_db, "b-", linewidth=1.0, label="Coarse optimum")
ax.plot(theta, yf_db_final, "r--", linewidth=1.0, label="Fine (HFSS, ASM)")
ax.set_title(f"ASM Result  (Coarse: {coarse_psll:.2f} / Fine: {asm_hist[-1]['PSLL']:.2f} dB)")
ax.set_xlabel("$\\theta$ (deg)")
ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3)
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(fig_dir / "asm_final.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 保存最优粗模型方向图
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, yc_star_db, "b-", linewidth=1.0)
ax.set_title(f"Coarse Optimum  (PSLL = {coarse_psll:.2f} dB)")
ax.set_xlabel("$\\theta$ (deg)")
ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3)
ax.grid(True, alpha=0.3)
fig.savefig(fig_dir / "best_coarse.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 保存最优细模型方向图
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, yf_db_final, "r-", linewidth=1.0)
ax.set_title(f"Fine Optimum (HFSS, ASM)  (PSLL = {asm_hist[-1]['PSLL']:.2f} dB)")
ax.set_xlabel("$\\theta$ (deg)")
ax.set_ylabel("Norm. Pattern (dB)")
ax.set_ylim(-60, 3)
ax.grid(True, alpha=0.3)
fig.savefig(fig_dir / "best_fine.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 保存方向图数据 CSV (粗最优 + 细最优)
with open(csv_dir / "best_patterns.csv", "w", newline="", encoding="utf-8") as cf:
    w = csv.writer(cf)
    w.writerow(["theta_deg", "coarse_optimum_dB", "fine_optimum_dB"])
    for j in range(len(theta)):
        w.writerow([theta[j], yc_star_db[j], yf_db_final[j]])

# 保存 ASM 结果
asm_data = {
    "asm_settings": asm_cfg,
    "asm_history": asm_hist,
    "final_fine_psll": asm_hist[-1]["PSLL"],
    "final_Xf": _denormalize(_sigmoid(Xf), lb, ub).tolist(),
}
with open(out_dir / "asmResult.json", "w", encoding="utf-8") as f:
    f.write(to_json_flat(asm_data))
print(f"\n  ASM 结果已保存: {out_dir}")
