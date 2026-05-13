"""训练 Kriging 代理模型并保存 — 用于快速预测 PSLL。

用法: cd examples/线阵稀布幅相优化代理模型 && python train_surrogate.py

训练完成后会保存:
  - surrogate_model.pkl  (Kriging 模型)
  - surrogate_config.json (训练配置, 用于加载时校验)
  - training_data.npz     (训练数据, 可选)
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pickle

from antopt import LMMapper, Pattern, SparseArrayProblem
from antopt.surrogate import KrigingSurrogate, lhs_sample
from antopt.element_pattern import ElementPattern

HERE = Path(__file__).resolve().parent


# ── 1. 加载配置 ──
with open(HERE / "Config.json", encoding="utf-8") as f:
    cfg = json.load(f)

freqs = np.sort(np.array(cfg["frequenciesGHz"], dtype=float))
asp = cfg["aspectAngle"]
_ta = asp["thetaDeg"]
theta_start, theta_end, theta_step = _ta[0], _ta[1], _ta[2]
theta0s = np.array(asp["theta0sDeg"], dtype=float)

optz = cfg["target"]
mode = optz["mode"]
amp_bounds = tuple(optz["amplitudeBounds"])
hpbw_target = optz.get("targetHPBW") or 180.0
use_pt_penalty = optz.get("mainLobePointingPenalty", True)
optimize_half = optz.get("optimizeHalfAmpPhase", False)

arr = cfg["antennaArray"]
Ne, L_wl, dmin_wl = arr["Ne"], arr["L_wavelength"], arr["dmin_wavelength"]
is_sym = arr.get("isSymmetryArray", False)
is_fixed = arr.get("isFixedAperture", False)

# ── 2. 单元方向图 ──
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

# ── 3. 映射器 + 方向图 + 适应度 ──
mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl, is_symmetric=is_sym,
                   is_fixed_aperture=is_fixed, use_sigmoid=False)
pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
              theta_deg_step=theta_step, theta0s_deg=theta0s, frequenciesGHz=freqs)

problem = SparseArrayProblem(mapper, pat, mode=mode,
    amplitude_bounds=amp_bounds, target_hpbw=hpbw_target,
    element_patterns=fe_patterns, use_pointing_penalty=use_pt_penalty,
    optimize_half_amp_phase=optimize_half)

n_vars = problem.n_vars

# ── 4. 采样 + 计算 PSLL ──
n_samples = cfg.get("training_samples", 500)  # 默认 500 个样本
seed = cfg.get("randomSeed", 42)
if seed is None:
    seed = int(np.random.randint(0, 2**31))

print(f"=== 训练 Kriging 代理模型 ===")
print(f"  阵列: {Ne}元, L={L_wl}λ, dmin={dmin_wl}λ, 对称={is_sym}")
print(f"  频率: {freqs.tolist()} GHz, 指向角: {theta0s.tolist()}")
print(f"  n_vars={n_vars}, 训练样本数={n_samples}")

lb = np.zeros(n_vars)
ub = np.ones(n_vars)

t0 = time.perf_counter()
X = lhs_sample(n_vars, n_samples, (lb, ub), seed=seed)
Y = np.array([problem.fitness(x) for x in X])
t_sample = time.perf_counter() - t0

print(f"  采样完成: {t_sample:.1f}s, PSLL 范围 [{Y.min():.4f}, {Y.max():.4f}] dB")

# ── 5. 训练 Kriging ──
t1 = time.perf_counter()
surr = KrigingSurrogate(n_vars)
surr.fit(X, Y)
t_train = time.perf_counter() - t1

# 交叉验证: 留一法估计预测精度
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
kernel = Matern(nu=2.5, length_scale=np.ones(n_vars), length_scale_bounds=(1e-5, 1e5))
gpr_cv = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3)
cv_scores = cross_val_score(gpr_cv, X, Y, cv=min(5, n_samples // 10), scoring='r2')

print(f"  训练完成: {t_train:.2f}s")
print(f"  交叉验证 R2 = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ── 6. 保存 ──
out_data = {
    "frequenciesGHz": freqs.tolist(),
    "aspectAngle": {"thetaDeg": [theta_start, theta_end, theta_step], "theta0sDeg": theta0s.tolist()},
    "antennaArray": {"Ne": Ne, "L_wavelength": L_wl, "dmin_wavelength": dmin_wl,
                     "isSymmetryArray": is_sym, "isFixedAperture": is_fixed},
    "ePattern": {"enabled": eg.get("enabled", False), "csvDirectory": eg.get("csvDirectory"),
                 "thetaDeg": eg.get("thetaDeg"), "isGain": eg.get("isGain"), "inDB": eg.get("inDB")},
    "target": {"mode": mode, "amplitudeBounds": list(amp_bounds)},
    "n_vars": n_vars, "n_samples": n_samples,
}

# 保存模型
model_path = HERE / "surrogate_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(surr, f)

# 保存配置
config_path = HERE / "surrogate_config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

# 保存训练数据
data_path = HERE / "training_data.npz"
np.savez(data_path, X=X, Y=Y)

print(f"\n  已保存:")
print(f"    模型: {model_path}")
print(f"    配置: {config_path}")
print(f"    数据: {data_path}")
