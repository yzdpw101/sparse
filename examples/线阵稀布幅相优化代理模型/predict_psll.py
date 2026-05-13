"""加载训练好的 Kriging 代理模型，预测 PSLL。

用法:
  # 预测单组位置
  python predict_psll.py --positions "0.1,0.2,0.3,..."

  # 预测 positions.txt 文件中的位置
  python predict_psll.py --file positions.txt

  # 同时计算真实 AF PSLL 做对比
  python predict_psll.py --file positions.txt --verify
"""
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pickle

from antopt import LMMapper, Pattern, SparseArrayProblem
from antopt.element_pattern import ElementPattern

HERE = Path(__file__).resolve().parent


def load_surrogate():
    """加载训练好的代理模型和配置。"""
    model_path = HERE / "surrogate_model.pkl"
    config_path = HERE / "surrogate_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先运行 train_surrogate.py")

    with open(model_path, "rb") as f:
        surr = pickle.load(f)

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    return surr, cfg


def build_problem(cfg):
    """根据配置构建 SparseArrayProblem (用于 --verify 对比)。"""
    freqs = np.sort(np.array(cfg["frequenciesGHz"], dtype=float))
    aa = cfg["aspectAngle"]
    _ta = aa["thetaDeg"]
    theta_start, theta_end, theta_step = _ta[0], _ta[1], _ta[2]
    theta0s = np.array(aa["theta0sDeg"], dtype=float)

    arr = cfg["antennaArray"]
    Ne, L_wl, dmin_wl = arr["Ne"], arr["L_wavelength"], arr["dmin_wavelength"]
    is_sym = arr.get("isSymmetryArray", False)
    is_fixed = arr.get("isFixedAperture", False)

    tgt = cfg["target"]
    mode = tgt["mode"]
    amp_bounds = tuple(tgt["amplitudeBounds"])

    # 单元方向图
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

    mapper = LMMapper(Ne=Ne, L=L_wl, dmin=dmin_wl, is_symmetric=is_sym,
                       is_fixed_aperture=is_fixed, use_sigmoid=False)
    pat = Pattern(theta_deg_start=theta_start, theta_deg_end=theta_end,
                  theta_deg_step=theta_step, theta0s_deg=theta0s, frequenciesGHz=freqs)
    problem = SparseArrayProblem(mapper, pat, mode=mode,
        amplitude_bounds=amp_bounds, element_patterns=fe_patterns,
        optimize_half_amp_phase=tgt.get("optimizeHalfAmpPhase", False))
    return problem


def main():
    parser = argparse.ArgumentParser(description="Kriging 代理模型预测 PSLL")
    parser.add_argument("--positions", type=str, help="逗号分隔的位置向量")
    parser.add_argument("--file", type=str, help="位置文件路径 (每行一组位置)")
    parser.add_argument("--verify", action="store_true", help="同时计算真实 AF PSLL 做对比")
    args = parser.parse_args()

    surr, cfg = load_surrogate()
    n_vars = cfg["n_vars"]

    # 解析输入
    if args.positions:
        x = np.array([float(v) for v in args.positions.split(",")])
        if len(x) != n_vars:
            print(f"错误: 位置维度 {len(x)} != 预期 {n_vars}")
            sys.exit(1)
        X_test = x.reshape(1, -1)
    elif args.file:
        X_test = np.loadtxt(args.file, delimiter=",")
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        if X_test.shape[1] != n_vars:
            print(f"错误: 位置维度 {X_test.shape[1]} != 预期 {n_vars}")
            sys.exit(1)
    else:
        print("请指定 --positions 或 --file")
        sys.exit(1)

    # Kriging 预测
    mu, sigma = surr.predict(X_test, return_std=True)

    print(f"=== Kriging 代理预测 ===")
    print(f"  配置: {cfg['antennaArray']['Ne']}元, "
          f"L={cfg['antennaArray']['L_wavelength']}λ, "
          f"{cfg['frequenciesGHz']} GHz")
    print(f"  输入维度: {n_vars}, 训练样本: {cfg['n_samples']}")
    print()

    for i in range(len(X_test)):
        print(f"  [{i+1}] PSLL = {mu[i]:.4f} dB (σ = {sigma[i]:.4f})")

    # 真实 AF 验证
    if args.verify:
        problem = build_problem(cfg)
        print(f"\n  --- AF 真实值对比 ---")
        for i in range(len(X_test)):
            true_psll = problem.fitness(X_test[i])
            err = abs(mu[i] - true_psll)
            print(f"  [{i+1}] Kriging={mu[i]:.4f}, AF={true_psll:.4f}, "
                  f"误差={err:.4f} dB")


if __name__ == "__main__":
    main()
