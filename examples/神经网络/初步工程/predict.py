"""加载训练好的神经网络代理模型，预测 PSLL。

用法:
  python predict.py --positions "0.1,0.2,0.3,..."
  python predict.py --file positions.txt --verify
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import torch

from model import PSLLNet

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pt"


def load_model(n_vars=15):
    """加载训练好的模型。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}\n请先运行 train.py")
    model = PSLLNet(n_vars=n_vars)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="神经网络代理预测 PSLL")
    parser.add_argument("--positions", type=str, help="逗号分隔的位置向量")
    parser.add_argument("--file", type=str, help="位置文件路径")
    parser.add_argument("--verify", action="store_true", help="同时计算真实 AF PSLL 做对比")
    parser.add_argument("--n_vars", type=int, default=15, help="输入维度")
    args = parser.parse_args()

    model = load_model(args.n_vars)

    # 解析输入
    if args.positions:
        x = np.array([float(v) for v in args.positions.split(",")])
        X_test = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
    elif args.file:
        data = np.loadtxt(args.file, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        X_test = torch.tensor(data, dtype=torch.float32)
    else:
        print("请指定 --positions 或 --file")
        sys.exit(1)

    # 预测
    with torch.no_grad():
        mu = model(X_test).numpy()

    print(f"=== 神经网络代理预测 ===")
    print(f"  模型: {MODEL_PATH}")
    print(f"  输入维度: {args.n_vars}")

    for i in range(len(mu)):
        print(f"  [{i+1}] PSLL = {mu[i]:.4f} dB")

    # 真实 AF 验证
    if args.verify:
        import warnings; warnings.filterwarnings("ignore")
        from antopt import LMMapper, Pattern, SparseArrayProblem
        from antopt.element_pattern import ElementPattern
        import json

        cfg_path = HERE.parent.parent / "线阵稀布幅相优化代理模型" / "surrogate_config.json"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        freqs = np.sort(np.array(cfg["frequenciesGHz"], dtype=float))
        aa = cfg["aspectAngle"]; _ta = aa["thetaDeg"]
        arr = cfg["antennaArray"]; eg = cfg.get("ePattern", {})

        fe_patterns = None
        if eg.get("enabled") and eg.get("csvDirectory"):
            epath = HERE.parent.parent / "线阵稀布幅相优化代理模型" / eg["csvDirectory"]
            if epath.exists():
                _ep = eg.get("thetaDeg", [-180, 180, 0.01])
                fe_patterns = ElementPattern.from_hfss_multi_freq(
                    str(epath), freqs,
                    np.arange(_ta[0], _ta[1] + _ta[2]/2, _ta[2]),
                    _ep[2], is_gain=eg.get("isGain", True), in_dB=eg.get("inDB", False),
                    input_theta_range=(_ep[0], _ep[1]))

        mapper = LMMapper(Ne=arr["Ne"], L=arr["L_wavelength"], dmin=arr["dmin_wavelength"],
                           is_symmetric=arr.get("isSymmetryArray", False),
                           is_fixed_aperture=arr.get("isFixedAperture", False), use_sigmoid=False)
        pat = Pattern(theta_deg_start=_ta[0], theta_deg_end=_ta[1], theta_deg_step=_ta[2],
                      theta0s_deg=np.array(aa["theta0sDeg"]), frequenciesGHz=freqs)
        tgt = cfg["target"]
        problem = SparseArrayProblem(mapper, pat, mode=tgt["mode"],
            amplitude_bounds=tuple(tgt["amplitudeBounds"]),
            element_patterns=fe_patterns,
            optimize_half_amp_phase=tgt.get("optimizeHalfAmpPhase", False))

        X_np = X_test.numpy()
        print(f"\n  --- AF 真实值对比 ---")
        for i in range(len(mu)):
            true_psll = problem.fitness(X_np[i])
            err = abs(mu[i] - true_psll)
            print(f"  [{i+1}] NN={mu[i]:.4f}, AF={true_psll:.4f}, 误差={err:.4f} dB")


if __name__ == "__main__":
    main()
