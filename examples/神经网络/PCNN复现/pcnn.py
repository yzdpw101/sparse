"""物理信息优化线阵方向图 — PyTorch 自动微分 + L-BFGS-B。

论文: "A Physics-Informed Convolutional Neural Network Approach for
       Radiation Pattern Synthesis of Linear Antenna Arrays"
作者: 钟志兴, 顾鹏飞 等 (南京理工大学)

核心: AF 公式作为可微计算图 → autograd 精确梯度 → L-BFGS-B 优化。

运行: cd examples/神经网络/PCNN复现 && python pcnn.py
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize as scipy_minimize

# ── 配置 ──
N_EL = 32
D_WL = 0.5
THETA_MAIN_DEG = 30.0
N_THETA = 10001      # 密网格, 确保副瓣峰值不被遗漏
LSE_TAU = 100.0      # log-sum-exp 温度 (越高越接近 max, 但梯度越陡)

EP_CSV = Path(__file__).resolve().parent.parent.parent / \
    "线阵稀布幅相优化" / "input" / "element_pattern" / \
    "patch_GainTotal" / "2.45GHz_0.01.csv"


def load_element_pattern(csv_path, theta_deg):
    """加载单元方向图 CSV → 场方向图 (线性)。"""
    thetas, gains = [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                try:
                    thetas.append(float(row[0]))
                    gains.append(float(row[1]))
                except ValueError:
                    continue
    return np.sqrt(np.maximum(np.interp(theta_deg, thetas, gains), 0.0))


def psll_lse(A, phi, y_n, sin_theta, fe, outside_mask, tau=LSE_TAU, phi_prog=None):
    """PyTorch: 计算 LSE 平滑副瓣电平 (dB)。

    AF(θ) = Σ A_n · exp(j·φ_n) · exp(j·2π·y_n·sinθ)
    total(θ) = AF(θ) · Fe(θ)
    P(θ) = |total(θ)|²
    LSE = (1/τ)·log(Σ_{outside} exp(τ·P(θ)/P_max))

    phi_prog: 渐进相位 (固定), 若提供则 phi 为残余相位
    Returns: psll_db (scalar tensor)
    """
    # 空间相位: (N, Nθ)
    phase = 2 * np.pi * torch.outer(y_n, sin_theta)
    # 复数激励 × 空间 steering
    if phi_prog is not None:
        # 残余相位模式: total_phase = phi_prog + delta_phi
        total_phase = phi_prog + phi
    else:
        total_phase = phi
    exc = A.unsqueeze(1) * torch.exp(1j * total_phase.unsqueeze(1))  # (N, 1)
    steering = torch.exp(1j * phase)  # (N, Nθ)
    # AF
    af = torch.sum(exc * steering, dim=0)  # (Nθ,)
    # 总方向图
    total = af * fe
    power = torch.abs(total) ** 2
    p_max = torch.max(power)
    # LSE 副瓣
    p_sl = power[outside_mask] / (p_max + 1e-30)
    # softmax (数值稳定)
    log_w = tau * p_sl
    log_w = log_w - torch.max(log_w)
    w = torch.exp(log_w)
    w = w / torch.sum(w)
    lse = torch.sum(w * p_sl)
    psll_db = 10 * torch.log10(lse + 1e-30)
    return psll_db


def psll_exact(A, phi, y_n, sin_theta, fe, outside_mask, phi_prog=None):
    """计算精确 PSLL (真实 max, 非 LSE 近似)。"""
    phase = 2 * np.pi * torch.outer(y_n, sin_theta)
    if phi_prog is not None:
        total_phase = phi_prog + phi
    else:
        total_phase = phi
    exc = A.unsqueeze(1) * torch.exp(1j * total_phase.unsqueeze(1))
    af = torch.sum(exc * torch.exp(1j * phase), dim=0)
    power = torch.abs(af * fe) ** 2
    p_max = torch.max(power)
    pat_dB = 10 * torch.log10(power / (p_max + 1e-30) + 1e-30)
    return torch.max(pat_dB[outside_mask]).item()


def make_scipy_obj(N_el, y_n_np, sin_theta_np, fe_np, outside_mask_np):
    """创建 scipy 可调用的目标函数 + 解析梯度 (通过 autograd)。"""
    y_n = torch.tensor(y_n_np, dtype=torch.float64)
    sin_theta = torch.tensor(sin_theta_np, dtype=torch.float64)
    fe = torch.tensor(fe_np, dtype=torch.float64)
    outside_mask = torch.tensor(outside_mask_np, dtype=torch.bool)

    def objective_and_grad(x_np):
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        A = x[:N_el]
        phi = x[N_el:]

        psll = psll_lse(A, phi, y_n, sin_theta, fe, outside_mask)
        # autograd 精确梯度
        psll.backward()

        grad = x.grad.clone().detach().numpy()
        val = psll.item()
        # 清除梯度
        x.grad = None

        return val, grad  # psll 已是负 dB, minimize → 越小越好

    return objective_and_grad


def make_scipy_obj_residual(N_el, y_n_np, sin_theta_np, fe_np, outside_mask_np, phi_prog_np):
    """残余相位模式: 只优化 delta_phi ∈ [-π, π], 渐进相位固定。"""
    y_n = torch.tensor(y_n_np, dtype=torch.float64)
    sin_theta = torch.tensor(sin_theta_np, dtype=torch.float64)
    fe = torch.tensor(fe_np, dtype=torch.float64)
    outside_mask = torch.tensor(outside_mask_np, dtype=torch.bool)
    phi_prog = torch.tensor(phi_prog_np, dtype=torch.float64)

    def objective_and_grad(x_np):
        x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        A = x[:N_el]
        delta_phi = x[N_el:]

        psll = psll_lse(A, delta_phi, y_n, sin_theta, fe, outside_mask,
                        phi_prog=phi_prog)
        psll.backward()

        grad = x.grad.clone().detach().numpy()
        val = psll.item()
        x.grad = None

        return val, grad

    return objective_and_grad


def optimize_lbfgsb(N_el=N_EL, theta_main_deg=THETA_MAIN_DEG, fe_pattern=None):
    """PyTorch autograd + L-BFGS-B, 残余相位模式。"""
    theta_main = np.radians(theta_main_deg)
    d_wl = D_WL
    y_n = np.arange(N_el) * d_wl - (N_el - 1) * d_wl / 2
    theta = np.linspace(-np.pi / 2, np.pi / 2, N_THETA)
    sin_theta = np.sin(theta)
    outside_mask = np.abs(theta - theta_main) > np.radians(5)

    if fe_pattern is None:
        fe_pattern = np.ones(N_THETA)

    # 渐进相位 (固定, 用于波束指向)
    phi_prog = -2 * np.pi * y_n * np.sin(theta_main)
    # 初始化: 均匀幅度 + 零残余相位
    x0 = np.concatenate([np.ones(N_el) * 0.5, np.zeros(N_el)])

    # 残余相位有界: A ∈ [0,1], delta_phi ∈ [-π, π]
    lb = np.concatenate([np.zeros(N_el), -np.pi * np.ones(N_el)])
    ub = np.concatenate([np.ones(N_el), np.pi * np.ones(N_el)])

    obj = make_scipy_obj_residual(N_el, y_n, sin_theta, fe_pattern, outside_mask, phi_prog)

    print(f"\n  残余相位 autograd L-BFGS-B: N={N_el}, θ0={theta_main_deg}°")
    t0 = time.perf_counter()
    res = scipy_minimize(
        obj, x0, method="L-BFGS-B", jac=True, bounds=list(zip(lb, ub)),
        options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-10}
    )
    elapsed = time.perf_counter() - t0

    # 计算最终 PSLL
    A = torch.tensor(res.x[:N_el], dtype=torch.float64)
    delta_phi = torch.tensor(res.x[N_el:], dtype=torch.float64)
    y_n_t = torch.tensor(y_n, dtype=torch.float64)
    sin_t = torch.tensor(sin_theta, dtype=torch.float64)
    fe_t = torch.tensor(fe_pattern, dtype=torch.float64)
    mask_t = torch.tensor(outside_mask, dtype=torch.bool)
    phi_prog_t = torch.tensor(phi_prog, dtype=torch.float64)
    psll = psll_lse(A, delta_phi, y_n_t, sin_t, fe_t, mask_t,
                    phi_prog=phi_prog_t).item()

    # 验证指向
    total_phase = phi_prog + res.x[N_el:]
    phase_shift = np.abs(total_phase - phi_prog)
    psll_real = psll_exact(A, delta_phi, y_n_t, sin_t, fe_t, mask_t,
                           phi_prog=phi_prog_t)
    print(f"  LSE PSLL = {psll:.2f} dB, 精确 PSLL = {psll_real:.2f} dB, "
          f"{elapsed:.1f}s, nfev={res.nfev}, niter={res.nit}")
    print(f"  最大残余相位: {np.max(np.abs(res.x[N_el:])):.2f} rad "
          f"({np.degrees(np.max(np.abs(res.x[N_el:]))):.1f}°)")
    return theta, None, psll_real, elapsed


def optimize_fd(N_el=N_EL, theta_main_deg=THETA_MAIN_DEG, fe_pattern=None):
    """有限差分 L-BFGS-B (对比), 残余相位模式。"""
    theta_main = np.radians(theta_main_deg)
    d_wl = D_WL
    y_n = np.arange(N_el) * d_wl - (N_el - 1) * d_wl / 2
    theta = np.linspace(-np.pi / 2, np.pi / 2, N_THETA)
    sin_theta = np.sin(theta)
    outside_mask = np.abs(theta - theta_main) > np.radians(5)

    if fe_pattern is None:
        fe_pattern = np.ones(N_THETA)

    phi_prog = -2 * np.pi * y_n * np.sin(theta_main)

    def objective(x_np):
        x = torch.tensor(x_np, dtype=torch.float64)
        A = x[:N_el]; delta_phi = x[N_el:]
        psll = psll_lse(A, delta_phi,
                        torch.tensor(y_n, dtype=torch.float64),
                        torch.tensor(sin_theta, dtype=torch.float64),
                        torch.tensor(fe_pattern, dtype=torch.float64),
                        torch.tensor(outside_mask, dtype=torch.bool),
                        phi_prog=torch.tensor(phi_prog, dtype=torch.float64))
        return psll.item()

    x0 = np.concatenate([np.ones(N_el) * 0.5, np.zeros(N_el)])
    lb = np.concatenate([np.zeros(N_el), -np.pi * np.ones(N_el)])
    ub = np.concatenate([np.ones(N_el), np.pi * np.ones(N_el)])

    print(f"\n  有限差分 L-BFGS-B: N={N_el}, θ0={theta_main_deg}°")
    t0 = time.perf_counter()
    res = scipy_minimize(
        objective, x0, method="L-BFGS-B", bounds=list(zip(lb, ub)),
        options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-10}
    )
    elapsed = time.perf_counter() - t0

    x = torch.tensor(res.x, dtype=torch.float64)
    A = x[:N_el]; delta_phi = x[N_el:]
    y_n_t = torch.tensor(y_n, dtype=torch.float64)
    sin_t = torch.tensor(sin_theta, dtype=torch.float64)
    fe_t = torch.tensor(fe_pattern, dtype=torch.float64)
    mask_t = torch.tensor(outside_mask, dtype=torch.bool)
    phi_prog_t = torch.tensor(phi_prog, dtype=torch.float64)
    psll = psll_lse(A, delta_phi, y_n_t, sin_t, fe_t, mask_t,
                    phi_prog=phi_prog_t).item()
    psll_real = psll_exact(A, delta_phi, y_n_t, sin_t, fe_t, mask_t,
                           phi_prog=phi_prog_t)

    print(f"  LSE PSLL = {psll:.2f} dB, 精确 PSLL = {psll_real:.2f} dB, "
          f"{elapsed:.1f}s, nfev={res.nfev}, niter={res.nit}")
    return theta, None, psll_real, elapsed


def optimize_de(N_el=N_EL, theta_main_deg=THETA_MAIN_DEG, fe_pattern=None):
    """差分进化 (对比), 残余相位模式。"""
    from scipy.optimize import differential_evolution

    theta_main = np.radians(theta_main_deg)
    d_wl = D_WL
    y_n = np.arange(N_el) * d_wl - (N_el - 1) * d_wl / 2
    theta = np.linspace(-np.pi / 2, np.pi / 2, N_THETA)
    sin_theta = np.sin(theta)
    outside_mask = np.abs(theta - theta_main) > np.radians(5)

    if fe_pattern is None:
        fe_pattern = np.ones(N_THETA)

    phi_prog = -2 * np.pi * y_n * np.sin(theta_main)

    def objective(x_np):
        x = torch.tensor(x_np, dtype=torch.float64)
        A = x[:N_el]; delta_phi = x[N_el:]
        psll = psll_lse(A, delta_phi,
                        torch.tensor(y_n, dtype=torch.float64),
                        torch.tensor(sin_theta, dtype=torch.float64),
                        torch.tensor(fe_pattern, dtype=torch.float64),
                        torch.tensor(outside_mask, dtype=torch.bool),
                        phi_prog=torch.tensor(phi_prog, dtype=torch.float64))
        return psll.item()

    bounds = [(0.01, 1.0)] * N_el + [(-np.pi, np.pi)] * N_el

    print(f"\n  差分进化: N={N_el}, θ0={theta_main_deg}°")
    t0 = time.perf_counter()
    res = differential_evolution(
        objective, bounds, maxiter=300, seed=42, tol=1e-8, popsize=20
    )
    elapsed = time.perf_counter() - t0

    x = torch.tensor(res.x, dtype=torch.float64)
    A = x[:N_el]; delta_phi = x[N_el:]
    y_n_t = torch.tensor(y_n, dtype=torch.float64)
    sin_t = torch.tensor(sin_theta, dtype=torch.float64)
    fe_t = torch.tensor(fe_pattern, dtype=torch.float64)
    mask_t = torch.tensor(outside_mask, dtype=torch.bool)
    phi_prog_t = torch.tensor(phi_prog, dtype=torch.float64)
    psll = psll_lse(A, delta_phi, y_n_t, sin_t, fe_t, mask_t,
                    phi_prog=phi_prog_t).item()
    psll_real = psll_exact(A, delta_phi, y_n_t, sin_t, fe_t, mask_t,
                           phi_prog=phi_prog_t)

    print(f"  LSE PSLL = {psll:.2f} dB, 精确 PSLL = {psll_real:.2f} dB, "
          f"{elapsed:.1f}s, nfev={res.nfev}")
    return theta, None, psll_real, elapsed


if __name__ == "__main__":
    theta_grid_deg = np.linspace(-90, 90, N_THETA)
    if EP_CSV.exists():
        fe = load_element_pattern(EP_CSV, theta_grid_deg)
        print(f"已加载单元方向图: {EP_CSV.name}")
    else:
        print(f"警告: 单元方向图不存在，使用各向同性")
        fe = np.ones(N_THETA)

    print("=" * 60)
    print("PCNN 复现: PyTorch autograd + L-BFGS-B vs 有限差分 vs DE")
    print(f"  N={N_EL}, θ0={THETA_MAIN_DEG}°, 含单元方向图")
    print("=" * 60)

    t1, _, psll1, e1 = optimize_lbfgsb(fe_pattern=fe)
    t2, _, psll2, e2 = optimize_fd(fe_pattern=fe)
    t3, _, psll3, e3 = optimize_de(fe_pattern=fe)

    print(f"\n{'='*60}")
    print(f"  对比 N={N_EL}, θ0={THETA_MAIN_DEG}°:")
    print(f"    autograd:    PSLL = {psll1:.2f} dB, {e1:.1f}s")
    print(f"    有限差分:    PSLL = {psll2:.2f} dB, {e2:.1f}s")
    print(f"    差分进化:    PSLL = {psll3:.2f} dB, {e3:.1f}s")
    if e1 > 0:
        print(f"    autograd vs FD: {e2/e1:.1f}x 加速")
    print(f"{'='*60}")
