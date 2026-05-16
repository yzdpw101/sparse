"""NNP (Neural Network Parameterization) 演示 — 可微分函数优化。

核心思想:
  用神经网络参数化解: x = NN(θ)
  直接最小化 f(NN(θ)), 反向传播更新 θ
  无需预训练, 每个任务从随机权重开始在线优化

与传统方法的区别:
  - 传统: 直接优化 x (梯度下降在 x 空间)
  - NNP: 优化 θ (梯度下降在 θ 空间), NN 结构提供更好的参数化

用法: python nnp_demo.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


# ── 可微分测试函数 (PyTorch) ──

def rastrigin(x):
    """Rastrigin: 大量局部极小, 全局最小在原点 f=0。"""
    A = 10.0
    return A * x.shape[1] + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x), dim=1)


def ackley(x):
    """Ackley: 中心深坑, 周围大量局部极小。"""
    n = x.shape[1]
    s1 = torch.sum(x**2, dim=1)
    s2 = torch.sum(torch.cos(2 * torch.pi * x), dim=1)
    return -20 * torch.exp(-0.2 * torch.sqrt(s1 / n + 1e-12)) - torch.exp(s2 / n) + 20 + torch.e


def rosenbrock(x):
    """Rosenbrock: 狭长弯曲山谷, 全局最小在 (1,1,...,1)。"""
    return torch.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, dim=1)


def griewank(x):
    """Griewank: 大量规则分布的局部极小。"""
    n = x.shape[1]
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return torch.sum(x**2, dim=1) / 4000 - torch.prod(torch.cos(x / torch.sqrt(idx)), dim=1) + 1


FUNCTIONS = {
    "rastrigin":  (rastrigin,  (-5.12, 5.12)),
    "ackley":     (ackley,     (-5.0, 5.0)),
    "rosenbrock": (rosenbrock, (-2.0, 2.0)),
    "griewank":   (griewank,   (-5.0, 5.0)),
}


# ── NNP 模型 ──

class NNPModel(nn.Module):
    """NNP 核心: 用神经网络参数化解。

    输入: 固定的任务嵌入 (全1, 无额外信息)
    输出: 解向量 x ∈ ℝ^dim, 通过 sigmoid 映射到 (0,1)

    优化过程中, 更新的是网络参数 θ, 而非 x 本身。
    这等价于: min_θ f(NNPModel(θ))
    """

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),  # 输出 (0,1), 天然有界
        )
        # 初始化输出层, 让初始 x 在 0.5 附近 (搜索范围中心)
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self):
        # 固定输入 (全1), 网络参数 θ 决定输出 x ∈ (0,1)
        dummy = torch.ones(1, 1, device=next(self.parameters()).device)
        return self.net(dummy).squeeze(0)  # (dim,), 值域 (0,1)


# ── 优化方法 ──

def optimize_nnp(func, dim, sr, n_iters=500, lr=0.01):
    """NNP 方法: 直接参数化解, 每个任务独立在线优化。

    流程:
      1. 随机初始化网络参数 θ
      2. 前向: x = NNPModel(θ)
      3. 损失: loss = f(x)
      4. 反向: ∇θ loss → 更新 θ
      5. 重复直到收敛
    """
    model = NNPModel(dim, hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    history = []
    best_x, best_f = None, float("inf")

    for t in range(n_iters):
        x01 = model()  # (dim,), sigmoid 输出 (0,1)
        # 线性映射到搜索范围
        x = x01 * (sr[1] - sr[0]) + sr[0]
        fval = func(x.unsqueeze(0)).squeeze()

        opt.zero_grad()
        fval.backward()
        opt.step()
        scheduler.step()

        fv = fval.item()
        history.append(fv)
        if fv < best_f:
            best_f = fv
            best_x = x.detach().clone()

    return best_x.cpu().numpy(), history


def optimize_gd(func, dim, sr, n_iters=500, lr=0.01):
    """基线: 梯度下降 + 动量。"""
    x = (torch.randn(1, dim) * (sr[1] - sr[0]) / 2).clamp(sr[0], sr[1])
    x = x.detach().requires_grad_(True)
    v = torch.zeros_like(x)

    history = []
    for t in range(n_iters):
        fval = func(x)
        fval.sum().backward()
        g = x.grad.detach()
        with torch.no_grad():
            v = 0.9 * v + lr * g
            x.data = (x.data - v).clamp(sr[0], sr[1])
        x.grad = None
        history.append(fval.item())
    return x.detach().numpy().flatten(), history


def optimize_adam(func, dim, sr, n_iters=500, lr=0.01):
    """基线: Adam 直接优化 x。"""
    x = (torch.randn(1, dim) * (sr[1] - sr[0]) / 2).clamp(sr[0], sr[1])
    x = x.detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    history = []
    for t in range(n_iters):
        fval = func(x)
        opt.zero_grad()
        fval.sum().backward()
        opt.step()
        with torch.no_grad():
            x.data = x.data.clamp(sr[0], sr[1])
        history.append(fval.item())
    return x.detach().numpy().flatten(), history


def optimize_random(func, dim, sr, n_iters=500):
    """基线: 随机搜索。"""
    best_x = torch.randn(1, dim) * (sr[1] - sr[0]) / 2
    best_x = best_x.clamp(sr[0], sr[1])
    best_f = func(best_x).item()
    history = [best_f]
    for _ in range(n_iters - 1):
        x = torch.randn(1, dim) * (sr[1] - sr[0]) / 2
        x = x.clamp(sr[0], sr[1])
        f = func(x).item()
        if f < best_f:
            best_x, best_f = x, f
        history.append(best_f)
    return best_x.numpy().flatten(), history


# ── 主流程 ──

def main():
    dim = 10
    n_iters = 500
    sr = (-5.0, 5.0)

    print("=== NNP (Neural Network Parameterization) 演示 ===")
    print(f"  维度: {dim}, 最大迭代: {n_iters}")
    print()

    results = {}
    for name, (func, search_range) in FUNCTIONS.items():
        print(f"--- {name} ({dim}D) ---")
        t0 = time.perf_counter()

        methods = {}
        for method_name, method_fn in [
            ("Random", lambda: optimize_random(func, dim, sr, n_iters)),
            ("GD",     lambda: optimize_gd(func, dim, sr, n_iters)),
            ("Adam",   lambda: optimize_adam(func, dim, sr, n_iters)),
            ("NNP",    lambda: optimize_nnp(func, dim, sr, n_iters)),
        ]:
            np.random.seed(42)
            torch.manual_seed(42)
            x, h = method_fn()
            methods[method_name] = h
            print(f"  {method_name:>8}: final = {h[-1]:.6f}  (x* = {np.mean(np.abs(x)):.4f})")

        elapsed = time.perf_counter() - t0
        print(f"  耗时: {elapsed:.1f}s\n")
        results[name] = methods

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = {
        "Random": "#95a5a6", "GD": "#3498db", "Adam": "#2ecc71",
        "NNP": "#e74c3c",
    }
    labels = {
        "Random": "Random", "GD": "GD+Momentum", "Adam": "Adam",
        "NNP": "NNP",
    }

    for i, (name, methods) in enumerate(results.items()):
        ax = axes[i]
        for method_name, h in methods.items():
            h_arr = np.maximum(np.array(h), 1e-10)
            lw = 2.5 if method_name == "NNP" else 1.0
            ls = "-" if method_name == "NNP" else "--"
            ax.semilogy(h_arr, color=colors[method_name], lw=lw,
                       label=labels[method_name], ls=ls)
        ax.set_title(f"{name.capitalize()} ({dim}D)", fontsize=13)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("f(x) (log)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("NNP vs Traditional Optimization Methods", fontsize=14)
    plt.tight_layout()
    out = "nnp_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"图片已保存: {out}")
    plt.close()


if __name__ == "__main__":
    main()
