"""对比实验: 神经网络优化器 vs 传统优化器。

在多个多模态函数上比较:
  1. 神经网络优化器 (GRU-based, 端到端可微分训练)
  2. 梯度下降 + 动量
  3. Adam
  4. 随机搜索

用法: python compare.py
"""

import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from nn_optimizer import (
    NeuralOptimizer, train_optimizer,
    pt_rastrigin, pt_ackley, pt_rosenbrock, PT_BENCHMARKS,
)


# ── 传统优化器 (numpy) ──

def grad_descent(func_np, x0, lr=0.01, n_iters=200, eps=1e-4):
    x = x0.copy(); v = np.zeros_like(x); history = []
    for t in range(n_iters):
        g = np.zeros_like(x)
        for d in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[d] += eps; xm[d] -= eps
            g[d] = (func_np(xp) - func_np(xm)) / (2 * eps)
        v = 0.9 * v + lr * g
        x = x - v
        history.append(func_np(x))
    return x, history


def adam_optim(func_np, x0, lr=0.01, n_iters=200, eps=1e-4):
    x = x0.copy(); m = np.zeros_like(x); v = np.zeros_like(x); history = []
    b1, b2 = 0.9, 0.999
    for t in range(1, n_iters + 1):
        g = np.zeros_like(x)
        for d in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[d] += eps; xm[d] -= eps
            g[d] = (func_np(xp) - func_np(xm)) / (2 * eps)
        m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g**2
        x = x - lr * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + 1e-8)
        history.append(func_np(x))
    return x, history


def random_search(func_np, dim, n_iters=200, sr=(-5, 5)):
    best_x = np.random.uniform(sr[0], sr[1], dim)
    best_f = func_np(best_x); history = [best_f]
    for _ in range(n_iters - 1):
        x = np.random.uniform(sr[0], sr[1], dim)
        f = func_np(x)
        if f < best_f: best_x, best_f = x, f
        history.append(best_f)
    return best_x, history


def nn_optimize(model, pt_func, dim, n_iters=200, sr=(-5, 5), device="cpu"):
    """用训练好的 NN 优化器求解 (推理模式)。"""
    model.eval()
    x = torch.randn(1, dim, device=device) * (sr[1] - sr[0]) / 2
    x = x.clamp(sr[0], sr[1])
    hidden = None; history = []
    for t in range(n_iters):
        x.requires_grad_(True)
        fval = pt_func(x)
        history.append(fval.item())
        fval.sum().backward()
        grads = x.grad.detach()
        with torch.no_grad():
            dx, hidden = model(x.detach(), grads, t, hidden)
        x = (x.detach() + dx).clamp(sr[0], sr[1])
    return x.detach().cpu().numpy().flatten(), history


# ── NumPy 版测试函数 (供传统优化器使用) ──

def np_rastrigin(x): return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
def np_ackley(x):
    n=len(x); s1=np.sum(x**2); s2=np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2*np.sqrt(s1/n))-np.exp(s2/n)+20+np.e
def np_rosenbrock(x): return sum(100*(x[i+1]-x[i]**2)**2+(1-x[i])**2 for i in range(len(x)-1))

NP_FUNCS = {"rastrigin": np_rastrigin, "ackley": np_ackley, "rosenbrock": np_rosenbrock}


# ── 主流程 ──

def main():
    dim = 10
    n_iters = 200
    n_train = 300
    sr = (-5.0, 5.0)

    print("=== 神经网络优化器 vs 传统优化器 ===")
    print(f"  维度: {dim}, 迭代步数: {n_iters}")

    # 1. 训练
    print("\n1. 训练神经网络优化器 (Rastrigin, 10D) ...")
    t0 = time.perf_counter()
    model, _ = train_optimizer(
        pt_rastrigin, dim=dim, n_iters=n_iters,
        n_episodes=n_train, lr=1e-3, search_range=sr, batch_size=32,
    )
    print(f"   完成 ({time.perf_counter()-t0:.1f}s)")

    # 2. 对比
    test_funcs = ["rastrigin", "ackley", "rosenbrock"]
    results = {}

    for name in test_funcs:
        pt_func = PT_BENCHMARKS[name][0]
        np_func = NP_FUNCS[name]
        print(f"\n2. 测试: {name} ({dim}D)")

        all_h = {"NN": [], "GD": [], "Adam": [], "Random": []}
        for run in range(5):
            np.random.seed(run * 42)
            x0 = np.random.uniform(sr[0], sr[1], dim)
            _, h = grad_descent(np_func, x0, lr=0.01, n_iters=n_iters)
            all_h["GD"].append(h)
            _, h = adam_optim(np_func, x0, lr=0.01, n_iters=n_iters)
            all_h["Adam"].append(h)
            _, h = random_search(np_func, dim, n_iters=n_iters, sr=sr)
            all_h["Random"].append(h)
            _, h = nn_optimize(model, pt_func, dim, n_iters=n_iters, sr=sr)
            all_h["NN"].append(h)

        for k in all_h:
            all_h[k] = np.median(np.array(all_h[k]), axis=0)
        results[name] = all_h

        for k in ["Random", "GD", "Adam", "NN"]:
            print(f"   {k:>8}: final = {all_h[k][-1]:.4f}")

    # 3. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"NN": "#e74c3c", "GD": "#3498db", "Adam": "#2ecc71", "Random": "#95a5a6"}
    labels = {"NN": "Neural Net", "GD": "Grad Descent", "Adam": "Adam", "Random": "Random"}

    for i, name in enumerate(test_funcs):
        ax = axes[i]
        for k in ["Random", "GD", "Adam", "NN"]:
            h = np.maximum(results[name][k], 1e-10)
            ax.semilogy(h, color=colors[k], lw=1.5 if k == "NN" else 1.0,
                       alpha=1.0 if k == "NN" else 0.7, label=labels[k],
                       ls="-" if k == "NN" else "--")
        ax.set_title(f"{name.capitalize()} ({dim}D)", fontsize=13)
        ax.set_xlabel("Iteration"); ax.set_ylabel("f(x) (log)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Neural Network Optimizer vs Traditional Methods", fontsize=14)
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图片已保存: {out}")
    plt.close()


if __name__ == "__main__":
    main()
