"""神经网络优化器 — 用 GRU 学习优化策略 (全 PyTorch 可微分)。

核心思想:
  传统优化器用手工规则决定搜索方向和步长。
  神经网络优化器通过训练, 自动学习从 (位置, 梯度, 历史) → (下一步移动) 的映射。

训练方式:
  在一批随机初始点上展开优化轨迹, 每步的函数值作为损失,
  端到端反向传播更新 GRU 参数。

架构:
  GRU(2*dim+1 → hidden) → Linear(hidden → dim)
  输入: [位置, 梯度, 归一化步数]
  输出: 位置更新 Δx
"""

import torch
import torch.nn as nn


class NeuralOptimizer(nn.Module):
    """基于 GRU 的神经网络优化器。"""

    def __init__(self, dim: int, hidden_size: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_size = hidden_size
        input_dim = 2 * dim + 1
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, dim)
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, grad, step, hidden=None):
        batch = x.shape[0]
        t_val = torch.tanh(torch.tensor(step / 500.0, device=x.device))
        t = t_val.unsqueeze(0).expand(batch, 1)
        inp = torch.cat([x, grad, t], dim=-1).unsqueeze(1)  # (batch, 1, input_dim)
        out, hidden = self.gru(inp, hidden) if hidden is not None else self.gru(inp)
        dx = self.fc(out.squeeze(1))
        return dx, hidden


# ── 可微分测试函数 (PyTorch) ──

def pt_rastrigin(x):
    A = 10.0
    return A * x.shape[1] + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x), dim=1)


def pt_ackley(x):
    n = x.shape[1]
    sum1 = torch.sum(x**2, dim=1)
    sum2 = torch.sum(torch.cos(2 * torch.pi * x), dim=1)
    return -20 * torch.exp(-0.2 * torch.sqrt(sum1 / n)) - torch.exp(sum2 / n) + 20 + torch.e


def pt_rosenbrock(x):
    return torch.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, dim=1)


def pt_griewank(x):
    n = x.shape[1]
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    sum_sq = torch.sum(x**2, dim=1) / 4000
    prod_cos = torch.prod(torch.cos(x / torch.sqrt(idx)), dim=1)
    return sum_sq - prod_cos + 1


PT_BENCHMARKS = {
    "rastrigin":  (pt_rastrigin,  [-5.12, 5.12]),
    "ackley":     (pt_ackley,     [-32.768, 32.768]),
    "rosenbrock": (pt_rosenbrock, [-5.0, 10.0]),
    "griewank":   (pt_griewank,   [-600.0, 600.0]),
}


def train_optimizer(
    func,
    dim=10,
    n_iters=200,
    n_episodes=500,
    lr=1e-3,
    search_range=(-5.0, 5.0),
    batch_size=32,
    device="cpu",
):
    """训练神经网络优化器。

    流程:
      1. 随机采样 batch_size 个初始点
      2. 逐步用 NN 优化器更新位置
      3. 每步的函数值参与损失计算
      4. 反向传播, 端到端更新 GRU 参数

    Returns:
        model, history
    """
    model = NeuralOptimizer(dim, hidden_size=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    for ep in range(n_episodes):
        # 随机初始点
        x = torch.rand(batch_size, dim, device=device)
        x = x * (search_range[1] - search_range[0]) + search_range[0]

        hidden = None
        total_loss = torch.tensor(0.0, device=device)

        for t in range(n_iters):
            x.requires_grad_(True)
            fvals = func(x)  # (batch,)

            # 计算梯度作为 NN 输入 (autograd)
            fvals.sum().backward()
            grads = x.grad.detach()

            # 隐藏状态断开 BPTT (节省内存)
            if hidden is not None:
                hidden = hidden.detach()

            # NN 决定下一步 (保持可微分链: dx → 模型参数)
            dx, hidden = model(x.detach(), grads, t, hidden)

            # 更新位置 (x_new 可微分依赖 dx → 模型参数)
            x_new = (x.detach() + dx).clamp(search_range[0], search_range[1])

            # 累积损失: 新位置的平均函数值
            total_loss = total_loss + func(x_new).mean()

            # 断开图, 下一轮迭代
            x = x_new.detach()

        # 反向传播: loss → func(x_new) → x_new → dx → model weights
        loss = total_loss / n_iters
        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append(func(x.detach()).mean().item())
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes}, loss={history[-1]:.4f}")

    return model, history
