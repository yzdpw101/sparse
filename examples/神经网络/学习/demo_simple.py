"""最简单的神经网络演示 — 学会 y = sin(x) 的映射。

理解神经网络的四个核心概念:
  1. 数据: 输入 → 输出 的对应关系
  2. 网络: 一堆权重 W 和偏置 b
  3. 训练: 调 W 和 b 让预测逼近真实值
  4. 预测: 用训练好的网络对新输入给输出

运行: cd examples/神经网络/学习 && python demo_simple.py
"""
import numpy as np
import torch
import torch.nn as nn

print("=" * 50)
print("演示: 神经网络学习 y = sin(x)")
print("=" * 50)

# ═══════════════════════════════════════════════════
# 第一步: 生成数据
# ═══════════════════════════════════════════════════
# 生成 50 个训练点
X_train = np.random.uniform(-np.pi, np.pi, 100000).astype(np.float32)
Y_train = np.sin(X_train).astype(np.float32)

print(f"\n[数据] 生成 {len(X_train)} 个训练点")
print(f"  输入 X 范围: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"  输出 Y 范围: [{Y_train.min():.2f}, {Y_train.max():.2f}]")

# ═══════════════════════════════════════════════════
# 第二步: 定义网络
# ═══════════════════════════════════════════════════
# 最简单的网络: 1 → 16 → 16 → 1
net = nn.Sequential(
    nn.Linear(1, 200),    # 第1层: 1个输入 → 16个神经元
    nn.ReLU(),           # 激活函数
    nn.Linear(200, 200),   # 第2层: 16 → 16
    nn.ReLU(),
    nn.Linear(200, 1)     # 第3层: 16 → 1个输出
)

print(f"\n[网络] 结构: 1 → 16 → 16 → 1")
print(f"  参数量: {sum(p.numel() for p in net.parameters())}")

# 打印第一层的 W 和 b 形状
W1 = net[0].weight  # shape: (16, 1)
b1 = net[0].bias    # shape: (16,)
print(f"\n  第1层 W 形状: {list(W1.shape)}  (16个神经元 × 1个输入)")
print(f"  第1层 b 形状: {list(b1.shape)}  (16个偏置)")
print(f"  W1 的值 (前4个): {W1.data[:4].flatten().tolist()}")

# ═══════════════════════════════════════════════════
# 第三步: 训练
# ═══════════════════════════════════════════════════
X_t = torch.tensor(X_train).unsqueeze(1)  # (50, 1)
Y_t = torch.tensor(Y_train)               # (50,)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

print(f"\n[训练] 开始训练 (500 轮)...")

for epoch in range(1, 501):
    # 前向传播: 输入 → 网络 → 预测值
    Y_pred = net(X_t).squeeze(1)

    # 算损失: 预测和真实的差距
    loss = criterion(Y_pred, Y_t)

    # 反向传播: 算梯度
    optimizer.zero_grad()
    loss.backward()

    # 更新 W 和 b
    optimizer.step()

    if epoch % 100 == 0:
        # 算一下在未见过的数据上的误差
        X_test = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]).unsqueeze(1)
        Y_true = torch.sin(X_test.squeeze(1))
        with torch.no_grad():
            Y_test_pred = net(X_test).squeeze(1)
        test_err = (Y_test_pred - Y_true).abs().mean().item()
        print(f"  Epoch {epoch:4d}: loss={loss.item():.6f}, 测试误差={test_err:.4f}")

# ═══════════════════════════════════════════════════
# 第四步: 验证
# ═══════════════════════════════════════════════════
print(f"\n[验证] 用训练好的网络预测:")
test_inputs = [-3.0, -1.5, 0.0, 1.5, 3.0]
print(f"  {'输入 x':>8s}  {'sin(x) 真实':>10s}  {'网络预测':>8s}  {'误差':>6s}")
print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}")

X_test = torch.tensor(test_inputs, dtype=torch.float32).unsqueeze(1)
Y_true = np.sin(test_inputs)
with torch.no_grad():
    Y_pred = net(X_test).squeeze(1).numpy()

for i in range(len(test_inputs)):
    err = abs(Y_pred[i] - Y_true[i])
    print(f"  {test_inputs[i]:8.2f}  {Y_true[i]:10.4f}  {Y_pred[i]:8.4f}  {err:.4f}")

# ═══════════════════════════════════════════════════
# 对比: 没训练的网络是什么样
# ═══════════════════════════════════════════════════
print(f"\n[对比] 如果不训练，网络输出是什么:")
net_new = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
with torch.no_grad():
    Y_random = net_new(X_test).squeeze(1).numpy()
for i in range(len(test_inputs)):
    print(f"  x={test_inputs[i]:5.1f}  sin(x)={Y_true[i]:7.4f}  随机网络={Y_random[i]:7.4f}")

print(f"\n{'=' * 50}")
print("结论: 训练前网络是随机的, 训练后网络学会了 sin(x) 的规律")
print(f"{'=' * 50}")
