"""训练神经网络代理模型 — 学习 位置 → PSLL 的映射。

用法: cd examples/神经网络/初步工程 && python train.py

需要先运行 Kriging 代理的 train_surrogate.py 生成 training_data.npz,
或者自行生成训练数据。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import PSLLNet

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent.parent / "线阵稀布幅相优化代理模型" / "training_data.npz"
MODEL_PATH = HERE / "model.pt"


def load_data():
    """加载训练数据 (X=位置, Y=PSLL)。"""
    if not DATA_PATH.exists():
        print(f"错误: 训练数据不存在: {DATA_PATH}")
        print("请先运行 examples/线阵稀布幅相优化代理模型/train_surrogate.py")
        sys.exit(1)

    data = np.load(DATA_PATH)
    X = torch.tensor(data["X"], dtype=torch.float32)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    return X, Y


def train():
    """训练神经网络。"""
    X, Y = load_data()
    n_vars = X.shape[1]
    n_samples = len(X)

    # 划分训练/验证集 (80/20)
    n_train = int(0.8 * n_samples)
    idx = torch.randperm(n_samples)
    X_train, X_val = X[idx[:n_train]], X[idx[n_train:]]
    Y_train, Y_val = Y[idx[:n_train]], Y[idx[n_train:]]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=64, shuffle=True
    )

    # 创建模型
    model = PSLLNet(n_vars=n_vars, hidden=128, n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()

    print(f"=== 训练神经网络代理 ===")
    print(f"  输入维度: {n_vars}")
    print(f"  训练样本: {n_train}, 验证样本: {n_samples - n_train}")
    print(f"  网络结构: {n_vars} → 128 → 128 → 128 → 1")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    patience = 50
    no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(1, 1001):
        # 训练
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()
            val_mae = (val_pred - Y_val).abs().mean().item()

        scheduler.step()

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            print(f"  Epoch {epoch:4d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_MAE={val_mae:.4f} dB, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}, {elapsed:.1f}s")

        if no_improve >= patience:
            print(f"  早停: 连续 {patience} 轮无改善")
            break

    elapsed = time.perf_counter() - t0
    print(f"\n  训练完成: {elapsed:.1f}s, 最佳验证 loss: {best_val_loss:.4f}")
    print(f"  模型已保存: {MODEL_PATH}")


if __name__ == "__main__":
    train()
