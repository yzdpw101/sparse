"""神经网络代理模型定义。"""

import torch
import torch.nn as nn


class PSLLNet(nn.Module):
    """位置 → PSLL 的神经网络代理。

    输入: 优化变量 x (n_vars,)
    输出: PSLL 预测值 (1,)
    """

    def __init__(self, n_vars=15, hidden=128, n_layers=3):
        """
        参数:
            n_vars   — 输入维度 (优化变量个数)
            hidden   — 每层隐藏神经元数
            n_layers — 隐藏层数 (不含输入/输出层)
        """
        super().__init__()
        layers = []
        in_dim = n_vars
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播。"""
        return self.net(x).squeeze(-1)
