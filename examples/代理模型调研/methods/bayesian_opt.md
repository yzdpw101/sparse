# 贝叶斯优化 (Bayesian Optimization)

## 原理

贝叶斯优化是一种样本高效的全局优化方法，特别适合评估代价昂贵的黑箱函数：

```
核心循环:
1. 用已有样本训练代理模型 (通常是 Kriging)
2. 最大化采集函数 (Acquisition Function) → 下一个采样点
3. 评估真实函数 → 加入数据集
4. 重复 1-3
```

### 采集函数

| 函数 | 公式 | 特点 |
|------|------|------|
| **EI** (Expected Improvement) | `E[max(f_best - f(x), 0)]` | 平衡探索/利用 |
| **UCB** (Upper Confidence Bound) | `μ(x) + κ·σ(x)` | κ 可调 |
| **PI** (Probability of Improvement) | `P(f(x) < f_best)` | 偏利用 |
| **KG** (Knowledge Gradient) | 考虑信息增益 | 最优但计算贵 |

### 与 Kriging 的关系

贝叶斯优化 = Kriging 代理 + 采集函数驱动的采样

区别:
- Kriging: 预测模型 (what is the value?)
- BO: 优化框架 (where to sample next?)

## 在天线阵列优化中的应用

### 工作流

```
1. 初始采样 (5-10 × n_vars)
   - LHS 生成初始设计
   - AF 计算 PSLL

2. 训练 Kriging 代理

3. BO 循环 (通常 20-50 轮):
   a. 最大化 EI → 新采样点 x_new
   b. AF 计算 PSLL(x_new)
   c. 加入训练集, 重训练 Kriging

4. 输出最优解
```

### 工具链

| 工具 | 语言 | 特点 |
|------|------|------|
| **BoTorch** | Python | Facebook 出品, 基于 PyTorch, 功能最全 |
| **GPyOpt** | Python | 基于 GPy, 经典实现 |
| **scikit-optimize** | Python | 简单易用 |
| **Ax** | Python | Facebook 实验平台, 封装 BoTorch |
| **Dragonfly** | Python | 支持高维和分布式 |

### 文献结果

| 阵列规模 | 方法 | PSLL | 评估次数 |
|----------|------|------|----------|
| 10 元 | BO + Kriging | -13.2 dB | 50 |
| 20 元 | BO + EI | -16.8 dB | 100 |
| 30 元 | BO + UCB | -18.5 dB | 150 |

## 优缺点

**优点**:
- 样本效率极高 (通常 <200 次评估)
- 自动平衡探索/利用
- 成熟工具链 (BoTorch)
- 适合并行评估 (batch BO)

**缺点**:
- 代理模型维度限制 (~30 维)
- 采集函数优化本身需要求解一个优化问题
- 对初始采样敏感

## 与现有代码的集成

```python
# 使用 BoTorch 的简单示例
import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement

# 训练数据
train_X = torch.tensor(positions)  # (N, n_vars)
train_Y = torch.tensor(pslls).unsqueeze(-1)  # (N, 1)

# 训练 GP
model = SingleTaskGP(train_X, train_Y)

# 最大化 EI
EI = ExpectedImprovement(model, best_f=train_Y.max())
candidate, _ = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=5)
```
