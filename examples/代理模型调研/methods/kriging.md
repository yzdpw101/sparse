# Kriging (高斯过程回归) 代理模型

## 原理

Kriging 是一种基于随机过程的插值方法，假设目标函数是一个高斯过程的实现：

```
f(x) = μ + Z(x)

其中:
- μ: 全局均值 (常数或线性趋势)
- Z(x): 均值为 0 的高斯随机场，协方差由核函数定义
```

### 核函数选择

| 核函数 | 公式 | 适用场景 |
|--------|------|----------|
| **Matern 5/2** | `(1 + √5r + 5r²/3) exp(-√5r)` | 非光滑函数 (推荐用于 PSLL) |
| Matern 3/2 | `(1 + √3r) exp(-√3r)` | 中等光滑度 |
| RBF (SE) | `exp(-r²/2)` | 光滑函数 (不适合 PSLL) |

**PSLL 景观特点**: 有大量局部极小值、不连续的梯度变化 → Matern 5/2 是最佳选择。

### 预测公式

给定训练集 {(x_i, y_i)} 和新点 x_*:

```
μ_* = μ + k(x_*)^T K^{-1} (y - μ·1)
σ²_* = k(x_*, x_*) - k(x_*)^T K^{-1} k(x_*)
```

其中 K 是训练点的核矩阵 (N×N)，k(x_*) 是新点与训练点的核向量。

**计算复杂度**: O(N³) 矩阵求逆 → 训练样本不宜超过 ~200。

## 在天线阵列优化中的应用

### 典型工作流

```
1. 初始设计 (DoE)
   - Latin Hypercube Sampling (LHS) 生成 N_init 个位置配置
   - 每个配置用 AF 计算 PSLL
   - 训练 Kriging: x → PSLL

2. 代理模型优化
   - CMA-ES/GA 在 Kriging 上搜索最优解
   - 输出: 候选解 x_candidate

3. 增量采样 (Infill)
   - 用 EI/UCB 准则评估候选解的质量
   - 选择最有价值的点加入训练集
   - 重训练 Kriging

4. 重复 2-3 直到收敛
```

### 采样准则

**Expected Improvement (EI)**:

```
EI(x) = (μ(x) - f_best) Φ(Z) + σ(x) φ(Z)
Z = (μ(x) - f_best) / σ(x)
```

- 平衡探索 (高 σ 区域) 和利用 (低 μ 区域)
- 适合单目标优化

**Upper Confidence Bound (UCB)**:

```
UCB(x) = μ(x) + κ · σ(x)
```

- κ 控制探索/利用权衡
- κ 大 → 偏探索, κ 小 → 偏利用

### 文献结果

| 阵列规模 | 方法 | PSLL | HFSS 调用次数 |
|----------|------|------|---------------|
| 20 元 | Kriging + GA | -17.8 dB | ~150 |
| 50 元 | Kriging + DE | -20.1 dB | ~400 |
| 100 元 | ECO + Kriging | -22.3 dB | ~300 |
| 132 元 | CMA-ES (无代理) | -27.0 dB | 0 (纯 AF) |

## 优缺点

**优点**:
- 预测自带不确定性估计 (σ²)
- 样本效率高 (50-200 个样本)
- 实现简单 (sklearn 一行代码)
- 可增量更新

**缺点**:
- O(N³) 复杂度，样本数受限
- 高维 (>30 维) 性能下降
- 核函数选择影响大

## 与现有代码的集成

```python
# antopt/surrogate.py (待实现)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class KrigingSurrogate:
    def __init__(self, n_vars):
        kernel = Matern(nu=2.5, length_scale=np.ones(n_vars))
        self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

    def fit(self, X, y):
        """训练: X=位置配置, y=PSLL"""
        self.gpr.fit(X, y)

    def predict(self, X, return_std=True):
        """预测: 返回 (均值, 标准差)"""
        return self.gpr.predict(X, return_std=return_std)

    def ei(self, X, f_best):
        """Expected Improvement 采样准则"""
        mu, sigma = self.predict(X, return_std=True)
        Z = (mu - f_best) / (sigma + 1e-30)
        ei = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei
```
