# 代理模型替代空间映射 — 技术调研报告

## 1. 问题背景

### 1.1 粗模型 vs 细模型

| | 粗模型 | 细模型 |
|---|---|---|
| 方法 | 方向图乘积定理 (AF × Fe) | HFSS 全波仿真 (FEM) |
| 耗时 | ~ms | ~min |
| 精度 | 忽略互耦 | 完整电磁模型 |
| 用途 | 优化搜索 | 验证/精调 |

### 1.2 空间映射 (SM) 的尝试与失败

**已尝试的方法**: Aggressive Space Mapping (ASM)

**流程**: 粗模型 CMA-ES 优化 → HFSS 评估 → PE (参数提取) → Broyden 更新

**失败现象**: PSLL 从 -14.05 改善到 -14.90 后收敛停滞，残差范数不下降。

**根因分析**:

1. **映射不光滑**: 互耦效应导致位置→方向图的映射是非线性、非光滑的。Broyden 拟牛顿方法假设映射是光滑的，无法逼近真实 Jacobian。
2. **响应空间不匹配**: 标准 SM 对齐 S11 参数（跨频率，一维标量函数），而方向图是角度的函数（18001 维向量），对齐难度大得多。
3. **PE 陷入局部最优**: 参数提取本身是一个优化问题，对于复杂方向图容易陷入局部最优。

## 2. 替代方案对比

| 方法 | 精度 | 训练样本 | 适用规模 | 实现难度 | 与现有代码兼容 |
|------|------|----------|----------|----------|----------------|
| **Kriging (GPR)** | 高 | 50-200 | ≤30 维 | 低 | 高 (sklearn) |
| **贝叶斯优化 (BO)** | 高 | 增量式 | ≤30 维 | 中 | 高 (BoTorch) |
| **RBF 网络** | 中 | 100-500 | ≤100 维 | 低 | 高 |
| **神经网络 (NN)** | 中 | 1000+ | 任意 | 高 | 中 |
| **多保真度 Kriging** | 高 | 30-100 | ≤30 维 | 中 | 中 |
| **直接代理+CMA-ES** | 高 | 200-500 | ≤50 维 | 低 | 最高 |

## 3. 推荐方案

### 首选: Kriging 代理模型 + CMA-ES

**理由**:

1. **现有代码直接复用**: `antopt/optimizer.py` (CMA-ES)、`mapping.py` (stick-breaking)、`pattern.py` (AF 计算) 构成完整管线
2. **样本效率高**: 50-200 次 AF 计算即可训练，比 HFSS 采样快 1000 倍
3. **实现简单**: `sklearn.gaussian_process.GaussianProcessRegressor` + Matern 5/2 核
4. **增量更新**: 新样本加入后可快速重训练

**工作流**:

```
1. LHS 采样 → mapper.synthesize() → positions
2. AF 计算 → pattern.fitness() → PSLL (粗模型)
3. 训练 Kriging: positions → PSLL
4. CMA-ES 在 Kriging 上优化 → 候选解
5. 候选解 → AF 验证 → 加入训练集
6. 重复 3-5 直到收敛
```

**关键参数**:
- 核函数: Matern 5/2 (适合非光滑 PSLL 景观)
- 初始样本: 10 × n_vars (至少)
- 增量采样: 每轮 5-10 个新样本
- 终止条件: 连续 3 轮无改善

### 备选: 多保真度 Kriging

如果需要融合粗模型 (AF) 和细模型 (HFSS) 的信息：
- 粗模型: AF 计算 (大量样本，~ms)
- 细模型: HFSS 仿真 (少量样本，~min)
- 多保真度 Kriging 自动学习两者关系

## 4. 与现有代码的集成点

```
antopt/
├── optimizer.py      # CMA-ES: minimize() → 直接用于代理模型优化
├── mapping.py        # LMMapper: synthesize() → 生成训练样本位置
├── pattern.py        # Pattern: linear_af() → 计算训练标签 (PSLL)
├── analysis.py       # get_psll() → 目标函数
└── surrogate.py      # [新增] Kriging 代理模型
```

新增文件 `antopt/surrogate.py` 需要实现:
- `KrigingSurrogate`: 封装 sklearn GPR
- `surrogate_fitness()`: 替换 pattern fitness 的廉价版本
- `infill_criterion()`: 选择下一个采样点 (EI/UCB)

## 5. 参考文献

详见 [references.md](references.md)
