# sparsearray — 稀布阵列天线优化库

## 项目概述

Python 稀布/稀疏阵列天线优化库。自研 CMA-ES 优化器 + stick-breaking 映射器。
优化器外部零依赖（仅 numpy），方向图计算用 numpy 向量化。

## 目录

```
sparse/
├── antopt/                         # 核心库
│   ├── optimizer.py                # 自研 CMA-ES + DE + GWO, minimize() API
│   ├── mapping.py                  # Stick-breaking 映射器 (use_sigmoid 开关)
│   ├── pattern.py                  # AF 计算 + 归一化 (对称/非对称, 多频/多角度)
│   ├── analysis.py                 # PSLL / 峰值搜索 / _find_peaks
│   ├── space_mapping.py            # 渐进空间映射 (⚠ PE 收敛待修复)
│   ├── element_pattern.py          # HFSS CSV 单元方向图导入
│   ├── surrogate.py                # Kriging 代理模型 (sklearn GPR + EI + LHS)
│   ├── geometry.py                 # Element / LinearArray / PlanarArray
│   └── utils.py                    # JSON / 方向图辅助
├── benchmarks/                     # 手动对比测试脚本
├── tests/                          # pytest (84+ tests)
├── examples/
│   ├── 线阵稀布幅相优化/            # 主工程 (CMA + checkpoint/resume)
│   ├── 线阵稀布幅相优化代理模型/    # Kriging + CMA-ES 示例
│   ├── 渐进空间映射稀布幅相优化线阵/ # ⚠ PE 不收敛，待修复
│   ├── 代理模型调研/                # Kriging/NN/贝叶斯/空间映射对比
│   └── 神经网络/
│       ├── 学习/                   # 神经网络基础概念 + demo
│       ├── PCNN复现/               # PyTorch autograd + L-BFGS-B
│       └── 初步工程/               # NN 代理模型 (PSLLNet)
└── docs/                           # 优化器/映射对比文档 + agent 文档
```

## 当前进展

- **optimizer.py**: 完全自研，`minimize(f, n_vars, method="cma", ...)` 统一入口
  - CMA-ES: C++ 风格，有界 [0,1] / 无界均可，Tent 混沌初始，ThreadPool 并行
  - 支持 checkpoint/resume 断点续跑
  - 已删 pycma(除 space_mapping)/nevergrad 依赖
- **mapping.py**: Stick-breaking 映射，约束天然满足
  - `_halfNe = Ne//2` (去中心占位 bug 已修)
  - `use_sigmoid=False` → clamp [0,1]；`use_sigmoid=True` → sigmoid ℝ→(0,1)
- **surrogate.py**: Kriging 代理模型
  - sklearn GPR (Matern 5/2) + EI 采集函数 + LHS 初始采样
  - `surrogate_optimize()` 入口, CMA-ES 搜索 + EI 填充
- **PCNN 复现**: PyTorch autograd + L-BFGS-B 梯度优化
  - 残余相位分解: `phi_total = phi_progressive + delta_phi`
  - LSE 平滑 PSLL + 精确 PSLL 验证
  - 发现: LSE (τ=10) 严重低估真实副瓣 ~20dB, τ=1000 时差距 ~7dB
- **有界 > 无界**: 全场景下 bounded [0,1] 优于 unbounded sigmoid (位置 +0.5dB, 幅度 +29dB)
- **132元/90.5λ** 达到 -27.02 dB (超过论文 DMDE -25.49)
- **空间映射**: PE CMA-ES 配合粗/细模型误差函数不收敛，待研究

## 开发约定

- Python 3.10+, NumPy 核心依赖
- 预计算优先，测试用 pytest
- 优化器: 默认无约束 (bounds=None), 有界用 `bounds=(lb, ub)`
- 幅度/相位变量不再过 sigmoid，直接用有界优化

## 测试

```bash
python -m pytest tests/ -v      # 84 tests
python benchmarks/optimizer2_132.py  # 132元 CMA 验证
```

## C++ 参考

`E:\Documents\南理工\阵列天线稀疏\Sparse\`

## Agent skills

### Issue tracker

GitHub Issues（gh CLI）. See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one CONTEXT.md + docs/adr/ at repo root. See `docs/agents/domain.md`.
