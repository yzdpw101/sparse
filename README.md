# sparsearray

稀疏/稀布阵列天线优化库 (Sparse Antenna Array Optimization Library)

用于阵列天线稀布优化，覆盖 on-grid thinning 和 off-grid 连续位置优化两种模式，支持多种优化算法和 HFSS 全波验证。

## 安装

```bash
pip install sparsearray
```

## 快速开始

```python
import sparsearray as sa
import numpy as np

# 定义均匀直线阵作为基准
ula = sa.UniformLinearArray(num_elements=20, spacing=0.5, frequency=10e9)

# 计算阵因子
af = sa.ArrayFactor(ula)
theta = np.linspace(-90, 90, 1801)
pattern = af.compute_af_normalized(theta)

# 分析方向图
p = sa.Pattern(theta, pattern)
print(f"MSLL: {p.get_msll():.2f} dB")
print(f"HPBW: {p.get_hpbw():.2f} deg")
```

## 目录结构

```
sparsearray/
├── core/           # 核心计算 (几何、阵因子、方向图分析)
├── optimization/   # 优化问题定义 (约束、目标函数)
├── optimizers/     # 优化算法 (GA, PSO, DE, CMA-ES)
├── visualization/  # 可视化 (阵列几何、方向图、收敛曲线)
├── integration/    # 外部工具集成 (HFSS, I/O)
└── utils/          # 工具函数
```

## License

MIT
