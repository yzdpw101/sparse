# sparsearray - 稀布阵列天线优化库

## 项目概述

Python 实现的稀布/稀疏阵列天线优化库，方向图乘积定理（阵因子 × 单元因子）。
只做核心计算，优化器用外部库（cma、pymoo、scipy.optimize 等）。

## 目录结构

```
sparse/
├── antopt/                  # 核心计算代码
│   ├── __init__.py
│   ├── pattern.py         # 方向图计算器（AF 计算 + 归一化）
│   ├── geometry.py        # 阵列几何定义（Element, LinearArray, PlanarArray）
│   ├── element_pattern.py # 单元方向图模型
│   └── analysis.py        # 方向图分析（PSLL, 峰值搜索）
├── tests/                 # pytest 测试
│   ├── conftest.py
│   ├── test_geometry.py
│   └── test_pattern.py
├── docs/                  # 文档
├── examples/              # 示例/notebook
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## C++ 参考代码

```
E:\Documents\南理工\阵列天线稀疏\Sparse\
├── 稀布幅相优化线阵\          # 最新版可执行程序
│   ├── src\Main.cpp         # CMA-ES 优化流程 + fitness 函数
│   ├── include\Main.h       # readFeMultiFreqFromCsvs
│   └── Config.json
├── antopt\                  # C++ 基础库
│   ├── src\antenna\
│   │   └── Pattern.cpp      # AF 计算核心（831 行）
│   ├── include\antenna\
│   │   └── Pattern.h
│   └── include\utils\
│       └── Extrema.h        # extrema1D 峰值检测
└── antopt.sdf
```

## 当前状态

- [x] `antopt/geometry.py` — Element、ArrayGeometry、LinearArray、PlanarArray、UniformLinearArray
- [x] `antopt/pattern.py` — 方向图计算器
  - [x] `__init__()` 预计算角度网格和 EM 常量
  - [x] `linear_af()` — 线阵非对称单频单角度
  - [x] `linear_af_symmetric()` — 对称阵列
  - [x] `linear_af_multi_scan()` — 多角度扫描
  - [x] `planar_af()` — 平面阵
  - [x] `planar_af_symmetric()` — 平面阵四象限对称
  - [x] `normalize()` — 归一化 dB 方向图
- [x] `antopt/element_pattern.py` — 各向同性、cosine-q、贴片、HFSS 导入
- [x] 基础测试通过 (12 tests)
- [ ] 后续：完善 Pattern 其他方法、方向图分析 utility、可视化

## Pattern 类的设计

构造时预计算角度相关量，计算时只传位置：

```python
pat = Pattern(theta, k=2*np.pi, scan_theta=0.0)
af = pat.linear_af(positions, amplitudes, phases)
```

- **构造预计算**：sinθ、sinθ₀、deltaSin 一次性算好，避免优化循环中重复计算
- **默认 k=2π**：位置均为波长数，不使用 wavelength 参数
- **不支持副瓣分析**：MSLL/HPBW 等由调用方自行处理

支持的维度组合（逐步实现中）：

| 维度 | 选项 |
|---|---|
| 阵列类型 | 线阵、平面阵 |
| 对称性 | 非对称、对称 |
| 频率 | 无波长 (k=2π)、单频有波长 (k=2π/λ)、多频 (多个 Pattern 实例) |
| 扫描角 | 单角度、多角度数组 |

## C++ → Python 映射

| C++ | Python | 说明 |
|---|---|---|
| `ElementBlock` | `Element` dataclass + `ArrayGeometry` | |
| `calcAF(deltaSinTheta, wlpos, exc)` | `Pattern.__init__` 预算 `_delta_sin` + `linear_af(wl_positions)` | 参数名 wl_ 前缀表示以 λ₀ 为单位 |
| `calcSymmetryAF` | `linear_af_symmetric(half_wl_positions)` | |
| `calcMultiFreqAF` | `Pattern(frequenciesGHz=array)` | 多频自动切换波数 |
| `calcMultAngleAF` | `Pattern(theta0s_deg=array)` | 多角度自动广播 |
| `calcUVAF` | `planar_af(wl_x, wl_y)` | |
| `calcSymmetryUVAF` | `planar_af_symmetric(quarter_wl_x, quarter_wl_y)` | |
| `extrema1D` + Main.cpp fitness | 调用方自行处理 | MSLL/HPBW 不在 Pattern 中 |

## 开发约定

- Python 3.10+, NumPy 核心依赖
- 预计算优先：构造时算好常量，计算时只传变化量
- 测试用 pytest
- 代码注释用中文，标识符和 API 用英文

## 测试

```bash
python -m pytest tests/ -v
```

当前 12 个测试全部通过：
- geometry (7 个): 构造、对称性、批量设置、activate_subset
- pattern (5 个): 输出形状、法向峰值、扫描、单阵元退化、归一化
