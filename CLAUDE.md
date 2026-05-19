# sparsearray — 稀布阵列天线优化库

## 项目概述

Python 稀布/稀疏阵列天线优化库。自研 CMA-ES 优化器 + stick-breaking 映射器。
优化器外部零依赖（仅 numpy），方向图计算用 numpy 向量化。

## 目录

```
sparse/
├── antopt/                         # 核心库
│   ├── optimizer.py                # CMA-ES + DE + GWO, minimize() API
│   ├── mapping.py                  # Stick-breaking 映射器 (shim)
│   ├── pattern.py                  # AF 计算 + 归一化 (shim)
│   ├── analysis.py                 # PSLL / 峰值搜索 / 方向性系数
│   ├── element_pattern.py          # HFSS CSV 单元方向图导入
│   ├── config.py                   # 分层配置 (Base/Sparse/ASM/MSM)
│   ├── hfss_io.py                  # HFSS CSV 读取 + AEP 加载
│   ├── plotting.py                 # 标准对比图
│   ├── surrogate.py                # Kriging 代理模型
│   ├── utils.py                    # JSON / 方向图辅助
│   ├── core/                       # 核心计算子包
│   │   ├── geometry.py             # Element / LinearArray / PlanarArray
│   │   ├── pattern.py              # AF 计算
│   │   └── element_pattern.py      # 单元方向图
│   ├── matrix_spacing/             # 位置映射
│   │   └── linear.py               # LMMapper (线阵 stick-breaking)
│   └── space_mapping/              # 空间映射
│       ├── base.py                 # Broyden / PE 公共逻辑
│       ├── asm.py                  # 渐进空间映射
│       └── msm.py                  # 流形空间映射
├── hfss/                            # HFSS 接口 (共享)
│   ├── run_patch.py                 # HFSS 贴片仿真
│   ├── run_aep.py                  # HFSS AEP 仿真
│   └── config.json
├── tests/                           # pytest (90 tests)
├── examples/
│   ├── 线阵稀布幅相优化/             # 主工程 (副瓣/零陷/三零陷)
│   ├── 渐进空间稀布幅相优化线阵/     # ASM 空间映射
│   └── ...
└── docs/
```

## 环境

- Python 3.10+, numpy, scipy, matplotlib
- 优化库: pymoo (可选), sklearn (surrogate)
- 测试: pytest

## 核心功能

### 优化任务
- **副瓣优化** (sidelobe): 最小化 PSLL，支持增益下降约束、方向性系数约束
- **单零陷** (null): 指定角度零陷 + 副瓣约束
- **三零陷** (triple_null): 三个独立零陷角度

### 幅度优化
- 支持线性值或 dB 值 (`amplitude.inDB`)
- 步长约束 (`amplitude.step`)，优化后自动量化到网格
- dB 转换: `linear = 10^(-dB/20)`

### 相位优化
- 步长约束 (`phase.step`)，弧度单位

### 方向性系数
- `compute_directivity(pattern, theta_range, phi_range)`
- 自动处理 HFSS 约定 (θ<0 翻正)
- 支持 1D (单 φ 面) 和 2D 方向图

## 运行

```bash
python -m pytest tests/ -v
cd examples/线阵稀布幅相优化 && python main.py
```

## C++ 参考

`E:\Documents\南理工\阵列天线稀疏\Sparse\`
