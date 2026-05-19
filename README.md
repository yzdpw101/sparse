# sparsearray

稀布/稀疏阵列天线优化库 — Python 实现。

## 功能

- **方向图计算**: 阵因子 × 单元因子，线阵/平面阵，对称/非对称，多频/多角度
- **稀布优化**: 自研 CMA-ES（有界 [0,1] + stick-breaking 映射），支持 DE / GWO
- **幅相联合优化**: 支持线性值/dB 值优化，步长约束自动量化
- **优化任务**: 副瓣优化、单零陷、三零陷，支持增益下降/方向性系数约束
- **方向性系数**: `compute_directivity()` 自动处理 HFSS/物理约定转换
- **空间映射**: 渐进空间映射 (ASM) + 流形空间映射 (MSM)
- **单元方向图**: HFSS CSV 导入，支持 AEP 模式
- **配置驱动**: JSON 配置 + 分层 Config 类 (Base/Sparse/ASM/MSM)

## 安装

```bash
pip install numpy scipy matplotlib
```

## 快速开始

```bash
cd examples/线阵稀布幅相优化
# 修改 Config.json → 运行
python main.py
```

## 目录

```
sparse/
├── antopt/                         # 核心库
│   ├── optimizer.py                # CMA-ES + DE + GWO (minimize API)
│   ├── analysis.py                 # PSLL / 峰值搜索 / compute_directivity
│   ├── config.py                   # 分层配置 (Base/Sparse/ASM/MSM)
│   ├── hfss_io.py                  # HFSS CSV 读取 + AEP 加载
│   ├── plotting.py                 # 标准对比图
│   ├── element_pattern.py          # HFSS CSV 单元方向图
│   ├── core/                       # 核心计算子包
│   ├── matrix_spacing/             # 位置映射 (LMMapper)
│   └── space_mapping/              # 空间映射 (ASM/MSM)
├── hfss/                            # HFSS 接口 (共享)
├── tests/                           # pytest (90 tests)
├── examples/
│   ├── 线阵稀布幅相优化/             # 主工程
│   ├── 渐进空间稀布幅相优化线阵/     # ASM 空间映射
│   └── ...
└── docs/
```

## License

MIT
