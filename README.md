# sparsearray

稀疏/稀布阵列天线优化库 — Python 实现，支持线阵 & 矩形平面阵。

## 功能

- **方向图计算**: 方向图乘积定理（阵因子 × 单元因子），线阵/平面阵，对称/非对称，多频/多角度
- **稀布优化**: CMA-ES (pycma) / nevergrad (NGOpt/DE/PSO) 多算法对比，无约束变量 + LM 映射器
- **幅相联合优化**: mode 三位编码控制位置/相位/幅度优化，支持从 JSON 导入初始配置
- **单元方向图**: HFSS 导出的多频 CSV 导入（dB/线性，增益/场量自适应）
- **独立打包**: PyInstaller → 95MB exe，无需安装 Python 环境

## 安装

```bash
pip install numpy scipy matplotlib cma nevergrad
```

## 快速开始

```bash
cd examples/线阵稀布幅相优化
# 修改 Config.json 参数
python main.py
```

## 目录结构

```
sparse/
├── antopt/                    # 核心计算代码
│   ├── pattern.py             # 方向图计算器 (AF + 归一化)
│   ├── analysis.py            # 方向图分析 (PSLL, 峰值搜索)
│   ├── mapping.py             # LM 线性映射器
│   ├── optimizer.py           # 优化器封装 (cma + nevergrad)
│   ├── element_pattern.py     # 单元方向图 (HFSS CSV 导入)
│   └── utils.py               # 工具函数 (JSON, 方向图)
├── visualization/             # 可视化 (1D/2D/3D 方向图)
├── tests/                     # pytest 测试 (84 tests)
├── examples/
│   ├── demos/                 # 各类 demo 脚本
│   ├── 线阵稀布幅相优化/       # 完整工程（主入口）
│   │   ├── Config.json        # 总配置
│   │   ├── main.py            # 优化主脚本
│   │   ├── input/             # 导入数据 (array_config/ + element_pattern/)
│   │   └── result/            # 输出 (optResult.json + figures/)
│   └── 渐进空间映射稀布线阵/    # 空间映射 + HFSS 细模型
├── pyproject.toml
└── README.md
```

## C++ 参考

```
E:\Documents\南理工\阵列天线稀疏\Sparse\
├── 稀布幅相优化线阵\          # CMA-ES 优化 (Main.cpp)
└── antopt\                   # C++ 基础库 (Pattern.cpp, Extrema.h)
```

## License

MIT
