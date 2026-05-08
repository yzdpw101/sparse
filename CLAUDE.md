# sparsearray - 稀布阵列天线优化库

## 项目概述

Python 实现的稀布/稀疏阵列天线优化库，方向图乘积定理（阵因子 × 单元因子）。
只做核心计算，优化器用外部库（cma、pymoo 等）。

## 目录结构

```
sparse/
├── antopt/                  # 核心计算代码
│   ├── __init__.py
│   ├── pattern.py           # 方向图计算器（AF 计算 + 归一化）
│   ├── geometry.py          # 阵列几何定义（Element, LinearArray, PlanarArray）
│   ├── element_pattern.py   # 单元方向图模型（含 HFSS CSV 导入）
│   ├── analysis.py          # 方向图分析（PSLL, 峰值搜索）
│   ├── mapping.py           # LM 线性映射器（[0,1] → 阵元位置）
│   ├── optimizer.py         # 优化器封装（cma + nevergrad 多算法）
│   └── utils.py             # 工具函数（JSON, 方向图计算）
├── visualization/           # 可视化
├── tests/                   # pytest 测试（84 tests）
├── examples/
│   ├── demos/               # 各类 demo 脚本
│   │   ├── demo_linear.py   # 线阵方向图
│   │   ├── demo_planar.py   # 平面阵方向图
│   │   ├── demo_compare.py  # 多算法对比
│   │   └── benchmark.py     # 计算性能 benchmark
│   ├── 线阵稀布幅相优化/       # 完整工程（可打包为 exe）
│   │   ├── Config.json      # 总配置（仿 C++）
│   │   ├── main.py          # 主入口
│   │   ├── input/           # 导入数据 (array_config/ + element_pattern/)
│   │   ├── result/          # 优化结果 (JSON + figures/)
│   │   └── release/         # PyInstaller 发布 (95MB exe, gitignored)
│   └── 渐进空间映射稀布线阵/    # 空间映射 + HFSS 细模型
├── pyproject.toml
└── README.md
```

## C++ 参考代码

```
E:\Documents\南理工\阵列天线稀疏\Sparse\
├── 稀布幅相优化线阵\
│   ├── src\Main.cpp         # CMA-ES 优化流程 + fitness 函数
│   ├── include\Main.h       # readFeMultiFreqFromCsvs
│   └── Config.json
└── antopt\
    ├── src\antenna\Pattern.cpp
    ├── include\antenna\Pattern.h
    └── include\utils\Extrema.h
```

## 开发约定

- Python 3.10+, NumPy 核心依赖
- 预计算优先：构造时算好常量，计算时只传变化量
- 测试用 pytest
- 代码注释用中文，标识符和 API 用英文
- 所有优化变量 ∈ ℝ 无约束，通过 sigmoid 映射到目标范围

## 开发前查文档

添加新依赖或实现新功能前，用以下工具查最新文档：

1. **Context7** — 查库/SDK/框架最新 API 用法
2. **Firecrawl** — 联网搜索技术方案和最新实现

避免像 cma 4.x 删掉逐代输出这种版本差异导致的问题。

## MCP 科研工具

已安装的 MCP Server（项目级，通过 `claude mcp add`）：

| 工具 | 安装命令 | 用途 |
|---|---|---|
| **semantic-scholar-mcp** | `claude mcp add semantic-scholar -- uvx semantic-scholar-mcp` | 2 亿论文搜索 + AI 推荐 + 引文网络 |
| **arxiv-mcp-server** | `claude mcp add arxiv -- arxiv-mcp-server --storage-path ...` | arXiv 论文搜索/下载/解读 |

用法示例：
- "搜索 2024-2025 年 sparse array optimization 的 arXiv 论文"
- "下载论文 2403.15137 并总结核心方法"
- "这篇论文和哪些其他工作有引用关系？"

## 测试

```bash
python -m pytest tests/ -v
```
