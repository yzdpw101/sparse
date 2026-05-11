# 稀布阵列优化 — 文档索引

## 目录

| 文档 | 内容 |
|---|---|
| [optimizers/comparison.md](optimizers/comparison.md) | 优化器全面对比 (CMA-ES vs DMDE vs GWO) |
| [optimizers/cma_variants.md](optimizers/cma_variants.md) | CMA-ES 变体 (pycma vs C++ 风格 vs 混沌初始化) |
| [mapping/comparison.md](mapping/comparison.md) | 映射方法对比 (stick-breaking vs proportional vs 直接位置) |
| [results/final.md](results/final.md) | 最终结果汇总 |

## 结论速览

| 方法 | 78元/73λ | 132元/90.5λ | 152元/98.5λ |
|---|---|---|---|
| **CMA-ES + stick-breaking** | -22.34 | **-25.02** | -24.12 |
| CMA-ES + 混沌 + SB | -21.97 | -25.00 | **-24.27** |
| DMDE + stick-breaking | -21.17 | -23.64 | -23.63 |
| GWO + stick-breaking | -18.11 | -20.90 | -21.66 |
| 直接位置 + 修复 | -19.98 | -22.71 | -23.24 |
| *论文 DMDE (直接位置)* | *-23.01* | *-25.49* | *-24.20* |

**最优方案: stick-breaking 映射 + C++ 风格 CMA-ES (可选 Tent 混沌初始化)。**

- Stick-breaking 天然满足间距约束，映射平滑可微，CMA-ES 协方差矩阵学习层级依赖
- DMDE/GWO 无法学习变量间依赖，比 CMA-ES 差 2-5 dB
- 直接位置编码 + 排序修复破坏优化景观，比 stick-breaking 差 2-3 dB
- 论文 DMDE 使用直接位置编码（与我们不同），且 θ 步长差异影响 PSLL 计算
