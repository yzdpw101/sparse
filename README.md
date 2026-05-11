# sparsearray

稀布/稀疏阵列天线优化库 — Python 实现。

## 功能

- **方向图计算**: 阵因子 × 单元因子，线阵/平面阵，对称/非对称，多频/多角度
- **稀布优化**: 自研 CMA-ES（有界 [0,1] + stick-breaking 映射），支持 DE / GWO
- **幅相联合优化**: mode 三位编码控制位置/相位/幅度，支持 JSON 导入初始配置
- **单元方向图**: HFSS 导出多频 CSV 导入
- **断点续跑**: 中断后自动恢复，跑完清理缓存

## 安装

```bash
pip install numpy scipy matplotlib cma
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
├── antopt/                     # 核心库
│   ├── optimizer.py            # 自研 CMA-ES + DE + GWO (minimize API)
│   ├── mapping.py              # Stick-breaking 线性映射器
│   ├── pattern.py              # 方向图计算
│   ├── analysis.py             # PSLL / 峰值搜索
│   ├── space_mapping.py        # 渐进空间映射 (⚠ 待修复)
│   ├── element_pattern.py      # HFSS CSV 单元方向图
│   └── utils.py
├── benchmarks/                 # 手动测试/对比脚本
├── tests/                      # pytest (84 tests)
├── examples/
│   ├── 线阵稀布幅相优化/        # 主工程
│   └── 渐进空间映射稀布线阵/    # 空间映射 (⚠ PE 收敛待修)
└── docs/                       # 优化器/映射对比报告
```

## 性能 (CMA + stick-breaking, 10000 iter)

| 配置 | PSLL |
|---|---|
| 78元 / 73λ | -22.96 dB |
| 132元 / 90.5λ | **-27.02 dB** |
| 152元 / 98.5λ | -24.41 dB |

## License

MIT
