# 最终结果汇总

## 测试条件

- 阵列: 对称固定孔径线阵, dmin=0.5λ, 无间距上限
- Theta: [-90°, 90°], theta0=0°, step=0.1°
- 优化器: 全部使用 C++ 风格 CMA-ES (`_run_cma_cpp`)
- 映射: stick-breaking, 无 sigmoid, clamp [0,1]
- MC=5, max_iter=500

## 综合对比

| 方法 | 78元/73λ | 132元/90.5λ | 152元/98.5λ |
|---|---|---|---|
| **CMA-ES (SB)** | -22.34 | **-25.02** | -24.12 |
| CMA-Chaos (SB) | -21.97 | -25.00 | **-24.27** |
| DMDE (SB) | -21.17 | -23.64 | -23.63 |
| DMDE (Prop) | -21.92 | -20.84 | -22.31 |
| GWO (SB) | -18.11 | -20.90 | -21.66 |
| 直接位置+修复 | -19.98 | -22.71 | -23.24 |
| *论文 DMDE* | *-23.01* | *-25.49* | *-24.20* |

## 与论文 DMDE 的差异分析

| 差异来源 | 估计影响 |
|---|---|
| 映射方式 (stick-breaking vs 直接位置) | 1-2 dB |
| θ 步长 (0.1° vs 未知) | 0-0.5 dB |
| 评估预算 (5 MC vs 50 MC) | 0.1-0.3 dB |
| PSLL 峰值搜索范围 (论文可能用 considerRegion) | 不可比 |

## 已发现的关键 bug

### 1. C++ PSLL 窗口截断 (已修)
`considerRegionHalfWidthDeg` 设得太小，排除了窗口外的真实副瓣峰值，导致 PSLL 被系统性低估 ~7 dB。

### 2. θ 步长不够 (待修)
90.5λ 孔径的窄波束 (~0.56°) 需要 ≤0.01° 步长才能正确采样副瓣峰值，0.1° 步长可能漏掉窄副瓣。

### 3. _halfNe 中心占位 (已修)
奇数对称的 `_halfNe` 含中心阵元（恒为 0），浪费一个 hpos 槽位。已改为 `_halfNe = Ne//2`。

### 4. 奇偶间距约束差异 (已修)
奇数: 中心→hpos[0] ≥ dmin; 偶数: 中心→hpos[0] ≥ dmin/2。

## 推荐方案

```json
{
  "optimizer": {
    "method": "cma_cpp",
    "cma_cpp": {
      "pop_size": null,
      "max_iter": 500,
      "sigma": 0.3,
      "n_jobs": 8,
      "verboseInterval": 50,
      "stopFitness": null
    }
  }
}
```

映射: `LMMapper(use_sigmoid=False)` + stick-breaking

可选: `chaotic_init=True` 对 78 元和 152 元有小幅改进
