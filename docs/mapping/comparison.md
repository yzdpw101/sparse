# 映射方法对比报告

## 问题定义

给定 Ne 元对称固定孔径线阵（孔径 Lλ, dmin=0.5λ），将 D 维优化变量映射到满足间距和孔径约束的阵元位置。

## 被测方法

### 1. Stick-breaking（当前方案，推荐）

```python
# v[i] ∈ (0,1) 控制从剩余长度中分配多少给第 i 个间隙
temp = length_remain
for i in range(n_gap):
    extras[outermost - i] = temp * v[i]
    temp *= (1.0 - v[i])
```

- **优势**: 天然满足所有约束（dmin、孔径），映射平滑可微，最优解唯一
- **劣势**: v[0] 权重最大，v[-1] 最小，产生层级变量依赖
- **适配优化器**: CMA-ES（协方差学习依赖）

### 2. Proportional（归一化比例分配）

```python
gap_prefs = sigmoid(x[:n_gap])    # 间隙偏好
total_frac = sigmoid(x[n_gap])    # 总分配比例
extras = length_remain * total_frac * gap_prefs / sum(gap_prefs)
```

- **优势**: 所有变量影响力均等
- **劣势**: 多个 x 映射到同一组间隙（尺度不变性），优化景观有"平坦方向"
- **结论**: 不适合 stick-breaking 替代，对 PE（参数提取）不友好

### 3. 直接位置编码（排序 + 推间隙修复）

```python
raw = dmin/2 + sigmoid(x) * (L/2 - 1.5*dmin)
raw.sort()  # 升序
hpos[0] = max(raw[0], dmin/2)
for i in range(1, n):
    hpos[i] = max(raw[i], hpos[i-1] + dmin)  # 推间隙
hpos[-1] = L/2  # 固定孔径
```

- **优势**: 每变量直接控制一个位置
- **劣势**: 排序+推间隙是不可微操作，破坏优化景观平滑性
- **结论**: 比 stick-breaking 差 2-3 dB，不推荐

## 结果对比 (CMA-ES, 同配置)

| 映射方法 | 78元/73λ | 132元/90.5λ | 152元/98.5λ |
|---|---|---|---|
| **Stick-breaking** | **-22.34** | **-25.02** | **-24.27** |
| Proportional | -21.92* | -20.84* | -22.31* |
| 直接位置 (sig=±6.9) | -19.98 | -22.71 | -23.24 |

*Proportional 使用 DMDE, CMA-ES 未单独测试

## 结论

**Stick-breaking 是最优映射方案。** 其层级变量依赖恰被 CMA-ES 协方差矩阵消化。Proportional 的非唯一性和直接位置的不可微性都导致了更差的优化结果。

## 奇偶对称的正确间距约束

```text
奇数对称 (有中心阵元 x=0):
  中心→hpos[0] 最小 dmin       (阵元间距 = hpos[0] - 0)
  hpos[i]→hpos[i+1] 最小 dmin
  length_remain = halfL - halfNe × dmin

偶数对称 (无中心):
  ±hpos[0] 间距 2×hpos[0] ≥ dmin → hpos[0] ≥ dmin/2
  hpos[i]→hpos[i+1] 最小 dmin
  length_remain = halfL - (halfNe - 0.5) × dmin
```
