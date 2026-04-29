# sparsearray - 稀布阵列天线优化库

## 项目概述

Python 实现的稀布/稀疏阵列天线优化库，方向图乘积定理（阵因子 × 单元因子）。

当前阶段：Phase 1 核心计算已完成，Phase 2 优化基础设施进行中。

## 当前状态 (Phase 1 完成)

- [x] `core/geometry.py` — Element、ArrayGeometry、LinearArray、PlanarArray、UniformLinearArray
- [x] `core/array_factor.py` — 向量化阵因子计算 (θ-φ 空间 + UV 空间)
- [x] `core/pattern.py` — 方向图分析 (MSLL、HPBW、SLL、方向性系数)
- [x] `core/element_pattern.py` — 各向同性、cosine-q、贴片、HFSS 导入
- [x] 基础测试通过 (13 tests)
- [ ] `optimization/` — 待实现
- [ ] `optimizers/` — 待实现
- [ ] `visualization/` — 待实现
- [ ] `integration/` — 待实现

## 关键技术要点

### 阵因子计算 (array_factor.py)

- **向量化**：NumPy 广播一次性算所有方向，不用 for 循环
- **单位处理**：geometry 里 `positions` 以**波长**为单位，array_factor 内部乘以 `wavelength` 转成米再乘以波数 k (1/m)
  ```python
  pos_m = self.geometry.positions[active] * self.geometry.wavelength
  k = 2 * np.pi / wavelength  # 波数 (1/m)
  ```
- 线性阵：`AF(θ) = Σ A_n · exp(j·k·x_n·(sinθ - sinθ₀))`
- 平面阵：`AF(θ,φ) = Σ A_n · exp(j·k·(x_n·u + y_n·v))`，u=sinθ·cosφ - sinθ₀·cosφ₀

### 方向图分析 (pattern.py)

- MSLL 检测：从主瓣向外搜索局部极小值确定主瓣边界，再扫描副瓣区域找所有峰值取最大
- HPBW 检测：在 -3dB 处线性插值提高精度
- 与 C++ 版本的区别：C++ 用 `extrema1D` 取第一副瓣，Python 取所有副瓣最大值
- **Pattern 只负责指标提取，fitness 中的惩罚逻辑不在该类中**

### C++ → Python 映射

| C++ | Python | 说明 |
|---|---|---|
| `ElementBlock` (Position + ElementProperty) | `Element` dataclass + `ArrayGeometry` | 扁平化 ndarray 替代 |
| `Pattern::Linear::calcAF(deltaSinTheta, wlpos)` | `ArrayFactor.compute_af(theta, phi)` | 改用 θ 角输入，内部转 sinθ |
| `Pattern::Planar::calcUVAF(deltaU, deltaV)` | `ArrayFactor.compute_uv_af(u, v)` | 保留 UV 空间计算 |
| `Pattern::calcUVPattern` 后处理 | `Pattern` 类 | 方向图分析和指标提取 |
| `extrema1D` → extrema[1] 取 PSLL | `get_msll()` 扫描所有副瓣峰值 | Python 取全局最大值 |
| LM mapper 位置编码 | 待实现 → `constraints.py` 或 `continuous.py` | Phase 2 |
| `FitFuncType = function<double(VectorXd)>` | `Callable[[ndarray], float]` | Python 原生支持 |
| CRTP + fluent setter | 关键字参数 + dataclass | Python 更简洁 |

### Config 策略

Python 用 `json.load()` 读配置非常轻量，不需要像 C++ 那样封装。推荐：
- 一个 `load_config()` 函数
- 一个 `ExperimentConfig` dataclass 存参数字段
- 不搞完整的大封装类

## 开发约定

- Python 3.10+, NumPy, SciPy, Matplotlib 为核心依赖
- 尽量用成熟库（pymoo、cma、scipy.optimize），不重复造轮子
- 测试用 pytest，conftest.py 放共享 fixtures
- 一个功能一个 PR style 增量开发
- 代码注释用中文，但标识符和 API 用英文
- 不改动已有的工作代码，新增功能时注意向后兼容

## 架构原则

- **优先解耦**：底层模块（ArrayFactor、Pattern）保持 stateless 纯计算；中间层只有在需要封装状态 + callable 时才用类，且职责单一；顶层不封装，直接写脚本/notebook 组装流程。
- **不搞上帝类**：不做大而全的 Workflow/Context 类，传参用 dataclass 或配置字典。
- **能写成独立函数就不加类**，有确定状态需要管理时才考虑用类。

## 测试

```bash
python -m pytest tests/ -v
```

当前 13 个测试全部通过，覆盖：
- geometry: 构造、对称性、批量设置、activate_subset
- array_factor: 输出形状、法向峰值、MSLL 理论值 (-12.97dB)、HPBW (10.19°)、扫描、单阵元退化
