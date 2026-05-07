"""稀布阵列优化器 — 支持多种无约束优化算法。

所有优化变量 ∈ ℝ（无约束），通过 sigmoid 映射到目标范围。
对应 C++ Main.cpp 的 minimizePSLL + getYc 流程。

可用算法:
  - cma: pycma CMA-ES（参考实现）
  - ngopt: nevergrad NGOpt（自动选优）
  - cma_ng: nevergrad CMA
  - de: nevergrad 差分进化
  - pso: nevergrad 粒子群
  - twopointsde: nevergrad TwoPointsDE
"""

from typing import Optional
import numpy as np
import cma
import nevergrad as ng

from .mapping import LMMapper
from .pattern import Pattern
from .analysis import get_psll

# nevergrad 算法名 → 优化器类
_NG_OPTIMIZERS = {
    "ngopt": ng.optimizers.NGOpt,
    "cma_ng": ng.optimizers.CMA,
    "de": ng.optimizers.DE,
    "pso": ng.optimizers.PSO,
    "twopointsde": ng.optimizers.TwoPointsDE,
    "random": ng.optimizers.RandomSearch,
}

TWO_PI = 2 * np.pi
_SIGMOID_CLIP = 20.0  # 与 LMMapper 一致


class SparseArrayProblem:
    """稀布阵列优化问题。

    所有优化变量 ∈ ℝ（无约束）：
      - 位置: sigmoid → (0,1) → LM 映射为物理位置
      - 相位: sigmoid → (0, 2π)
      - 幅度: sigmoid → [amp_lower, amp_upper]
    """

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        optimize_phase: bool = False,
        optimize_amplitude: bool = False,
        init_positions: Optional[np.ndarray] = None,
        init_phases_deg: Optional[np.ndarray] = None,
        init_amplitudes: Optional[np.ndarray] = None,
        amplitude_bounds: tuple[float, float] = (0.0, 1.0),
        target_hpbw: float = 180.0,
        mainlobe_region: Optional[tuple] = None,
    ):
        self.mapper = mapper
        self.pattern = pattern
        self.optimize_phase = optimize_phase
        self.optimize_amplitude = optimize_amplitude
        self.init_positions = init_positions
        self.init_phases_deg = init_phases_deg
        self.init_amplitudes = init_amplitudes
        self.amplitude_lower, self.amplitude_upper = amplitude_bounds
        self.target_hpbw = target_hpbw
        self.mainlobe_region = mainlobe_region

        self.n_pos = mapper.n_vars
        self.n_phase = mapper.Ne if optimize_phase else 0
        self.n_amp = mapper.Ne if optimize_amplitude else 0
        self.n_vars = self.n_pos + self.n_phase + self.n_amp

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """sigmoid: ℝ → (0, 1)。"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -_SIGMOID_CLIP, _SIGMOID_CLIP)))

    def fitness(self, x: np.ndarray) -> float:
        """适应度 = PSLL + HPBW_惩罚。cma.fmin 要求 f(x) 返回标量。"""
        x = np.asarray(x, dtype=float)

        # 1. 位置：直传 synthesize（内部已有 sigmoid）
        if self.n_pos > 0:
            pos = self.mapper.synthesize(x[:self.n_pos])
        elif self.init_positions is not None:
            pos = self.init_positions
        else:
            raise RuntimeError("无位置来源")

        # 2. 相位：sigmoid → [0, 2π]
        Ne = self.mapper.Ne
        if self.optimize_phase:
            phase_frac = self._sigmoid(
                x[self.n_pos:self.n_pos + self.n_phase]
            )
            phases = phase_frac * TWO_PI
        elif self.init_phases_deg is not None:
            phases = np.deg2rad(self.init_phases_deg)
        else:
            phases = np.zeros(Ne)

        # 3. 幅度：sigmoid → [amp_lb, amp_ub]
        if self.optimize_amplitude:
            amp_frac = self._sigmoid(x[-self.n_amp:])
            amps = self.amplitude_lower + (
                self.amplitude_upper - self.amplitude_lower
            ) * amp_frac
        elif self.init_amplitudes is not None:
            amps = self.init_amplitudes
        else:
            amps = np.ones(Ne)

        # 4. 计算 AF
        if self.pattern.is_planar:
            af = self.pattern.planar_af(pos, np.zeros_like(pos), amps, phases)
        else:
            if self.mapper.is_symmetric:
                halfNe = self.mapper._halfNe
                half_pos = pos[halfNe:]
                half_amps = amps[halfNe:]
                half_phases = phases[halfNe:]
                af = self.pattern.linear_af_symmetric(
                    half_pos, has_center=self.mapper.has_center,
                    amplitudes=half_amps, phases=half_phases,
                )
            else:
                af = self.pattern.linear_af(pos, amps, phases)

        af_db = Pattern.to_dB(af)

        # 5. PSLL
        if self.pattern.is_planar:
            pslls, _ = get_psll(
                af_db, self.pattern.theta_deg, self.pattern.phi_deg,
                mainlobe_region=self.mainlobe_region,
            )
            if af_db.ndim == 2:
                valid = ~np.isinf(pslls)
                psll_val = float(np.max(pslls[valid])) if valid.any() else 0.0
            else:
                psll_val = float(pslls)
        else:
            # 线阵: 解析 mainlobe_region
            mr_1d = None
            if self.mainlobe_region is not None:
                if (isinstance(self.mainlobe_region, tuple) and
                    len(self.mainlobe_region) == 2 and
                    isinstance(self.mainlobe_region[0], (int, float, np.floating))):
                    mr_1d = self.mainlobe_region
            psll_val, _ = get_psll(
                af_db, self.pattern.theta_deg, mainlobe_region=mr_1d,
            )

        # 6. HPBW 惩罚
        hpbw_penalty = 0.0
        if self.target_hpbw < 180.0:
            hpbw = self._compute_hpbw(af_db)
            if hpbw > self.target_hpbw:
                hpbw_penalty = (hpbw - self.target_hpbw) * 100.0

        return float(psll_val) + hpbw_penalty

    def _compute_hpbw(self, af_db: np.ndarray) -> float:
        """计算半功率波束宽度（度）。"""
        if self.pattern.is_planar:
            j0 = np.argmin(np.abs(self.pattern.phi_deg))
            pattern_1d = af_db[:, j0]
        else:
            pattern_1d = af_db

        peak_idx = int(np.argmax(pattern_1d))
        target = -3.0
        left = pattern_1d[:peak_idx + 1]
        left_idx = int(np.argmin(np.abs(left - target)))
        right = pattern_1d[peak_idx:]
        right_idx = int(np.argmin(np.abs(right - target))) + peak_idx
        return (right_idx - left_idx) * np.abs(
            self.pattern.theta_deg[1] - self.pattern.theta_deg[0]
        )

    def get_result(self, x_opt: np.ndarray) -> dict:
        """从最优无界变量提取结果字典。"""
        x = np.asarray(x_opt, dtype=float)
        result = {}

        # 位置: 直传 synthesize（内部已有 sigmoid）
        if self.n_pos > 0:
            pos = self.mapper.synthesize(x[:self.n_pos])
        else:
            pos = self.init_positions
        result["positions"] = pos

        # 相位: sigmoid → deg
        if self.optimize_phase:
            phase_frac = self._sigmoid(
                x[self.n_pos:self.n_pos + self.n_phase]
            )
            result["phases_deg"] = np.rad2deg(phase_frac * TWO_PI)
        elif self.init_phases_deg is not None:
            result["phases_deg"] = self.init_phases_deg
        else:
            result["phases_deg"] = np.zeros_like(pos)

        # 幅度: sigmoid → [lb, ub]
        if self.optimize_amplitude:
            amp_frac = self._sigmoid(x[-self.n_amp:])
            result["amplitudes"] = self.amplitude_lower + (
                self.amplitude_upper - self.amplitude_lower
            ) * amp_frac
        elif self.init_amplitudes is not None:
            result["amplitudes"] = self.init_amplitudes
        else:
            result["amplitudes"] = np.ones_like(pos)

        return result


def run_optimization(
    mapper: LMMapper,
    pattern: Pattern,
    x0: Optional[np.ndarray] = None,
    sigma0: float = 1.0,
    pop_size: Optional[int] = None,
    max_iter: int = 10000,
    seed: int = 0,
    verbose: bool = True,
    method: str = "cma",
    **kwargs,
) -> dict:
    """运行稀疏阵列优化。

    Args:
        mapper: LM 映射器
        pattern: Pattern 方向图计算器
        x0: 初始解 (ℝ^n)，None = 全零
        sigma0: 初始标准差（仅 method="cma"）
        pop_size: 种群大小，None = 自适应
        max_iter: 最大迭代次数
        seed: 随机种子，0 = 随机
        verbose: 打印进度
        method: 算法选择
            - "cma": pycma CMA-ES（参考实现）
            - "ngopt": nevergrad 自动选优
            - "cma_ng": nevergrad CMA
            - "de": nevergrad 差分进化
            - "pso": nevergrad 粒子群
            - "twopointsde": nevergrad TwoPointsDE
        **kwargs: 传给 SparseArrayProblem

    Returns:
        dict: {"x": 最优变量, "f": 最优适应度, "result": 结果字典, "method": 算法名}
    """
    if seed == 0:
        seed = np.random.randint(1, 2**31)

    problem = SparseArrayProblem(mapper, pattern, **kwargs)
    n_vars = problem.n_vars

    if method == "cma":
        return _run_cma(problem, n_vars, x0, sigma0, pop_size, max_iter, seed, verbose)
    elif method in _NG_OPTIMIZERS:
        return _run_nevergrad(problem, n_vars, method, pop_size, max_iter, seed, verbose)
    else:
        raise ValueError(f"未知 method: {method}, 可用: cma, {list(_NG_OPTIMIZERS)}")


def _run_cma(problem, n_vars, x0, sigma0, pop_size, max_iter, seed, verbose):
    if x0 is None:
        x0 = np.zeros(n_vars)

    cma_verbose = 0 if verbose else -9
    opts = {
        "seed": seed,
        "maxfevals": max_iter * (pop_size or (4 + int(3 * np.log(n_vars)))),
        "verbose": cma_verbose,
        "CMA_diagonal": n_vars > 30,
    }
    if pop_size is not None:
        opts["popsize"] = pop_size

    res = cma.fmin(problem.fitness, x0, sigma0, options=opts)
    x_opt, f_opt = res[0], res[1]
    return {"x": x_opt, "f": f_opt, "seed": seed, "method": "cma",
            "result": problem.get_result(x_opt)}


def _run_nevergrad(problem, n_vars, method, pop_size, max_iter, seed, verbose):
    budget = max_iter * (pop_size or (4 + int(3 * np.log(n_vars))))
    # 无界实数变量
    parametrization = ng.p.Array(shape=(n_vars,))

    OptimizerCls = _NG_OPTIMIZERS[method]
    optimizer = OptimizerCls(parametrization=parametrization,
                             budget=budget, num_workers=1)

    # nevergrad 直接传 np.ndarray
    def _fitness_ng(x):
        return problem.fitness(np.asarray(x, dtype=float))

    result = optimizer.minimize(_fitness_ng, verbosity=int(verbose))

    x_opt = np.array(result.args[0], dtype=float)
    f_opt = result.loss if result.loss is not None else problem.fitness(x_opt)

    return {"x": x_opt, "f": f_opt, "seed": seed, "method": method,
            "result": problem.get_result(x_opt)}
