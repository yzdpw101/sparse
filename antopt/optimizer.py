"""稀布阵列优化器 — 支持多种无约束优化算法。

所有优化变量 ∈ ℝ（无约束），通过 sigmoid 映射到目标范围。
对应 C++ Main.cpp 的 minimizePSLL + getYc 流程。

mode 编码（3 位，每位 0/1/2，位置必为 1 或 2）：
  百位 = 位置: 1=优化, 2=导入
  十位 = 相位: 0=等相位(全0), 1=优化, 2=导入
  个位 = 幅度: 0=等幅度(全1), 1=优化, 2=导入
  三位中至少有一位为 1（否则无优化变量）
  14 种: 100-102,110-112,120-122, 201,210-212,221

可用算法:
  - cma: pycma CMA-ES（参考实现）
  - ngopt: nevergrad NGOpt
  - cma_ng: nevergrad CMA
  - de: nevergrad DE
  - pso: nevergrad PSO
"""

import os
from typing import Optional
from multiprocessing.pool import ThreadPool
import numpy as np
import cma
import nevergrad as ng

from .mapping import LMMapper
from .pattern import Pattern
from .analysis import get_psll

_NG_OPTIMIZERS = {
    "ngopt": ng.optimizers.NGOpt,
    "cma_ng": ng.optimizers.CMA,
    "de": ng.optimizers.DE,
    "pso": ng.optimizers.PSO,
    "twopointsde": ng.optimizers.TwoPointsDE,
    "random": ng.optimizers.RandomSearch,
}

TWO_PI = 2 * np.pi
_SIGMOID_CLIP = 20.0


class SparseArrayProblem:
    """稀布阵列优化问题。"""

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        mode: int = 100,
        init_positions: Optional[np.ndarray] = None,
        init_phases_deg: Optional[np.ndarray] = None,
        init_amplitudes: Optional[np.ndarray] = None,
        amplitude_bounds: tuple[float, float] = (0.0, 1.0),
        target_hpbw: float = 180.0,
        mainlobe_region: Optional[tuple] = None,
    ):
        # 解码 mode: 百位=位置, 十位=相位, 个位=幅度
        p_mode = (mode // 100) % 10
        h_mode = (mode // 10) % 10
        a_mode = mode % 10

        if p_mode not in (1, 2):
            raise ValueError(f"mode={mode}: 百位(位置)必须是 1 或 2")
        if h_mode not in (0, 1, 2) or a_mode not in (0, 1, 2):
            raise ValueError(f"mode={mode}: 十位/个位必须是 0/1/2")
        if 1 not in (p_mode, h_mode, a_mode):
            raise ValueError(f"mode={mode}: 至少一位为 1")

        self.mapper = mapper
        self.pattern = pattern
        self.mode = mode
        self._p_mode = p_mode
        self._h_mode = h_mode
        self._a_mode = a_mode
        self.init_positions = init_positions
        self.init_phases_deg = init_phases_deg
        self.init_amplitudes = init_amplitudes
        self.amplitude_lower, self.amplitude_upper = amplitude_bounds
        self.target_hpbw = target_hpbw
        self.mainlobe_region = mainlobe_region

        # 变量维度
        self.n_pos = mapper.n_vars if p_mode == 1 else 0
        self.n_phase = mapper.Ne if h_mode == 1 else 0
        self.n_amp = mapper.Ne if a_mode == 1 else 0
        self.n_vars = self.n_pos + self.n_phase + self.n_amp

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -_SIGMOID_CLIP, _SIGMOID_CLIP)
        return 1.0 / (1.0 + np.exp(-x))

    def fitness(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        Ne = self.mapper.Ne

        # 1. 位置
        if self._p_mode == 1:
            pos = self.mapper.synthesize(x[:self.n_pos])
        else:  # _p_mode == 2
            pos = self.init_positions

        # 2. 相位
        if self._h_mode == 1:
            frac = self._sigmoid(x[self.n_pos:self.n_pos + self.n_phase])
            phases = frac * TWO_PI
        elif self._h_mode == 2:
            phases = np.deg2rad(self.init_phases_deg)
        else:
            phases = np.zeros(Ne)

        # 3. 幅度
        if self._a_mode == 1:
            frac = self._sigmoid(x[-self.n_amp:])
            amps = self.amplitude_lower + (self.amplitude_upper - self.amplitude_lower) * frac
        elif self._a_mode == 2:
            amps = self.init_amplitudes
        else:
            amps = np.ones(Ne)

        # 4. 计算 AF
        if self.pattern.is_planar:
            af = self.pattern.planar_af(pos, np.zeros_like(pos), amps, phases)
        elif self.mapper.is_symmetric:
            halfNe = self.mapper._halfNe
            af = self.pattern.linear_af_symmetric(
                pos[halfNe:], has_center=self.mapper.has_center,
                amplitudes=amps[halfNe:], phases=phases[halfNe:],
            )
        else:
            af = self.pattern.linear_af(pos, amps, phases)

        af_db = Pattern.to_dB(af)

        # 5. PSLL
        if self.pattern.is_planar:
            pslls, _ = get_psll(af_db, self.pattern.theta_deg, self.pattern.phi_deg,
                                mainlobe_region=self.mainlobe_region)
            if af_db.ndim == 2:
                valid = ~np.isinf(pslls)
                psll_val = float(np.max(pslls[valid])) if valid.any() else 0.0
            else:
                psll_val = float(pslls)
        else:
            mr_1d = None
            if (isinstance(self.mainlobe_region, tuple) and len(self.mainlobe_region) == 2
                    and isinstance(self.mainlobe_region[0], (int, float, np.floating))):
                mr_1d = self.mainlobe_region
            psll_val, _ = get_psll(af_db, self.pattern.theta_deg, mainlobe_region=mr_1d)

        # 6. HPBW 惩罚
        penalty = 0.0
        if self.target_hpbw < 180.0:
            hpbw = self._compute_hpbw(af_db)
            if hpbw > self.target_hpbw:
                penalty = (hpbw - self.target_hpbw) * 100.0

        return float(psll_val) + penalty

    def _compute_hpbw(self, af_db: np.ndarray) -> float:
        if self.pattern.is_planar:
            j0 = np.argmin(np.abs(self.pattern.phi_deg))
            pattern_1d = af_db[:, j0]
        else:
            pattern_1d = af_db
        peak_idx = int(np.argmax(pattern_1d))
        left = pattern_1d[:peak_idx + 1]
        right = pattern_1d[peak_idx:]
        left_idx = int(np.argmin(np.abs(left - (-3.0))))
        right_idx = int(np.argmin(np.abs(right - (-3.0)))) + peak_idx
        dtheta = abs(self.pattern.theta_deg[1] - self.pattern.theta_deg[0])
        return (right_idx - left_idx) * dtheta

    def get_result(self, x_opt: np.ndarray) -> dict:
        x = np.asarray(x_opt, dtype=float)
        r = {}
        if self._p_mode == 1:
            r["positions"] = self.mapper.synthesize(x[:self.n_pos])
        else:  # _p_mode == 2
            r["positions"] = self.init_positions

        if self._h_mode == 1:
            frac = self._sigmoid(x[self.n_pos:self.n_pos + self.n_phase])
            r["phases_deg"] = np.rad2deg(frac * TWO_PI)
        elif self._h_mode == 2:
            r["phases_deg"] = self.init_phases_deg
        else:
            r["phases_deg"] = np.zeros_like(r["positions"])

        if self._a_mode == 1:
            frac = self._sigmoid(x[-self.n_amp:])
            r["amplitudes"] = self.amplitude_lower + (self.amplitude_upper - self.amplitude_lower) * frac
        elif self._a_mode == 2:
            r["amplitudes"] = self.init_amplitudes
        else:
            r["amplitudes"] = np.ones_like(r["positions"])

        return r


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
    n_jobs: int = 1,
    **kwargs,
) -> dict:
    if seed == 0:
        seed = np.random.randint(1, 2**31)

    problem = SparseArrayProblem(mapper, pattern, **kwargs)
    n_vars = problem.n_vars

    if method == "cma":
        return _run_cma(problem, n_vars, x0, sigma0, pop_size, max_iter, seed, verbose, n_jobs)
    elif method in _NG_OPTIMIZERS:
        return _run_nevergrad(problem, n_vars, method, pop_size, max_iter, seed, verbose)
    else:
        raise ValueError(f"未知 method: {method}, 可用: cma, {list(_NG_OPTIMIZERS)}")


def _run_cma(problem, n_vars, x0, sigma0, pop_size, max_iter, seed, verbose, n_jobs):
    if x0 is None:
        x0 = np.zeros(n_vars)
    if n_jobs < 0:
        n_jobs = os.cpu_count() or 4
    if pop_size is None:
        pop_size = 4 + int(3 * np.log(n_vars))

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "seed": seed, "maxfevals": max_iter * pop_size,
        "verbose": 0 if verbose else -9,
        "CMA_diagonal": n_vars > 30, "popsize": pop_size,
    })
    pool = ThreadPool(n_jobs) if n_jobs > 1 else None
    try:
        while not es.stop():
            X = es.ask()
            fits = pool.map(problem.fitness, X) if pool else [problem.fitness(xi) for xi in X]
            es.tell(X, fits)
        result = es.result
    finally:
        if pool:
            pool.terminate()
    x_opt, f_opt = result[0], result[1]
    return {"x": x_opt, "f": f_opt, "seed": seed, "method": "cma",
            "result": problem.get_result(x_opt)}


def _run_nevergrad(problem, n_vars, method, pop_size, max_iter, seed, verbose):
    budget = max_iter * (pop_size or (4 + int(3 * np.log(n_vars))))
    opt = _NG_OPTIMIZERS[method](parametrization=ng.p.Array(shape=(n_vars,)),
                                  budget=budget, num_workers=1)
    res = opt.minimize(lambda x: problem.fitness(np.asarray(x, dtype=float)),
                        verbosity=int(verbose))
    x_opt = np.array(res.args[0], dtype=float)
    f_opt = res.loss if res.loss is not None else problem.fitness(x_opt)
    return {"x": x_opt, "f": f_opt, "seed": seed, "method": method,
            "result": problem.get_result(x_opt)}
