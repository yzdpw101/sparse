"""稀布阵列优化器 — 基于 pymoo CMA-ES。

对应 C++ Main.cpp 的 minimizePSLL + getYc 流程。
"""

from typing import Optional
import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.termination import get_termination

from .mapping import LMMapper
from .pattern import Pattern
from .analysis import get_psll

TWO_PI = 2 * np.pi


class SparseArrayProblem(Problem):
    """稀布阵列优化问题 —— 包装为 pymoo Problem。"""

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
        """
        Args:
            mapper: LM 映射器
            pattern: Pattern 方向图计算器
            optimize_phase: 是否优化相位
            optimize_amplitude: 是否优化幅度
            init_positions: 导入的初始位置 (Ne,)，不可与 optimize_pos 共用
            init_phases_deg: 导入的初始相位 (Ne,)，不可与 optimize_phase 共用
            init_amplitudes: 导入的初始幅度 (Ne,)，不可与 optimize_amplitude 共用
            amplitude_bounds: (lower, upper) 幅度范围
            target_hpbw: 目标半功率波束宽度（度），≤HPBW 时不惩罚
            mainlobe_region: 主瓣排除区域，传给 get_psll
        """
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
        n_vars = self.n_pos + self.n_phase + self.n_amp

        # 边界：位置用大范围（sigmoid 钳位在 ±20），相位 [0, 2π]，幅度 [lb, ub]
        xl = np.empty(n_vars)
        xu = np.empty(n_vars)
        xl[:self.n_pos] = -50.0
        xu[:self.n_pos] = 50.0
        xl[self.n_pos:self.n_pos + self.n_phase] = 0.0
        xu[self.n_pos:self.n_pos + self.n_phase] = TWO_PI
        xl[-self.n_amp or n_vars:] = amplitude_bounds[0]
        xu[-self.n_amp or n_vars:] = amplitude_bounds[1]

        super().__init__(n_var=n_vars, n_obj=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        n_pop = len(x)
        f = np.zeros(n_pop)

        for i in range(n_pop):
            f[i] = self._fitness(x[i])

        out["F"] = f.reshape(-1, 1)

    def _fitness(self, x: np.ndarray) -> float:
        """适应度 = PSLL + HPBW_惩罚。"""
        # 1. 提取位置
        if self.n_pos > 0:
            pos = self.mapper.synthesize(x[:self.n_pos])
        elif self.init_positions is not None:
            pos = self.init_positions
        else:
            raise RuntimeError("无位置来源")

        # 2. 构建激励
        Ne = self.mapper.Ne
        amps = np.ones(Ne)
        phases = np.zeros(Ne)

        if self.optimize_phase:
            phases = x[self.n_pos:self.n_pos + self.n_phase]
        elif self.init_phases_deg is not None:
            phases = np.deg2rad(self.init_phases_deg)

        if self.optimize_amplitude:
            amps = x[-self.n_amp:]
        elif self.init_amplitudes is not None:
            amps = self.init_amplitudes

        exc = amps * np.exp(1j * phases)

        # 3. 计算 AF
        if self.pattern.is_planar:
            af = self.pattern.planar_af(pos, np.zeros_like(pos), amps, phases)
        else:
            if self.mapper.is_symmetric:
                # synthesize 返回完整位置；linear_af_symmetric 只需半边(x>0)
                # 激励也需对应半边
                halfNe = self.mapper._halfNe
                half_pos = pos[halfNe:]      # 右侧位置 (hpos 或 hpos[1:])
                half_amps = amps[halfNe:]    # 对应激励
                half_phases = phases[halfNe:]
                af = self.pattern.linear_af_symmetric(
                    half_pos, has_center=self.mapper.has_center,
                    amplitudes=half_amps, phases=half_phases,
                )
            else:
                af = self.pattern.linear_af(pos, amps, phases)

        af_db = Pattern.to_dB(af)

        # 4. PSLL
        if self.pattern.is_planar:
            psll, _ = get_psll(
                af_db, self.pattern.theta_deg, self.pattern.phi_deg,
                mainlobe_region=self.mainlobe_region,
            )
            # 2D get_psll 返回 (pslls, coords)，取最差值
            if af_db.ndim == 2:
                valid = ~np.isinf(psll)
                psll_val = float(np.max(psll[valid])) if valid.any() else 0.0
            else:
                psll_val = float(psll)
        else:
            psll_val, _ = get_psll(
                af_db, self.pattern.theta_deg,
                mainlobe_region=(
                    self.mainlobe_region[0]
                    if isinstance(self.mainlobe_region, tuple)
                    and len(self.mainlobe_region) == 2
                    and isinstance(self.mainlobe_region[0], (int, float))
                    else self.mainlobe_region
                ),
            )

        # 5. HPBW 惩罚
        hpbw_penalty = 0.0
        if self.target_hpbw < 180.0:
            hpbw = self._compute_hpbw(af_db)
            if hpbw > self.target_hpbw:
                hpbw_penalty = (hpbw - self.target_hpbw) * 100.0

        return float(psll_val) + hpbw_penalty

    def _compute_hpbw(self, af_db: np.ndarray) -> float:
        """计算半功率波束宽度（度）。"""
        if self.pattern.is_planar:
            # 取 φ=0 切片
            j0 = np.argmin(np.abs(self.pattern.phi_deg))
            pattern_1d = af_db[:, j0]
        else:
            pattern_1d = af_db

        peak_idx = int(np.argmax(pattern_1d))
        target = -3.0

        # 左半
        left = pattern_1d[:peak_idx + 1]
        left_diff = np.abs(left - target)
        left_idx = int(np.argmin(left_diff))

        # 右半
        right = pattern_1d[peak_idx:]
        right_diff = np.abs(right - target)
        right_idx = int(np.argmin(right_diff)) + peak_idx

        return (right_idx - left_idx) * np.abs(
            self.pattern.theta_deg[1] - self.pattern.theta_deg[0]
        )

    def get_result(self, x_opt: np.ndarray) -> dict:
        """从最优变量提取结果字典。"""
        result = {}

        # 位置
        if self.n_pos > 0:
            pos = self.mapper.synthesize(x_opt[:self.n_pos])
        else:
            pos = self.init_positions
        result["positions"] = pos

        # 相位
        if self.optimize_phase:
            result["phases_deg"] = np.rad2deg(
                x_opt[self.n_pos:self.n_pos + self.n_phase]
            )
        elif self.init_phases_deg is not None:
            result["phases_deg"] = self.init_phases_deg
        else:
            result["phases_deg"] = np.zeros_like(pos)

        # 幅度
        if self.optimize_amplitude:
            result["amplitudes"] = x_opt[-self.n_amp:]
        elif self.init_amplitudes is not None:
            result["amplitudes"] = self.init_amplitudes
        else:
            result["amplitudes"] = np.ones_like(pos)

        return result


def run_optimization(
    mapper: LMMapper,
    pattern: Pattern,
    pop_size: int = 50,
    max_iter: int = 10000,
    sigma: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """运行 CMA-ES 稀疏阵列优化。

    Args:
        mapper: LM 映射器
        pattern: Pattern 方向图计算器
        pop_size: 种群大小
        max_iter: 最大迭代次数
        sigma: 初始步长
        seed: 随机种子，0 = 随机
        verbose: 是否打印进度
        **kwargs: 传给 SparseArrayProblem

    Returns:
        dict: {"X": 最优变量, "F": 最优适应度, "result": 结果字典}
    """
    if seed == 0:
        seed = np.random.randint(1, 2**31)

    problem = SparseArrayProblem(mapper, pattern, **kwargs)

    algorithm = CMAES(
        x0=None,
        sigma=sigma,
        pop_size=pop_size,
    )

    termination = get_termination("n_eval", max_iter * pop_size)

    algorithm.setup(problem, seed=seed, verbose=verbose)
    algorithm.run()

    res = algorithm.result()

    x_opt = res.X
    f_opt = float(res.F[0]) if hasattr(res.F, '__len__') else float(res.F)

    return {
        "X": x_opt,
        "F": f_opt,
        "seed": seed,
        "result": problem.get_result(x_opt),
    }
