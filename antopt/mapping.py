"""线性映射器 (LM)：将无界优化变量映射为满足间距/孔径约束的线阵位置。

对应 C++ MatrixMapping::LM，扩展支持奇数元对称阵列。
"""

from typing import Optional
import numpy as np

# sigmoid 钳位阈值：|x| > CLIP 时直接取 0 或 1，避免 exp 溢出
_SIGMOID_CLIP = 20.0


class LMMapper:
    """线性映射器。

    接受无界实数变量 x ∈ ℝ^D，通过 sigmoid 映射为 (0,1) 比例，
    再分配剩余长度得到实际阵元坐标（以波长 λ₀ 为单位）。
    保证 dmin ≤ 间距 ≤ dmax 且位于孔径 [-L/2, L/2] 内。

    变量维度 D 由阵列类型决定：
      - 非对称:           Ne       (- 2 若固定孔径)
      - 对称, 偶/奇 Ne:   halfNe   (- n_fixed)
         halfNe = 右半侧含中心位置数, n_fixed = 固定位置数
    """

    def __init__(
        self,
        Ne: int,
        L: float,
        dmin: float,
        dmax: Optional[float] = None,
        is_symmetric: bool = False,
        is_fixed_aperture: bool = False,
    ):
        """
        Args:
            Ne: 阵元总数
            L: 孔径总长度（以 λ₀ 为单位）
            dmin: 最小阵元间距（以 λ₀ 为单位）
            dmax: 最大阵元间距，None=无上限
            is_symmetric: 是否关于原点对称
            is_fixed_aperture: 是否固定孔径（两端阵元钉在 ±L/2）
        """
        # 已废弃：这两种模式的最后一个阵元无法满足间距要求
        if dmax is not None:
            raise NotImplementedError("dmax 已废弃，最后一个阵元无法满足间距要求")
        if is_fixed_aperture:
            raise NotImplementedError(
                "is_fixed_aperture 已废弃，最后一个阵元无法满足间距要求"
            )
        if Ne <= 0:
            raise ValueError(f"Ne 必须为正整数, 实际 {Ne}")
        if L <= 0:
            raise ValueError(f"L 必须大于 0, 实际 {L}")
        if is_symmetric and Ne <= 1:
            raise ValueError(f"对称阵列 Ne 必须 ≥ 2, 实际 {Ne}")
        if dmin <= 0:
            raise ValueError(f"dmin 必须为正数, 实际 {dmin}")
        if dmin * (Ne - 1) > L:
            raise ValueError(f"dmin * (Ne-1) = {dmin * (Ne - 1)} > L = {L}")

        self._Ne = Ne
        self._L = L
        self._dmin = dmin
        self._is_symmetric = is_symmetric

        # 预计算半侧参数
        if is_symmetric:
            self._has_center = (Ne % 2 == 1)
            self._halfNe = (Ne + 1) // 2 if self._has_center else Ne // 2
            self._n_fixed = 1 if self._has_center else 0  # 仅奇数中心固定
        else:
            self._has_center = False
            self._halfNe = Ne
            self._n_fixed = 0

    # -- 属性 --
    @property
    def Ne(self) -> int:
        return self._Ne

    @property
    def L(self) -> float:
        return self._L

    @property
    def dmin(self) -> float:
        return self._dmin

    @property
    def is_symmetric(self) -> bool:
        return self._is_symmetric

    @property
    def has_center(self) -> bool:
        """对称阵列是否有中心阵元（Ne 奇数）。"""
        return self._has_center

    @property
    def n_vars(self) -> int:
        """优化变量维度 D = halfNe - n_fixed。"""
        return self._halfNe - self._n_fixed

    # -- 核心映射 --
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """sigmoid: ℝ → (0, 1), |x| > CLIP 时钳位到 0 或 1。"""
        x_clipped = np.clip(x, -_SIGMOID_CLIP, _SIGMOID_CLIP)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def synthesize(self, opt_vector: np.ndarray) -> np.ndarray:
        """将无界优化变量映射为阵元位置。

        Args:
            opt_vector: 优化变量, shape (n_vars,), 任意实数值

        Returns:
            阵元位置数组, shape (Ne,), 已排序, 以 λ₀ 为单位
        """
        opt_vector = np.asarray(opt_vector, dtype=float)
        if len(opt_vector) != self.n_vars:
            raise ValueError(
                f"opt_vector 长度 {len(opt_vector)} != n_vars {self.n_vars}"
            )

        v = self._sigmoid(opt_vector)
        halfL = self._L / 2.0

        if self._is_symmetric:
            return self._synthesize_symmetric(v, halfL)
        else:
            return self._synthesize_asymmetric(v, halfL)

    # ----------------------------------------------------------------
    #  对称
    # ----------------------------------------------------------------
    def _synthesize_symmetric(self, v: np.ndarray, halfL: float) -> np.ndarray:
        dmin = self._dmin
        halfNe = self._halfNe
        hpos = np.empty(halfNe)

        if self._has_center:
            hpos[0] = 0.0
            length_remain = halfL - (halfNe - 1) * dmin
            for i in range(halfNe - 1):
                hpos[i + 1] = hpos[i] + dmin + length_remain * v[i]
                length_remain *= (1.0 - v[i])
        else:
            hpos[0] = dmin / 2.0
            length_remain = halfL - (halfNe - 0.5) * dmin
            hpos[0] += length_remain * v[0]
            length_remain *= (1.0 - v[0])
            for i in range(1, halfNe):
                hpos[i] = hpos[i - 1] + dmin + length_remain * v[i]
                length_remain *= (1.0 - v[i])

        pos = np.empty(self._Ne)
        if self._has_center:
            pos[:halfNe - 1] = -hpos[1:][::-1]
            pos[halfNe - 1:] = hpos
        else:
            pos[:halfNe] = -hpos[::-1]
            pos[halfNe:] = hpos
        return pos

    # ----------------------------------------------------------------
    #  非对称
    # ----------------------------------------------------------------
    def _synthesize_asymmetric(self, v: np.ndarray, halfL: float) -> np.ndarray:
        dmin = self._dmin
        pos = np.empty(self._Ne)
        pos[0] = -halfL
        length_remain = self._L - (self._Ne - 1) * dmin

        for i in range(self._Ne - 1):
            pos[i + 1] = pos[i] + dmin + length_remain * v[i]
            length_remain *= (1.0 - v[i])
        # 剩余孔径用于整体偏移
        rl = self._L - (pos[self._Ne - 1] - pos[0])
        pos += rl * v[self._Ne - 1]

        return pos
