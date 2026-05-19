"""线性映射器 (LM)：将无界优化变量映射为满足间距/孔径约束的线阵位置。

对应 C++ MatrixMapping::LM，扩展支持奇数元对称阵列。
"""

from typing import Optional
import numpy as np

# sigmoid 钳位阈值：|x| > CLIP 时直接取 0 或 1，避免 exp 溢出
_SIGMOID_CLIP = 100


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
        use_sigmoid: bool = True,
    ):
        """
        Args:
            Ne: 阵元总数
            L: 孔径总长度（以 λ₀ 为单位）
            dmin: 最小阵元间距（以 λ₀ 为单位）
            dmax: 最大阵元间距，None=无上限
            is_symmetric: 是否关于原点对称
            is_fixed_aperture: 是否固定孔径（两端阵元钉在 ±L/2）
            use_sigmoid: True=sigmoid 映射 (ℝ→(0,1)), False=clamp 到 [0,1]
        """
        if dmax is not None and is_fixed_aperture:
            raise ValueError(
                "dmax 和 is_fixed_aperture 不能同时启用"
            )
        if Ne <= 0:
            raise ValueError(f"Ne 必须为正整数, 实际 {Ne}")
        if L <= 0:
            raise ValueError(f"L 必须大于 0, 实际 {L}")
        if is_symmetric and Ne <= 1:
            raise ValueError(f"对称阵列 Ne 必须 ≥ 2, 实际 {Ne}")
        if is_symmetric and is_fixed_aperture and Ne <= 3:
            raise ValueError(f"对称固定孔径 Ne 必须 ≥ 4, 实际 {Ne}")
        if not is_symmetric and is_fixed_aperture and Ne <= 2:
            raise ValueError(f"非对称固定孔径 Ne 必须 ≥ 3, 实际 {Ne}")
        if dmin <= 0:
            raise ValueError(f"dmin 必须为正数, 实际 {dmin}")
        if dmax is not None and dmax <= dmin:
            raise ValueError(f"dmax ({dmax}) 必须大于 dmin ({dmin})")
        if dmax is not None and dmax * (Ne - 1) > L:
            raise ValueError(f"dmax * (Ne-1) = {dmax * (Ne - 1)} > L = {L}")
        if dmin * (Ne - 1) > L:
            raise ValueError(f"dmin * (Ne-1) = {dmin * (Ne - 1)} > L = {L}")

        self._Ne = Ne
        self._L = L
        self._dmin = dmin
        self._dmax = dmax  # None = 无上限
        self._is_symmetric = is_symmetric
        self._is_fixed_aperture = is_fixed_aperture
        self._use_sigmoid = use_sigmoid

        # 预计算半侧参数
        # _halfNe = 右半侧阵元数（不含中心，中心由 has_center / linear_af_symmetric 处理）
        if is_symmetric:
            self._has_center = (Ne % 2 == 1)
            self._halfNe = Ne // 2               # 右半侧只含 x>0 的阵元
            self._n_fixed = 1 if is_fixed_aperture else 0  # 仅孔径端固定
        else:
            self._has_center = False
            self._halfNe = Ne
            self._n_fixed = 2 if is_fixed_aperture else 0

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
    def dmax(self) -> float:
        return self._dmax if self._dmax is not None else np.inf

    @property
    def is_symmetric(self) -> bool:
        return self._is_symmetric

    @property
    def is_fixed_aperture(self) -> bool:
        return self._is_fixed_aperture

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
        """将优化变量映射为阵元位置。

        Args:
            opt_vector: 优化变量, shape (n_vars,)
                use_sigmoid=True  → ℝ 任意实数
                use_sigmoid=False → [0, 1] 有界

        Returns:
            阵元位置数组, shape (Ne,), 已排序, 以 λ₀ 为单位
        """
        opt_vector = np.asarray(opt_vector, dtype=float)
        if len(opt_vector) != self.n_vars:
            raise ValueError(
                f"opt_vector 长度 {len(opt_vector)} != n_vars {self.n_vars}"
            )

        if self._use_sigmoid:
            v = self._sigmoid(opt_vector)
        else:
            v = np.clip(opt_vector, 0.0, 1.0)
        halfL = self._L / 2.0

        if np.isinf(self.dmax):
            return self._synthesize_no_dmax(v, halfL)
        else:
            return self._synthesize_with_dmax(v, halfL)

    # ----------------------------------------------------------------
    #  无 dmax 上限
    # ----------------------------------------------------------------
    def _synthesize_no_dmax(self, v: np.ndarray, halfL: float) -> np.ndarray:
        dmin = self._dmin

        if self._is_symmetric:
            return self._synthesize_symmetric_no_dmax(v, halfL, dmin)
        else:
            return self._synthesize_asymmetric_no_dmax(v, halfL, dmin)

    def _synthesize_symmetric_no_dmax(
        self, v: np.ndarray, halfL: float, dmin: float
    ) -> np.ndarray:
        """对称无 dmax 映射。

        hpos 只含右半侧阵元（x > 0），不含中心。
        奇数：中心→hpos[0] 最小 dmin（中心阵元存在）
        偶数：中心→hpos[0] 最小 dmin/2（±hpos[0] 间距 = dmin）
        """
        halfNe = self._halfNe

        if self._is_fixed_aperture:
            n_gap = halfNe - 1
        else:
            n_gap = halfNe

        if self._has_center:
            length_remain = halfL - halfNe * dmin
        else:
            length_remain = halfL - dmin * (halfNe - 0.5)

        # 反向 stick-breaking: v[0]→最外侧, v[-1]→最内侧
        extras = np.empty(n_gap)
        temp = length_remain
        for i in range(n_gap):
            extras[n_gap - 1 - i] = temp * v[i]
            temp *= (1.0 - v[i])

        hpos = np.empty(halfNe)
        if self._has_center:
            hpos[0] = dmin + extras[0]               # 奇数：中心→右一 ≥ dmin
        else:
            hpos[0] = dmin / 2.0 + extras[0]         # 偶数：±右一 ≥ dmin
        for i in range(1, n_gap):
            hpos[i] = hpos[i - 1] + dmin + extras[i]

        if self._is_fixed_aperture:
            hpos[halfNe - 1] = halfL  # 孔径端钉住
        # 非固定孔径: n_gap == halfNe, hpos[halfNe-1] 已由循环设置

        # 拼接全阵
        pos = np.empty(self._Ne)
        if self._has_center:
            pos[:halfNe] = -hpos[::-1]        # 左半侧
            pos[halfNe] = 0.0                 # 中心
            pos[halfNe + 1:] = hpos           # 右半侧
        else:
            pos[:halfNe] = -hpos[::-1]        # 左半侧
            pos[halfNe:] = hpos               # 右半侧
        return pos

    def _synthesize_asymmetric_no_dmax(
        self, v: np.ndarray, halfL: float, dmin: float
    ) -> np.ndarray:
        pos = np.empty(self._Ne)
        pos[0] = -halfL
        length_remain = self._L - (self._Ne - 1) * dmin

        if self._is_fixed_aperture:
            n_v = self._Ne - 2
            for i in range(n_v):
                pos[i + 1] = pos[i] + dmin + length_remain * v[i]
                length_remain *= (1.0 - v[i])
            pos[self._Ne - 1] = halfL
        else:
            n_v = self._Ne - 1
            for i in range(n_v):
                pos[i + 1] = pos[i] + dmin + length_remain * v[i]
                length_remain *= (1.0 - v[i])
            rl = self._L - (pos[self._Ne - 1] - pos[0])
            pos += rl * v[self._Ne - 1]

        return pos

    # ----------------------------------------------------------------
    #  有 dmax 上限
    # ----------------------------------------------------------------
    def _synthesize_with_dmax(
        self, v: np.ndarray, halfL: float
    ) -> np.ndarray:
        dmin = self._dmin
        delta = self._dmax - dmin

        if self._is_symmetric:
            return self._synthesize_symmetric_with_dmax(v, halfL, dmin, delta)
        else:
            return self._synthesize_asymmetric_with_dmax(v, halfL, dmin, delta)

    def _synthesize_symmetric_with_dmax(
        self, v: np.ndarray, halfL: float, dmin: float, delta: float
    ) -> np.ndarray:
        """对称有 dmax 映射。"""
        halfNe = self._halfNe
        hpos = np.empty(halfNe)

        if self._has_center:
            hpos[0] = dmin + delta * v[0]
        else:
            hpos[0] = dmin / 2.0 + delta * v[0] / 2.0
        for i in range(1, halfNe):
            hpos[i] = hpos[i - 1] + dmin + delta * v[i]

        pos = np.empty(self._Ne)
        if self._has_center:
            pos[:halfNe] = -hpos[::-1]
            pos[halfNe] = 0.0
            pos[halfNe + 1:] = hpos
        else:
            pos[:halfNe] = -hpos[::-1]
            pos[halfNe:] = hpos
        return pos

    def _synthesize_asymmetric_with_dmax(
        self, v: np.ndarray, halfL: float, dmin: float, delta: float
    ) -> np.ndarray:
        pos = np.empty(self._Ne)
        pos[0] = -halfL

        n_v = self._Ne - 1
        for i in range(n_v):
            pos[i + 1] = pos[i] + dmin + delta * v[i]
        rl = self._L - (pos[self._Ne - 1] - pos[0])
        pos += rl * v[self._Ne - 1]

        return pos
