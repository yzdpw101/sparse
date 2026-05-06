"""方向图计算器。

构造时预计算所有电磁常量（波数）和角度网格（sinθ、deltaSin），
计算时只传阵元位置和激励，避免优化循环中重复计算。

位置坐标均以波长为单位。单频时 k=2π（不使用 wavelength），
多频时根据频率比缩放波数。

支持的维度组合（通过构造函数参数控制）：
  阵列类型: 线阵 (phi 参数不传) / 平面阵 (传 phi 参数)
  扫描角:   单角度 (theta0s 标量) / 多角度 (theta0s 数组)
  频率:     单频 (frequenciesGHz 标量) / 多频 (frequenciesGHz 数组)
"""

from typing import Optional, Union, Literal
import numpy as np


class Pattern:
    """方向图计算器。

    构造时预计算角度网格和电磁常量，后续 compute 只传阵元配置。

    Args:
        array_type: 阵列类型, "linear"=线阵, "planar"=平面阵
        theta_deg_start: 俯仰角起始（度）
        theta_deg_end: 俯仰角终止（度）
        theta_deg_step: 俯仰角步长（度）
        theta0s_deg: 波束指向俯仰角（度），标量=单角度，数组=多角度
        phi_deg_start: 方位角起始（度），array_type="planar" 时必须指定
        phi_deg_end: 方位角终止（度）
        phi_deg_step: 方位角步长（度）
        phi0s_deg: 波束指向方位角（度），标量=单个，数组=多个
        frequenciesGHz: 工作频率（GHz），标量=单频，数组=多频
    """

    def __init__(
        self,
        array_type: Literal["linear", "planar"] = "linear",
        theta_deg_start: float = -90,
        theta_deg_end: float = 90,
        theta_deg_step: float = 0.1,
        theta0s_deg: Union[float, np.ndarray] = 0.0,
        phi_deg_start: Optional[float] = None,
        phi_deg_end: Optional[float] = None,
        phi_deg_step: Optional[float] = None,
        phi0s_deg: Union[float, np.ndarray, None] = None,
        frequenciesGHz: Union[float, np.ndarray] = 1.0,
    ):
        # ── 阵列类型 ──
        if array_type not in ("linear", "planar"):
            raise ValueError(f"array_type 必须是 'linear' 或 'planar', 实际 '{array_type}'")
        self._array_type = array_type

        # ── 频率 ──
        self.frequenciesGHz = np.atleast_1d(
            np.asarray(frequenciesGHz, dtype=float)
        )
        self._n_freq = len(self.frequenciesGHz)
        self._ks = 2 * np.pi * np.ones(self._n_freq)

        # ── theta 网格 ──
        self.theta_deg = np.arange(
            theta_deg_start, theta_deg_end + theta_deg_step / 2, theta_deg_step
        )
        self._n_theta = len(self.theta_deg)
        self._sin_theta = np.sin(np.deg2rad(self.theta_deg))

        # ── phi 网格 ──
        if array_type == "planar":
            if phi_deg_start is None or phi_deg_end is None or phi_deg_step is None:
                raise ValueError("平面阵必须指定 phi_deg_start, phi_deg_end, phi_deg_step")
            self.phi_deg = np.arange(
                phi_deg_start, phi_deg_end + phi_deg_step / 2, phi_deg_step
            )
            self._n_phi = len(self.phi_deg)
            self._sin_phi = np.sin(np.deg2rad(self.phi_deg))
            self._cos_phi = np.cos(np.deg2rad(self.phi_deg))
        else:
            self.phi_deg = None
            self._n_phi = 0

        # ── 扫描角 ──
        self.theta0s_deg = np.atleast_1d(
            np.asarray(theta0s_deg, dtype=float)
        )
        self._n_scan = len(self.theta0s_deg)
        self._sin_theta0s = np.sin(np.deg2rad(self.theta0s_deg))

        # 预计算 delta_sin = sinθ − sinθ₀
        _delta = self._sin_theta[None, :] - self._sin_theta0s[:, None]
        if self._n_scan == 1:
            self._delta_sin = _delta[0]          # (Nθ,)
        else:
            self._delta_sin = _delta              # (Nscan, Nθ)

        # 平面阵扫描角预处理
        if array_type == "planar":
            phi0s = np.atleast_1d(
                np.asarray(phi0s_deg if phi0s_deg is not None else 0.0, dtype=float)
            )
            self._phi0s_deg = phi0s
            self._n_phi0 = len(phi0s)
            self._sin_phi0s = np.sin(np.deg2rad(phi0s))
            self._cos_phi0s = np.cos(np.deg2rad(phi0s))
        else:
            self._phi0s_deg = None

    # ============================================================
    #  属性
    # ============================================================

    @property
    def array_type(self) -> str:
        return self._array_type

    @property
    def is_planar(self) -> bool:
        return self._array_type == "planar"

    @property
    def is_multi_freq(self) -> bool:
        return self._n_freq > 1

    @property
    def is_multi_scan(self) -> bool:
        return self._n_scan > 1

    # ============================================================
    #  统一 AF 入口
    # ============================================================

    def af(
        self,
        positions_x: np.ndarray,
        positions_y: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """统一阵因子入口，根据 array_type 自动分派。

        Args:
            positions_x: 阵元 x 坐标（波长单位），shape (N,)
            positions_y: 阵元 y 坐标，仅 array_type="planar" 时需要
            amplitudes: 激励幅度，shape (N,)，默认全 1
            phases: 激励相位（弧度），shape (N,)，默认全 0

        Returns:
            复数阵因子
        """
        if self._array_type == "linear":
            return self.linear_af(positions_x, amplitudes, phases)
        if positions_y is None:
            raise ValueError("平面阵必须提供 positions_y")
        return self.planar_af(positions_x, positions_y, amplitudes, phases)

    # ============================================================
    #  线阵
    # ============================================================

    def linear_af(
        self,
        positions: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """线阵非对称阵因子。

        对应 C++: Pattern::Linear::calcAF(deltaSinTheta, wlpos, excitations)
        delta_sin 已在构造时预计算。

        AF(θ) = Σ A_n · exp(j · 2π · x_n · (sinθ − sinθ₀))

        Args:
            positions: 阵元 x 坐标（波长单位），shape (N,)
            amplitudes: 激励幅度，shape (N,)，默认全 1
            phases: 激励相位（弧度），shape (N,)，默认全 0

        Returns:
            复数阵因子
              单频单角度: (Nθ,)
              单频多角度: (Nscan, Nθ)
              多频单角度: (Nfreq, Nθ)
              多频多角度: (Nfreq, Nscan, Nθ)
        """
        positions = np.asarray(positions, dtype=float)
        n = len(positions)
        amps = np.ones(n) if amplitudes is None else np.asarray(amplitudes, dtype=float)
        phs = np.zeros(n) if phases is None else np.asarray(phases, dtype=float)

        excitations = amps * np.exp(1j * phs)

        if self._n_freq == 1:
            k = self._ks[0]
            # phase: (N, *delta_sin_shape)
            phase = k * np.outer(positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(n, *self._delta_sin.shape)
            # einsum('i,i...->...') 支持任意维度: (Nθ,) 或 (Nscan, Nθ)
            return np.einsum("i,i...->...", excitations, np.exp(1j * phase))

        # 多频 → (Nfreq, *delta_sin_shape)
        afs = []
        for k in self._ks:
            phase = k * np.outer(positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(n, *self._delta_sin.shape)
            afs.append(np.einsum("i,i...->...", excitations, np.exp(1j * phase)))
        return np.array(afs)

    def linear_af_symmetric(
        self,
        half_positions: np.ndarray,
        has_center: bool = True,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """线阵对称阵因子。框架。"""
        raise NotImplementedError("待实现")

    # ============================================================
    #  平面阵
    # ============================================================

    def planar_af(
        self,
        positions_x: np.ndarray,
        positions_y: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """平面阵阵因子。框架。"""
        raise NotImplementedError("待实现")

    def planar_uv_af(
        self,
        positions_x: np.ndarray,
        positions_y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """平面阵 UV 空间阵因子。框架。"""
        raise NotImplementedError("待实现")

    # ============================================================
    #  通用后处理
    # ============================================================

    @staticmethod
    def normalize(af: np.ndarray) -> np.ndarray:
        """线性归一化: |af| / max(|af|), 范围 [0, 1]。

        Args:
            af: 复数阵因子

        Returns:
            归一化幅度
        """
        return np.abs(af) / np.max(np.abs(af))

    @staticmethod
    def to_dB(af: np.ndarray, normalized: bool = True) -> np.ndarray:
        """阵因子转 dB。

        Args:
            af: 复数阵因子
            normalized: True 时先归一化再转 dB（峰值 0 dB），默认开启

        Returns:
            dB 方向图
        """
        if normalized:
            return 20 * np.log10(np.abs(af) / np.max(np.abs(af)) + 1e-30)
        return 20 * np.log10(np.abs(af) + 1e-30)
