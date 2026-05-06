"""方向图计算器。

构造时预计算所有电磁常量（波数）和角度网格（sinθ、deltaSin），
计算时只传阵元位置和激励，避免优化循环中重复计算。

位置坐标均以波长为单位。单频时 k=2π（不使用 wavelength），
多频时根据频率比缩放波数。

支持的维度组合（通过构造函数参数控制）：
  阵列类型: 线阵 / 平面阵 (array_type)
  对称性:   非对称 / 对称 (symmetric)
  扫描角:   单角度 (theta0s 标量) / 多角度 (theta0s 数组)
  频率:     单频 (frequenciesGHz 标量) / 多频 (frequenciesGHz 数组)
"""

from typing import Optional, Union, Literal
import numpy as np

TWO_PI = 2 * np.pi


class Pattern:
    """方向图计算器。

    构造时预计算角度网格和电磁常量，后续 compute 只传阵元配置。

    Args:
        array_type: 阵列类型, "linear"=线阵, "planar"=平面阵
        symmetric: 是否对称阵列，True 时只传半边位置
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
        symmetric: bool = False,
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
        self._symmetric = symmetric

        # ── 频率 ──
        self.frequenciesGHz = np.atleast_1d(
            np.asarray(frequenciesGHz, dtype=float)
        )
        self._n_freq = len(self.frequenciesGHz)
        # k = 2π * (f / f₀), 单频时 f/f₀ = 1 → k = 2π
        self._ks = TWO_PI * self.frequenciesGHz / self.frequenciesGHz[0]

        # ── theta 网格 ──
        self.theta_deg = np.arange(
            theta_deg_start, theta_deg_end + theta_deg_step / 2, theta_deg_step
        )
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
            self._sin_phi0s = np.sin(np.deg2rad(phi0s))
            self._cos_phi0s = np.cos(np.deg2rad(phi0s))
        else:
            self._phi0s_deg = None

        # ── 平面阵 UV 空间预计算 ──
        if array_type == "planar":
            # deltaU = sinθ·cosφ − sinθ₀·cosφ₀  (Nθ, Nφ)
            # deltaV = sinθ·sinφ − sinθ₀·sinφ₀  (Nθ, Nφ)
            if self._n_scan == 1:
                sin0 = self._sin_theta0s[0]
                cos0 = self._cos_phi0s[0]
                sinp0 = self._sin_phi0s[0]
                self._delta_u = (
                    np.outer(self._sin_theta, self._cos_phi) - sin0 * cos0
                )
                self._delta_v = (
                    np.outer(self._sin_theta, self._sin_phi) - sin0 * sinp0
                )
            else:
                # 多角度: (Nscan, Nθ, Nφ)
                sin_th = self._sin_theta[None, :, None]   # (1, Nθ, 1)
                cos_ph = self._cos_phi[None, None, :]     # (1, 1, Nφ)
                sin_ph = self._sin_phi[None, None, :]     # (1, 1, Nφ)
                sin0s = self._sin_theta0s[:, None, None]   # (Nscan, 1, 1)
                c0s = self._cos_phi0s[:, None, None]      # (Nscan, 1, 1)
                s0s = self._sin_phi0s[:, None, None]      # (Nscan, 1, 1)
                self._delta_u = sin_th * cos_ph - sin0s * c0s
                self._delta_v = sin_th * sin_ph - sin0s * s0s
        else:
            self._delta_u = None
            self._delta_v = None

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
    def symmetric(self) -> bool:
        return self._symmetric

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
        wl_x: np.ndarray,
        wl_y: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """统一阵因子入口，根据 array_type 自动分派。

        Args:
            wl_x: 阵元 x 坐标（以第一频率波长 λ₀ 为单位），shape (N,)
            wl_y: 阵元 y 坐标（波长单位），仅 array_type="planar" 时需要
            amplitudes: 激励幅度，shape (N,)，默认全 1
            phases: 激励相位（弧度），shape (N,)，默认全 0

        Returns:
            复数阵因子
        """
        if self._array_type == "linear":
            if self._symmetric:
                return self.linear_af_symmetric(wl_x, amplitudes=amplitudes,
                                                phases=phases)
            return self.linear_af(wl_x, amplitudes, phases)
        # planar
        if wl_y is None:
            raise ValueError("平面阵必须提供 wl_y")
        if self._symmetric:
            return self.planar_af_symmetric(wl_x, wl_y,
                                            amplitudes, phases)
        return self.planar_af(wl_x, wl_y, amplitudes, phases)

    # ============================================================
    #  线阵
    # ============================================================

    def linear_af(
        self,
        wl_positions: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """线阵非对称阵因子。

        AF(θ) = Σ A_n · exp(j · k · x_n · (sinθ − sinθ₀))

        Args:
            wl_positions: 阵元 x 坐标（以第一频率波长 λ₀ 为单位），shape (N,)
            amplitudes: 激励幅度，shape (N,)，None=全 1
            phases: 激励相位（弧度），shape (N,)，None=全 0

        Returns:
            复数阵因子
        """
        positions = np.asarray(wl_positions, dtype=float)
        n = len(positions)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        if self._n_freq == 1:
            k = TWO_PI
            phase = k * np.outer(positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(n, *self._delta_sin.shape)
            e = np.exp(1j * phase)

            if not has_amps and not has_phs:
                return np.sum(e, axis=0)
            if not has_phs:
                b = amps.reshape(n, *((1,) * (e.ndim - 1)))
                return np.sum(b * e, axis=0)
            return np.einsum("i,i...->...", exc, e)

        # 多频
        afs = []
        for k in self._ks:
            phase = k * np.outer(positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(n, *self._delta_sin.shape)
            e = np.exp(1j * phase)

            if not has_amps and not has_phs:
                afs.append(np.sum(e, axis=0))
            elif not has_phs:
                b = amps.reshape(n, *((1,) * (e.ndim - 1)))
                afs.append(np.sum(b * e, axis=0))
            else:
                afs.append(np.einsum("i,i...->...", exc, e))
        return np.array(afs)

    def linear_af_symmetric(
        self,
        half_wl_positions: np.ndarray,
        has_center: bool = False,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """线阵对称阵因子（关于原点 x→−x 对称）。

        AF = Σ 2·A_n·cos(k·x_n·(sinθ−sinθ₀))  (+ center)

        Args:
            half_wl_positions: 半边阵元 x 坐标（x > 0，以 λ₀ 为单位），shape (Nh,)
            has_center: 是否有中心阵元（x=0，激励 1+0j）
            amplitudes: 半边激励幅度，shape (Nh,)，None=全 1
            phases: 半边激励相位（弧度），shape (Nh,)，None=全 0

        Returns:
            复数阵因子
        """
        half_positions = np.asarray(half_wl_positions, dtype=float)
        nh = len(half_positions)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        if self._n_freq == 1:
            k = TWO_PI
            phase = k * np.outer(half_positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(nh, *self._delta_sin.shape)
            c = np.cos(phase)

            if not has_amps and not has_phs:
                af = 2 * np.sum(c, axis=0)
            elif not has_phs:
                b = amps.reshape(nh, *((1,) * (c.ndim - 1)))
                af = 2 * np.sum(b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, 2 * c)

            if has_center:
                af = af + 1.0
            return af

        # 多频
        afs = []
        for k in self._ks:
            phase = k * np.outer(half_positions, self._delta_sin.reshape(-1))
            phase = phase.reshape(nh, *self._delta_sin.shape)
            c = np.cos(phase)

            if not has_amps and not has_phs:
                af = 2 * np.sum(c, axis=0)
            elif not has_phs:
                b = amps.reshape(nh, *((1,) * (c.ndim - 1)))
                af = 2 * np.sum(b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, 2 * c)

            if has_center:
                af = af + 1.0
            afs.append(af)
        return np.array(afs)

    # ============================================================
    #  平面阵
    # ============================================================

    def planar_af(
        self,
        wl_x: np.ndarray,
        wl_y: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """平面阵非对称阵因子（UV 空间）。

        AF = Σ A_n · exp(j·k·(x_n·deltaU + y_n·deltaV))

        Args:
            wl_x: 阵元 x 坐标（以第一频率波长 λ₀ 为单位），shape (N,)
            wl_y: 阵元 y 坐标（波长单位），shape (N,)
            amplitudes: 激励幅度，shape (N,)，None=全 1
            phases: 激励相位（弧度），shape (N,)，None=全 0

        Returns:
            复数阵因子
              单频单角度: (Nθ, Nφ)
              单频多角度: (Nscan, Nθ, Nφ)
        """
        if self._array_type != "planar":
            raise ValueError("array_type='linear' 不支持 planar_af")

        x = np.asarray(wl_x, dtype=float)
        y = np.asarray(wl_y, dtype=float)
        n = len(x)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        if self._n_freq == 1:
            # phase = k * (x_i·deltaU + y_i·deltaV)
            # shape: (N, *delta_shape)
            phase = TWO_PI * (
                np.outer(x, self._delta_u.ravel()) +
                np.outer(y, self._delta_v.ravel())
            )
            phase = phase.reshape(n, *self._delta_u.shape)
            e = np.exp(1j * phase)

            if not has_amps and not has_phs:
                return np.sum(e, axis=0)
            if not has_phs:
                b = amps.reshape(n, *((1,) * (e.ndim - 1)))
                return np.sum(b * e, axis=0)
            return np.einsum("i,i...->...", exc, e)

        # 多频
        afs = []
        for k in self._ks:
            phase = k * (
                np.outer(x, self._delta_u.ravel()) +
                np.outer(y, self._delta_v.ravel())
            )
            phase = phase.reshape(n, *self._delta_u.shape)
            e = np.exp(1j * phase)

            if not has_amps and not has_phs:
                afs.append(np.sum(e, axis=0))
            elif not has_phs:
                b = amps.reshape(n, *((1,) * (e.ndim - 1)))
                afs.append(np.sum(b * e, axis=0))
            else:
                afs.append(np.einsum("i,i...->...", exc, e))
        return np.array(afs)

    def planar_af_symmetric(
        self,
        quarter_wl_x: np.ndarray,
        quarter_wl_y: np.ndarray,
        has_center: bool = False,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """平面阵四象限对称阵因子（关于 x→−x, y→−y 对称）。

        AF = Σ sym_i · A_i · cos(k·x_i·Δu) · cos(k·y_i·Δv)  (+ center)

        其中 sym_i = 4（x>0,y>0）、2（轴上单元素）、原点由 has_center 处理。

        Args:
            quarter_wl_x: 第一象限阵元 x 坐标（以 λ₀ 为单位），shape (Nq,)
            quarter_wl_y: 第一象限阵元 y 坐标（以 λ₀ 为单位），shape (Nq,)
            has_center: 是否有中心阵元（x=0,y=0，激励 1+0j）
            amplitudes: 四分之一象限激励幅度，shape (Nq,)，None=全 1
            phases: 四分之一象限激励相位（弧度），shape (Nq,)，None=全 0

        Returns:
            复数阵因子
              单频单角度: (Nθ, Nφ)
              单频多角度: (Nscan, Nθ, Nφ)
              多频: (Nfreq, ...)
        """
        if self._array_type != "planar":
            raise ValueError("array_type='linear' 不支持 planar_af_symmetric")

        qx = np.asarray(quarter_wl_x, dtype=float)
        qy = np.asarray(quarter_wl_y, dtype=float)
        nq = len(qx)

        has_amps = amplitudes is not None
        has_phs = phases is not None

        if has_amps:
            amps = np.asarray(amplitudes, dtype=float)
        if has_phs:
            exc = np.exp(1j * np.asarray(phases, dtype=float))
            if has_amps:
                exc = amps * exc

        # 对称因子：x>0且y>0 → 4, 仅在一个轴上 → 2
        eps = 1e-12
        x_on_axis = qx <= eps
        y_on_axis = qy <= eps
        sym_factor = np.where(
            x_on_axis & y_on_axis, 0.0,          # 原点由 has_center 处理
            np.where(x_on_axis | y_on_axis, 2.0, 4.0),
        )

        if self._n_freq == 1:
            k = TWO_PI
            phase_u = k * np.outer(qx, self._delta_u.ravel())
            phase_v = k * np.outer(qy, self._delta_v.ravel())
            phase_u = phase_u.reshape(nq, *self._delta_u.shape)
            phase_v = phase_v.reshape(nq, *self._delta_v.shape)
            c = np.cos(phase_u) * np.cos(phase_v)  # (Nq, Nθ, Nφ) 或 (Nq, Nscan, Nθ, Nφ)

            sf = sym_factor.reshape(nq, *((1,) * (c.ndim - 1)))

            if not has_amps and not has_phs:
                af = np.sum(sf * c, axis=0)
            elif not has_phs:
                b = amps.reshape(nq, *((1,) * (c.ndim - 1)))
                af = np.sum(sf * b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, sf * c)

            if has_center:
                af = af + 1.0
            return af

        # 多频
        afs = []
        for k in self._ks:
            phase_u = k * np.outer(qx, self._delta_u.ravel())
            phase_v = k * np.outer(qy, self._delta_v.ravel())
            phase_u = phase_u.reshape(nq, *self._delta_u.shape)
            phase_v = phase_v.reshape(nq, *self._delta_v.shape)
            c = np.cos(phase_u) * np.cos(phase_v)

            sf = sym_factor.reshape(nq, *((1,) * (c.ndim - 1)))

            if not has_amps and not has_phs:
                af = np.sum(sf * c, axis=0)
            elif not has_phs:
                b = amps.reshape(nq, *((1,) * (c.ndim - 1)))
                af = np.sum(sf * b * c, axis=0)
            else:
                af = np.einsum("i,i...->...", exc, sf * c)

            if has_center:
                af = af + 1.0
            afs.append(af)
        return np.array(afs)

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
        return np.abs(af) / (np.max(np.abs(af)) + 1e-30)

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
