"""向量化阵因子计算。

对应 C++ Pattern::Linear::calcAF / Pattern::Planar::calcUVAF。
"""

from typing import Optional
import numpy as np

from .geometry import ArrayGeometry


class ArrayFactor:
    """向量化阵因子计算。

    使用 NumPy 广播一次性计算所有方向上的阵因子，
    避免显式循环，适合在优化迭代中反复调用。

    Args:
        geometry: 阵列几何对象
    """
    def __init__(self, geometry: ArrayGeometry):
        self.geometry = geometry
        self.k = 2 * np.pi / geometry.wavelength  # 波数

    def compute_af(self, theta: np.ndarray, phi: Optional[np.ndarray] = None,
                   scan_theta: float = 0.0, scan_phi: float = 0.0) -> np.ndarray:
        """计算阵因子（θ-φ 空间）。

        线性阵 AF(θ) = Σ A_n · exp(j · k · x_n · (sinθ - sinθ₀))
        平面阵 AF(θ,φ) = Σ A_n · exp(j · k · (x_n·u + y_n·v))
        其中 u = sinθ·cosφ - sinθ₀·cosφ₀, v = sinθ·sinφ - sinθ₀·sinφ₀

        Args:
            theta: 俯仰角 (度)，shape (Nθ,)
            phi: 方位角 (度)，若为 None 则计算线性阵 (phi=0)
            scan_theta: 波束指向俯仰角 (度)
            scan_phi: 波束指向方位角 (度)

        Returns:
            线性阵: shape (Nθ,)
            平面阵: shape (Nθ, Nφ)
        """
        theta_rad = np.deg2rad(theta)
        sin_theta = np.sin(theta_rad)
        sin_theta0 = np.sin(np.deg2rad(scan_theta))

        active = self.geometry.active_mask
        # pos 以波长为单位，转成米再与波数 k(1/m) 相乘
        pos_m = self.geometry.positions[active] * self.geometry.wavelength
        amps = self.geometry.amplitudes[active]
        phases = self.geometry.phases[active]
        k = self.k

        # 复激励: A_n · exp(j·phase_n)
        excitations = amps * np.exp(1j * phases)

        if phi is None:
            # === 线性阵 ===
            # AF(θ) = Σ A_n · exp(j · k · x_n · (sinθ - sinθ₀))
            x = pos_m[:, 0]  # (N,)，单位为米
            # (N,) @ (Nθ,) → (N, Nθ)
            phase = k * np.outer(x, sin_theta - sin_theta0)
            af = excitations @ np.exp(1j * phase)  # (Nθ,)
        else:
            # === 平面阵 ===
            phi_rad = np.deg2rad(phi)
            cos_phi0 = np.cos(np.deg2rad(scan_phi))
            sin_phi0 = np.sin(np.deg2rad(scan_phi))

            u = np.outer(sin_theta, np.cos(phi_rad)) - sin_theta0 * cos_phi0
            v = np.outer(sin_theta, np.sin(phi_rad)) - sin_theta0 * sin_phi0

            x = pos_m[:, 0]  # (N,)，单位为米
            y = pos_m[:, 1]  # (N,)

            phase = k * (x[:, None, None] * u[None, :, :]
                         + y[:, None, None] * v[None, :, :])
            af = np.sum(excitations[:, None, None] * np.exp(1j * phase), axis=0)

        return af

    def compute_uv_af(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """UV 空间阵因子计算（对应 C++: Planar::calcUVAF）。

        AF(u,v) = Σ A_n · exp(j · k · (x_n·u + y_n·v))

        Args:
            u: u 坐标网格，u = sinθ·cosφ (2D 矩阵)
            v: v 坐标网格，v = sinθ·sinφ (2D 矩阵)

        Returns:
            shape (Nu, Nv) 的阵因子复数数组
        """
        active = self.geometry.active_mask
        pos_m = self.geometry.positions[active] * self.geometry.wavelength
        amps = self.geometry.amplitudes[active]
        phases = self.geometry.phases[active]
        k = self.k

        excitations = amps * np.exp(1j * phases)
        x = pos_m[:, 0]
        y = pos_m[:, 1]

        phase = k * (x[:, None, None] * u[None, :, :]
                     + y[:, None, None] * v[None, :, :])
        af = np.sum(excitations[:, None, None] * np.exp(1j * phase), axis=0)
        return af

    def compute_af_normalized(self, theta: np.ndarray,
                              phi: Optional[np.ndarray] = None,
                              scan_theta: float = 0.0,
                              scan_phi: float = 0.0) -> np.ndarray:
        """计算归一化阵因子方向图（dB）。

        Returns:
            线性阵: shape (Nθ,)，最大值归一化到 0 dB
            平面阵: shape (Nθ, Nφ)，最大值归一化到 0 dB
        """
        af = self.compute_af(theta, phi, scan_theta, scan_phi)
        af_dB = 20 * np.log10(np.abs(af) + 1e-30)
        return af_dB - np.max(af_dB)

    def compute_array_factor(self, theta: np.ndarray,
                             phi: Optional[np.ndarray] = None) -> np.ndarray:
        """别名，保留兼容性。"""
        return self.compute_af_normalized(theta, phi)
