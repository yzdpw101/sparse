"""辅助数学工具函数。"""

import numpy as np


def theta_phi_to_uv(theta_deg: np.ndarray, phi_deg: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """将 (θ, φ) 角度网格转换为 (u, v) 坐标。

    u = sinθ·cosφ,  v = sinθ·sinφ
    """
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    sin_theta = np.sin(theta_rad)
    u = sin_theta * np.cos(phi_rad)
    v = sin_theta * np.sin(phi_rad)
    return u, v


def uv_to_theta_phi(u: np.ndarray, v: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """将 (u, v) 坐标转换回 (θ, φ) 角度（度）。"""
    theta_rad = np.arcsin(np.sqrt(u**2 + v**2))
    phi_rad = np.arctan2(v, u)
    return np.rad2deg(theta_rad), np.rad2deg(phi_rad)


def wavelength(frequency_hz: float) -> float:
    """计算自由空间波长 (m)。"""
    return 299792458.0 / frequency_hz


def wavenumber(frequency_hz: float) -> float:
    """计算波数 k = 2π/λ。"""
    return 2 * np.pi / wavelength(frequency_hz)


def quantize_amplitude(amplitudes: np.ndarray, bits: int = 4) -> np.ndarray:
    """幅度量化（用于幅相加权优化验证）。"""
    levels = 2**bits
    return np.round(amplitudes * (levels - 1)) / (levels - 1)
