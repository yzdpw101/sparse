"""单元方向图模型。

提供各向同性、cosine-q、微带贴片等单元方向图，
以及从 HFSS 导出的单元方向图文件导入功能。
"""

from typing import Optional
import numpy as np


class ElementPattern:
    """单元方向图模型。"""
    @staticmethod
    def isotropic(theta: np.ndarray) -> np.ndarray:
        """各向同性单元方向图，在所有方向增益为 1。"""
        return np.ones_like(theta)

    @staticmethod
    def cosine_q(theta: np.ndarray, q: float = 1.0) -> np.ndarray:
        """cos^q(theta) 单元方向图。

        半功率波束宽度随 q 增大而变窄。
        q=1: HPBW ≈ 90°,  q=2: HPBW ≈ 65°
        """
        theta_rad = np.deg2rad(theta)
        pattern = np.abs(np.cos(theta_rad)) ** q
        return pattern

    @staticmethod
    def patch(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """微带贴片天线近似方向图。

        E 面 (phi=0°):  cos(θ)
        H 面 (phi=90°): 1
        """
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        # 简化的贴片模型
        e_plane = np.abs(np.cos(theta_rad))
        # E 面和 H 面的混合
        pattern = e_plane[:, None] * np.ones_like(phi_rad[None, :])
        return pattern

    @staticmethod
    def from_hfss(filepath: str, theta: np.ndarray) -> np.ndarray:
        """从 HFSS 导出的单元方向图文件导入。

        Args:
            filepath: CSV 文件路径，格式应为 [theta_deg, gain_dB]
            theta: 需要插值到的 θ 角度网格

        Returns:
            线性插值后的单元方向图幅值 (线性刻度)
        """
        import csv
        thetas = []
        gains = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        t = float(row[0])
                        g = float(row[1])
                        thetas.append(t)
                        gains.append(g)
                    except ValueError:
                        continue

        if len(thetas) < 2:
            raise ValueError(f"HFSS 文件 {filepath} 数据不足")

        thetas = np.array(thetas)
        gains = np.array(gains)

        # 线性插值到目标 θ 网格
        gain_interp = np.interp(theta, thetas, gains)
        # dB → 线性
        pattern_linear = 10 ** (gain_interp / 20.0)
        return pattern_linear

    @staticmethod
    def from_csv(filepath: str, theta: np.ndarray) -> np.ndarray:
        """从 CSV 文件导入单元方向图（别名）。"""
        return ElementPattern.from_hfss(filepath, theta)
