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

    @staticmethod
    def from_hfss_multi_freq(
        eGainCsvDirectory: str,
        frequenciesGHz: np.ndarray,
        theta: np.ndarray,
        oriDegStep: float,
        phiIdx: int = 1,
    ) -> list[np.ndarray]:
        """多频单元方向图导入 — 对应 C++ readFeMultiFreqFromCsvs。

        每个频率一个 CSV 文件: eGain_{freq}GHz.csv。

        Args:
            eGainCsvDirectory: CSV 文件目录
            frequenciesGHz: 频率数组 (GHz)，shape (Nf,)
            theta: 目标 θ 角度网格 (度)
            oriDegStep: CSV 原始 θ 步长 (度)
            phiIdx: CSV 列索引 (0-based), 默认 1 = phi=0° 增益列

        Returns:
            list of np.ndarray, 每个频率一个 Fe (field pattern = sqrt(gain))
        """
        import csv, os

        theta_start = float(theta[0])
        theta_end = float(theta[-1])
        target_step = abs(float(theta[1] - theta[0]))
        step_ratio = int(round(target_step / oriDegStep))
        if step_ratio < 1:
            raise ValueError(f"target step ({target_step}) < ori step ({oriDegStep})")

        result = []
        for fGHz in frequenciesGHz:
            # 格式化频率名: 去掉末尾无效零, 先找 C++ 命名, 再找任意匹配
            freq_str = f"{fGHz:g}"
            csv_path = os.path.join(eGainCsvDirectory, f"eGain_{freq_str}GHz.csv")
            if not os.path.exists(csv_path):
                # 回退: 找以 "{freq_str}GHz" 开头的 CSV
                candidates = [
                    f for f in os.listdir(eGainCsvDirectory)
                    if f.startswith(f"{freq_str}GHz") and f.endswith(".csv")
                ]
                if candidates:
                    csv_path = os.path.join(eGainCsvDirectory, candidates[0])
                else:
                    raise FileNotFoundError(f"单元方向图文件不存在: {csv_path}")

            # 读第二列（phi=0 数据）
            gains = []
            row_start = 1 + int(round((180.0 + theta_start) / oriDegStep))
            row_end = 1 + int(round((180.0 + theta_end) / oriDegStep))

            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue  # 跳过表头
                    if i < row_start:
                        continue
                    if i > row_end:
                        break
                    if len(row) >= 2:
                        try:
                            gains.append(float(row[phiIdx]))
                        except (ValueError, IndexError):
                            continue

            if not gains:
                raise RuntimeError(f"无法从 {csv_path} 读取数据")

            # 降采样
            gains = gains[::step_ratio]
            gains = np.array(gains, dtype=float)

            # Fe = sqrt(max(0, gain))
            Fe = np.sqrt(np.maximum(gains, 0.0))

            # 确保长度匹配 theta
            if len(Fe) != len(theta):
                # 长度不匹配时重采样
                Fe = np.interp(theta, np.linspace(theta_start, theta_end, len(Fe)), Fe)

            result.append(Fe)

        return result
