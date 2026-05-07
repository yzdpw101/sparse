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
        is_gain: bool = True,
        in_dB: bool = False,
        input_theta_range: tuple[float, float] = (-180.0, 180.0),
    ) -> list[np.ndarray]:
        """多频单元方向图导入 — 对应 C++ readFeMultiFreqFromCsvs。

        每个频率一个 CSV 文件: eGain_{freq}GHz.csv 或 {freq}GHz_*.csv。

        Args:
            eGainCsvDirectory: CSV 文件目录
            frequenciesGHz: 频率数组 (GHz)
            theta: 目标 θ 角度网格 (度)
            oriDegStep: CSV 原始 θ 步长 (度)
            phiIdx: CSV 列索引 (0-based), 默认 1
            is_gain: True=CSV 为增益(功率), False=场量
            in_dB: True=CSV 为 dB 值, False=线性值
            input_theta_range: CSV 数据覆盖的 θ 范围 (默认 -180~180)

        转换规则 (增益+dB): Fe=10^(val/20); (场量+dB): Fe=10^(val/10);
                  (增益+线性): Fe=sqrt(val);  (场量+线性): Fe=val.

        Returns:
            list of np.ndarray, 每个频率一个 Fe (field pattern)
        """
        import csv, os

        theta_start = float(theta[0])
        theta_end = float(theta[-1])
        target_step = abs(float(theta[1] - theta[0]))
        step_ratio = int(round(target_step / oriDegStep))
        if step_ratio < 1:
            raise ValueError(f"target step ({target_step}) < ori step ({oriDegStep})")

        in_t0, in_t1 = input_theta_range
        # 文件内行偏移: 第 1 行 = in_t0, 由 input_theta_range 决定
        _row0_theta = in_t0

        result = []
        for fGHz in frequenciesGHz:
            freq_str = f"{fGHz:g}"
            csv_path = os.path.join(eGainCsvDirectory, f"eGain_{freq_str}GHz.csv")
            if not os.path.exists(csv_path):
                candidates = [
                    f for f in os.listdir(eGainCsvDirectory)
                    if f.startswith(f"{freq_str}GHz") and f.endswith(".csv")
                ]
                if candidates:
                    csv_path = os.path.join(eGainCsvDirectory, candidates[0])
                else:
                    raise FileNotFoundError(f"单元方向图文件不存在: {csv_path}")

            # 读取数据 — 行号 i 对应 theta = _row0_theta + i * oriDegStep
            raw = []
            row_start = int(round((theta_start - _row0_theta) / oriDegStep))
            row_end = int(round((theta_end - _row0_theta) / oriDegStep))
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i < row_start:
                        continue
                    if i > row_end:
                        break
                    try:
                        val = float(row[phiIdx]) if len(row) > phiIdx else float(row[0])
                        raw.append(val)
                    except (ValueError, IndexError):
                        continue

            if not raw:
                raise RuntimeError(f"无法从 {csv_path} 读取数据")

            # 降采样
            raw = raw[::step_ratio]

            raw = np.array(raw, dtype=float)

            # 转换为场方向图 Fe
            if in_dB:
                if is_gain:
                    raw = 10.0 ** (raw / 20.0)  # dB 增益 → Fe
                else:
                    raw = 10.0 ** (raw / 10.0)  # dB 场量 → Fe
            else:
                if is_gain:
                    raw = np.sqrt(np.maximum(raw, 0.0))  # 线性增益 → Fe
                # else: 线性场量 → 无需转换

            # 输入范围不足 -180~180 时，补零到 3601 网格
            need_full = (in_t0 > -180.0 or in_t1 < 180.0)
            if need_full and abs(in_t1 - in_t0) < 179.9:
                full_n = 1 + int(round(360.0 / oriDegStep))
                padded = np.full(full_n, 0.0)
                pad_start = int(round((180.0 + in_t0) / oriDegStep))
                n_avail = min(len(raw), full_n - pad_start)
                padded[pad_start:pad_start + n_avail] = raw[:n_avail]
                raw = padded

            Fe = raw

            # 确保长度匹配 theta
            if len(Fe) != len(theta):
                Fe = np.interp(theta, np.linspace(-180.0, 180.0, len(Fe)), Fe)

            result.append(Fe)

        return result
