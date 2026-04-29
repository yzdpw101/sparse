"""方向图分析：MSLL、HPBW、SLL 等关键指标提取。"""

from typing import Optional
import numpy as np


class Pattern:
    """方向图分析类。

    从阵因子方向图中提取最大副瓣电平 (MSLL)、
    半功率波束宽度 (HPBW)、副瓣峰值列表等指标。

    Args:
        theta: 俯仰角 (度)，shape (Nθ,)
        af_dB: 归一化方向图 (dB)，最大值应为 0 dB
        phi: 方位角 (度)，若不为 None 则为平面阵方向图
    """
    def __init__(self, theta: np.ndarray, af_dB: np.ndarray,
                 phi: Optional[np.ndarray] = None):
        self.theta = np.asarray(theta, dtype=float)
        self.af_dB = np.asarray(af_dB, dtype=float)
        self.phi = np.asarray(phi, dtype=float) if phi is not None else None
        self._is_planar = phi is not None

        # 平面阵暂不支持自动 MSLL 检测（需要 2D 主瓣区域定义）
        # 线性阵 θ 范围应为 [-90, 90] 或 [0, 180]
        self._theta_min = float(np.min(self.theta))
        self._theta_max = float(np.max(self.theta))

    def _find_mainlobe_region(self) -> tuple[int, int]:
        """自动检测主瓣区域（线性阵）。

        从峰值向两侧搜索第一个局部极小值（近似为零点），
        由此确定主瓣边界。

        Returns:
            (left_idx, right_idx): 主瓣边界索引
        """
        if self._is_planar:
            raise NotImplementedError("平面阵需手动指定 mainlobe_region")

        pattern = self.af_dB
        peak_idx = int(np.argmax(pattern))

        def find_first_null(start_idx: int, direction: int) -> int:
            """从 start_idx 沿 direction 方向搜索第一个局部极小值。"""
            idx = start_idx
            n = len(pattern)
            # 先下降到 -10dB 以下（越过主瓣肩部）
            while 0 <= idx < n and pattern[idx] > -10:
                idx += direction
            # 然后找第一个局部极小值（零点）
            while 0 < idx < n - 1:
                if direction == 1:  # 向右
                    if pattern[idx] <= pattern[idx - 1] and pattern[idx] <= pattern[idx + 1]:
                        break
                else:  # 向左
                    if pattern[idx] <= pattern[idx - 1] and pattern[idx] <= pattern[idx + 1]:
                        break
                idx += direction
            return max(0, min(idx, n - 1))

        left = find_first_null(peak_idx, -1)
        right = find_first_null(peak_idx, 1)

        return (left, right)

    def _find_side_lobe_peaks(self, mainlobe_region: tuple[int, int]) -> np.ndarray:
        """在副瓣区域中查找所有峰值电平。

        对主瓣左右两侧的副瓣区域分别检测峰值。
        """
        left, right = mainlobe_region
        pattern = self.af_dB

        # 将主瓣区域置为 -inf，避免干扰副瓣检测
        search_pattern = pattern.copy()
        search_pattern[left:right + 1] = -np.inf

        peaks = []
        i = 1
        while i < len(search_pattern) - 1:
            if (search_pattern[i] >= search_pattern[i - 1]
                    and search_pattern[i] >= search_pattern[i + 1]
                    and np.isfinite(search_pattern[i])):
                peaks.append(search_pattern[i])
                # 跳过相邻的峰值点
                while i < len(search_pattern) - 1 and search_pattern[i] >= search_pattern[i + 1]:
                    i += 1
            i += 1

        return np.array(peaks) if peaks else np.array([-np.inf])

    def get_msll(self, mainlobe_region: Optional[tuple] = None) -> float:
        """计算最大副瓣电平 (MSLL)。

        Args:
            mainlobe_region: (theta_start, theta_end) 主瓣区域（度），
                             若为 None 则自动检测

        Returns:
            MSLL 值 (dB)，例如均匀直线阵理论值为 -13.26 dB
        """
        if self._is_planar:
            raise NotImplementedError("平面阵 MSLL 计算暂未实现")

        if mainlobe_region is None:
            idx_range = self._find_mainlobe_region()
        else:
            start_deg, end_deg = mainlobe_region
            start_idx = int(np.argmin(np.abs(self.theta - start_deg)))
            end_idx = int(np.argmin(np.abs(self.theta - end_deg)))
            idx_range = (start_idx, end_idx)

        peaks = self._find_side_lobe_peaks(idx_range)
        return float(np.max(peaks)) if len(peaks) > 0 else -np.inf

    def get_hpbw(self) -> float:
        """半功率波束宽度 (HPBW)，单位度。

        在主瓣两侧找到 -3dB 交叉点，计算角度差。
        """
        pattern = self.af_dB
        peak_idx = int(np.argmax(pattern))
        peak_val = pattern[peak_idx]
        hp_level = peak_val - 3.0  # -3dB

        # 向左搜索 -3dB 点
        left = peak_idx
        while left > 0 and pattern[left] > hp_level:
            left -= 1
        # 线性插值提高精度
        if left < len(pattern) - 1 and pattern[left + 1] != pattern[left]:
            frac = (hp_level - pattern[left]) / (pattern[left + 1] - pattern[left])
            left_deg = self.theta[left] + frac * (self.theta[left + 1] - self.theta[left])
        else:
            left_deg = self.theta[left]

        # 向右搜索 -3dB 点
        right = peak_idx
        while right < len(pattern) - 1 and pattern[right] > hp_level:
            right += 1
        if right > 0 and pattern[right] != pattern[right - 1]:
            frac = (hp_level - pattern[right - 1]) / (pattern[right] - pattern[right - 1])
            right_deg = self.theta[right - 1] + frac * (self.theta[right] - self.theta[right - 1])
        else:
            right_deg = self.theta[right]

        return float(abs(right_deg - left_deg))

    def get_sll(self, mainlobe_region: Optional[tuple] = None) -> np.ndarray:
        """返回所有副瓣峰值电平 (dB)。"""
        if mainlobe_region is None:
            idx_range = self._find_mainlobe_region()
        else:
            start, end = mainlobe_region
            idx_range = (int(np.argmin(np.abs(self.theta - start))),
                         int(np.argmin(np.abs(self.theta - end))))
        return self._find_side_lobe_peaks(idx_range)

    def get_directivity(self) -> float:
        """方向性系数 (dBi)，通过数值积分计算。

        D = 4π / ∫∫|F(θ,φ)|² sinθ dθ dφ
        """
        if self._is_planar:
            theta_rad = np.deg2rad(self.theta)
            phi_rad = np.deg2rad(self.phi)
            # |F|² = 10^(af_dB/10)
            f_sq = 10 ** (self.af_dB / 10.0)
            # 数值积分
            sin_theta = np.sin(theta_rad)
            integrand = f_sq * sin_theta[:, None]
            d_theta = np.deg2rad(np.abs(self.theta[1] - self.theta[0]))
            d_phi = np.deg2rad(np.abs(self.phi[1] - self.phi[0]))
            integral = np.trapz(np.trapz(integrand, theta_rad, axis=0), phi_rad)
        else:
            theta_rad = np.deg2rad(self.theta)
            f_sq = 10 ** (self.af_dB / 10.0)
            sin_theta = np.sin(theta_rad)
            integrand = f_sq * sin_theta  # 假设 φ 方向均匀
            integral = np.trapz(integrand, theta_rad) * 2 * np.pi

        d = 4 * np.pi / (integral + 1e-30)
        return float(10 * np.log10(d))
