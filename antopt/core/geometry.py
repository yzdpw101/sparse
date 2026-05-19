"""阵列几何定义：阵元、线性阵列、平面阵列。"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Element:
    """单个阵元。

    Attributes:
        x, y, z: 坐标（波长单位）
        amplitude: 激励幅度
        phase: 激励相位（弧度）
        active: thinning 时标记该元素是否激活
        label: 可选的标识符
    """
    x: float
    y: float = 0.0
    z: float = 0.0
    amplitude: float = 1.0
    phase: float = 0.0
    active: bool = True
    label: str = ""


class ArrayGeometry:
    """阵列几何基类，管理阵元集合及其属性。"""
    def __init__(self, elements: list[Element], frequency: float):
        if not elements:
            raise ValueError("阵元列表不能为空")
        self._elements = list(elements)
        self.frequency = frequency  # Hz
        self.wavelength = 299792458.0 / frequency  # 自由空间波长

    @property
    def num_elements(self) -> int:
        """总阵元数。"""
        return len(self._elements)

    @property
    def num_active(self) -> int:
        """激活的阵元数（仅用于 thinning）。"""
        return sum(1 for e in self._elements if e.active)

    @property
    def positions(self) -> np.ndarray:
        """所有阵元位置，shape (N, 3)。"""
        return np.array([[e.x, e.y, e.z] for e in self._elements])

    @property
    def positions_x(self) -> np.ndarray:
        return np.array([e.x for e in self._elements])

    @property
    def positions_y(self) -> np.ndarray:
        return np.array([e.y for e in self._elements])

    @property
    def positions_z(self) -> np.ndarray:
        return np.array([e.z for e in self._elements])

    @property
    def amplitudes(self) -> np.ndarray:
        return np.array([e.amplitude for e in self._elements])

    @property
    def phases(self) -> np.ndarray:
        return np.array([e.phase for e in self._elements])

    @property
    def active_mask(self) -> np.ndarray:
        return np.array([e.active for e in self._elements])

    @property
    def aperture_size_x(self) -> float:
        """x 方向孔径（波长单位）。"""
        xs = self.positions_x[self.active_mask]
        return float(np.max(xs) - np.min(xs)) if len(xs) > 1 else 0.0

    def set_excitations(self, amplitudes: Optional[np.ndarray] = None,
                        phases: Optional[np.ndarray] = None):
        """批量设置激励幅度和相位。"""
        if amplitudes is not None:
            if amplitudes.size != self.num_elements:
                raise ValueError("amplitudes 长度必须与阵元数一致")
            for e, a in zip(self._elements, amplitudes):
                e.amplitude = a
        if phases is not None:
            if phases.size != self.num_elements:
                raise ValueError("phases 长度必须与阵元数一致")
            for e, p in zip(self._elements, phases):
                e.phase = p

    def activate_subset(self, indices: np.ndarray):
        """仅激活指定索引的阵元，其余关闭。"""
        mask = np.zeros(self.num_elements, dtype=bool)
        mask[indices] = True
        for e, active in zip(self._elements, mask):
            e.active = active

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(N={self.num_elements}, "
                f"active={self.num_active}, f={self.frequency/1e9:.2f}GHz)")


class LinearArray(ArrayGeometry):
    """线性阵列：阵元沿 x 轴排列。

    Args:
        positions: 阵元 x 坐标（波长单位）
        frequency: 工作频率 (Hz)
        amplitudes: 激励幅度，若为 None 则全为 1
        symmetric: 若为 True，positions 为半边位置，自动镜像
    """
    def __init__(self, positions: np.ndarray, frequency: float,
                 amplitudes: Optional[np.ndarray] = None,
                 symmetric: bool = False):
        positions = np.asarray(positions, dtype=float)
        n = len(positions)

        if symmetric:
            # 对称阵列：半边位置镜像，中心元素在 0 处
            if positions[0] == 0:
                # 包含中心元素
                full_pos = np.concatenate([-positions[:0:-1], positions])
                if amplitudes is not None:
                    amp = np.asarray(amplitudes, dtype=float)
                    full_amp = np.concatenate([amp[:0:-1], amp])
                else:
                    full_amp = None
            else:
                # 不包含中心元素
                full_pos = np.concatenate([-positions[::-1], positions])
                if amplitudes is not None:
                    amp = np.asarray(amplitudes, dtype=float)
                    full_amp = np.concatenate([amp[::-1], amp])
                else:
                    full_amp = None

            elements = []
            for i, x in enumerate(full_pos):
                amp = full_amp[i] if full_amp is not None else 1.0
                elements.append(Element(x=x, amplitude=amp))
        else:
            if amplitudes is not None:
                amps = np.asarray(amplitudes, dtype=float)
                if amps.size != n:
                    raise ValueError("amplitudes 长度必须与 positions 一致")
            else:
                amps = np.ones(n)
            elements = [Element(x=float(p), amplitude=float(amps[i]))
                        for i, p in enumerate(positions)]

        super().__init__(elements, frequency)

    @property
    def positions_x(self) -> np.ndarray:
        return np.array([e.x for e in self._elements])


class UniformLinearArray(LinearArray):
    """均匀直线阵列（对比基准）。

    Args:
        num_elements: 阵元数
        spacing: 阵元间距（波长单位）
        frequency: 工作频率 (Hz)
    """
    def __init__(self, num_elements: int, spacing: float, frequency: float):
        if spacing <= 0:
            raise ValueError("间距必须 > 0")
        positions = np.arange(num_elements) * spacing
        # 中心对称：以 0 为中心排列
        positions -= np.mean(positions)
        super().__init__(positions, frequency)


class PlanarArray(ArrayGeometry):
    """平面阵列：阵元在 xy 平面内分布。

    Args:
        positions_x: 阵元 x 坐标（波长单位）
        positions_y: 阵元 y 坐标（波长单位）
        frequency: 工作频率 (Hz)
        symmetric: 是否对称（预留）
    """
    def __init__(self, positions_x: np.ndarray, positions_y: np.ndarray,
                 frequency: float, symmetric: bool = False):
        px = np.asarray(positions_x, dtype=float)
        py = np.asarray(positions_y, dtype=float)
        if px.size != py.size:
            raise ValueError("positions_x 和 positions_y 长度必须一致")

        n = px.size
        elements = [Element(x=float(px[i]), y=float(py[i])) for i in range(n)]
        super().__init__(elements, frequency)
        self._symmetric = symmetric
