"""输入验证工具函数。"""

import numpy as np


def check_positive(value, name: str = "value"):
    """检查值是否为正数。"""
    if value <= 0:
        raise ValueError(f"{name} 必须 > 0, 得到 {value}")


def check_non_negative(value, name: str = "value"):
    """检查值是否为非负数。"""
    if value < 0:
        raise ValueError(f"{name} 必须 >= 0, 得到 {value}")


def check_1d_array(arr: np.ndarray, name: str = "array"):
    """检查是否为 1D numpy 数组。"""
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是 1D 数组, 得到 shape {arr.shape}")


def check_same_length(a: np.ndarray, b: np.ndarray, name_a: str = "a", name_b: str = "b"):
    """检查两个数组长度是否一致。"""
    if a.size != b.size:
        raise ValueError(f"{name_a} 长度 ({a.size}) 与 {name_b} 长度 ({b.size}) 不一致")
