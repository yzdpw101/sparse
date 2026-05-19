"""HFSS IO 工具: CSV 读取、AEP 加载、相位计算。"""

import csv as _csv
import os
from pathlib import Path

import numpy as np


def read_hfss_csv(csv_path):
    """读取 HFSS 导出的 CSV, 返回线性值 numpy 数组 (跳过表头)。

    Args:
        csv_path: CSV 文件路径

    Returns:
        np.ndarray: 第二列的浮点值
    """
    values = []
    with open(csv_path, "r") as f:
        reader = _csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                try:
                    values.append(float(row[1]))
                except ValueError:
                    continue
    return np.array(values)


def read_hfss_csv_db(csv_path, power_db=True):
    """读取 HFSS CSV 并转为归一化 dB 方向图。

    Args:
        csv_path: CSV 文件路径
        power_db: True 用 10*log10 (功率), False 用 20*log10 (场强)

    Returns:
        np.ndarray: 归一化 dB 方向图
    """
    yf_linear = read_hfss_csv(csv_path)
    log_fn = np.log10(np.maximum(yf_linear, 1e-30))
    yf_db = (10.0 if power_db else 20.0) * log_fn
    return yf_db - np.max(yf_db)


def read_hfss_dir(hfss_dir):
    """读取目录下第一个 CSV, 返回归一化 dB 方向图。

    Args:
        hfss_dir: 包含 HFSS CSV 的目录

    Returns:
        np.ndarray: 归一化 dB 方向图
    """
    csv_files = [f for f in os.listdir(hfss_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"目录下无 CSV 文件: {hfss_dir}")
    return read_hfss_csv_db(os.path.join(hfss_dir, csv_files[0]))


def load_aep_patterns(aep_csv_dir, freqs, theta, num_elements,
                      is_gain=True, in_dB=False,
                      input_theta_range=(-180, 180)):
    """加载 AEP 单元方向图 (封装 ElementPattern.from_hfss_aep)。

    Args:
        aep_csv_dir: AEP CSV 目录 (包含 aep_elem_*.csv)
        freqs: 频率数组 (GHz)
        theta: theta 角度网格 (度), 步长由 theta 数组自身决定
        num_elements: 单元数
        is_gain: True=Gain, False=RealizedGain
        in_dB: CSV 数据是否 dB
        input_theta_range: CSV theta 范围 (start, end)

    Returns:
        list: AEP 方向图数据 (from_hfss_aep 返回格式)
    """
    from .element_pattern import ElementPattern

    theta_step = abs(float(theta[1] - theta[0]))
    return ElementPattern.from_hfss_aep(
        str(aep_csv_dir), np.asarray(freqs), theta,
        num_elements=num_elements,
        oriDegStep=theta_step,
        is_gain=is_gain,
        in_dB=in_dB,
        input_theta_range=input_theta_range,
    )


def compute_total_phases(pos_wl, phases_residual_deg, theta0s_deg, lam0):
    """将残余相位 + steering phase 合成为 HFSS total phases。

    AF 内部通过 _delta_sin = sin(θ) - sin(θ₀) 处理波束指向,
    但 HFSS 需要显式的 total phases = 残余 + steering。

    Args:
        pos_wl: 单元位置 (λ)
        phases_residual_deg: 残余相位 (度)
        theta0s_deg: 波束指向角列表 (度), 取第一个
        lam0: 波长 (m)

    Returns:
        np.ndarray: total phases (度), 范围 [0, 360)
    """
    pos_m = np.asarray(pos_wl) * lam0
    phases_res_rad = np.deg2rad(np.asarray(phases_residual_deg))
    k = 2 * np.pi / lam0
    theta0_rad = np.deg2rad(theta0s_deg[0])

    phases_steer_rad = -k * pos_m * np.sin(theta0_rad)
    total_rad = (phases_res_rad + phases_steer_rad) % (2 * np.pi)
    return np.rad2deg(total_rad) % 360
