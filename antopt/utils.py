"""工具函数。"""

import json
from pathlib import Path
import numpy as np


def compute_pattern(pos, amps, phases, mapper, pat, fe_patterns=None):
    """计算方向图乘积: pattern = Fe * |AF| (实数量)。

    Args:
        pos: 阵元位置数组 (Ne,)
        amps: 激励幅度 (Ne,)
        phases: 激励相位 (rad, Ne,)
        mapper: LMMapper 实例
        pat: Pattern 实例
        fe_patterns: 单元方向图列表 [Fe_array, ...] 或 None

    Returns:
        np.ndarray: 实值方向图 (未 dB, 未归一化)
    """
    if pat.is_planar or not mapper.is_symmetric:
        af = pat.linear_af(pos, amps, phases)
    else:
        halfNe = mapper._halfNe
        kwargs = dict(has_center=mapper.has_center,
                      amplitudes=amps[halfNe:], phases=phases[halfNe:])
        if mapper.has_center:
            kwargs["center_amplitude"] = amps[halfNe - 1]
            kwargs["center_phase"] = phases[halfNe - 1]
        af = pat.linear_af_symmetric(pos[halfNe:], **kwargs)
    af_abs = np.abs(af)
    if fe_patterns is not None:
        af_abs = af_abs * fe_patterns[0]
    return af_abs


def to_json_flat(obj, indent=0):
    """仿 C++ toJsonFlatArrays: 对象换行, 数组紧凑单行。"""
    sp = " " * (indent + 2)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            items.append(f'{sp}{json.dumps(k)}: {to_json_flat(v, indent + 2)}')
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    elif isinstance(obj, list):
        return json.dumps(obj)
    else:
        return json.dumps(obj)


def load_array_config(filename, key, base_dir=None):
    """从 JSON 文件加载 array_config 格式或纯数组。支持相对/绝对路径。

    Args:
        filename: JSON 文件路径
        key: JSON 键名 (如 "xCenters", "phasesDeg", "amplitudes")
        base_dir: 相对路径的基准目录, None 表示当前工作目录

    Returns:
        np.ndarray or None
    """
    if not filename:
        return None
    path = Path(filename)
    if not path.is_absolute():
        path = (Path(base_dir) if base_dir else Path.cwd()) / filename
    if not path.exists():
        raise FileNotFoundError(f"导入文件不存在: {path}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and key in data:
        data = data[key]
    return np.array(data, dtype=float) if data is not None else None
