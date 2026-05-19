"""分层配置类: BaseConfig → SparseConfig → ASMConfig / MSMConfig。"""

import json
from dataclasses import dataclass, field
from pathlib import Path


def _check_step(step, range_val, name):
    """验证 step 能整除 range。"""
    if step <= 0:
        raise ValueError(f"{name} step must be > 0, got {step}")
    ratio = range_val / step
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(
            f"{name} step ({step}) 不能整除范围 ({range_val}), "
            f"ratio={ratio:.6f}")


@dataclass
class BaseConfig:
    """公共配置: 频率、角度、阵列位置、激励、单元方向图、优化器。"""

    # 频率
    frequenciesGHz: list = field(default_factory=lambda: [2.45])
    randomSeed: int = 42

    # 角度范围
    theta_start: float = -90.0
    theta_end: float = 90.0
    theta_step: float = 0.1
    theta0s: list = field(default_factory=lambda: [0.0])

    # 阵列位置
    position_source: str = ""
    position_symmetric: bool = False
    position_fixed_aperture: bool = True

    # 幅度
    amplitude_source: str = "default"
    amplitude_bounds: tuple = (0.0, 1.0)
    amplitude_optimize_half: bool = False
    amplitude_step: float = None
    amplitude_in_db: bool = False

    # 相位
    phase_source: str = "optimize"
    phase_optimize_half: bool = False
    phase_step: float = None

    # 单元方向图
    ep_enabled: bool = False
    ep_csv_dir: str = ""
    ep_theta_range: tuple = (-180, 180)
    ep_theta_step: float = 0.01
    ep_is_gain: bool = True
    ep_in_db: bool = False
    ep_aep_mode: bool = False

    # 优化器
    opt_method: str = "cma"
    opt_sigma: float = 0.5
    opt_pop_size: int = None
    opt_max_iter: int = 500
    opt_n_jobs: int = -1
    opt_verbose: bool = True
    opt_stop_fitness: float = None

    # 目标
    target_type: str = "sidelobe"
    target_psll: float = None
    target_hpbw: float = None
    target_gain_drop_limit_db: object = None  # float | [abs, rel] | None
    target_directivity_limit_dbi: object = None  # float | [abs, rel] | None
    target_pointing_penalty: bool = True

    @classmethod
    def from_json(cls, path, base_dir=None):
        """从 Config.json 加载配置。

        Args:
            path: Config.json 路径
            base_dir: 相对路径基准目录 (默认=path 的父目录)
        """
        path = Path(path)
        if base_dir is None:
            base_dir = path.parent
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 解析 aspectAngle
        aa = raw.get("aspectAngle", {})
        theta_deg = aa.get("thetaDeg", [-90, 90, 0.1])

        # 解析 position
        pos = raw.get("position", {})
        pos_source = pos.get("source", "")
        if pos_source and not Path(pos_source).is_absolute():
            pos_source = str(base_dir / pos_source)

        # 解析 amplitude
        amp = raw.get("amplitude", {})
        amp_bounds = tuple(amp.get("bounds", [0.0, 1.0]))

        # 解析 ePattern
        ep = raw.get("ePattern", {})
        ep_dir = ep.get("csvDirectory", "")
        if ep_dir and not Path(ep_dir).is_absolute():
            ep_dir = str(base_dir / ep_dir)
        ep_theta = ep.get("thetaDeg", [-180, 180, 0.01])

        # 解析 optimizer
        opt = raw.get("optimizer", {})

        # 解析 target
        tgt = raw.get("target", {})

        cfg = cls(
            frequenciesGHz=raw.get("frequenciesGHz", [2.45]),
            randomSeed=raw.get("randomSeed", 42),
            theta_start=theta_deg[0],
            theta_end=theta_deg[1],
            theta_step=theta_deg[2],
            theta0s=aa.get("theta0sDeg", [0.0]),
            position_source=pos_source,
            position_symmetric=pos.get("symmetric", False),
            position_fixed_aperture=pos.get("fixedAperture", True),
            amplitude_source=amp.get("source", "default"),
            amplitude_bounds=amp_bounds,
            amplitude_optimize_half=amp.get("optimizeHalf", False),
            amplitude_step=amp.get("step"),
            amplitude_in_db=amp.get("inDB", False),
            phase_source=raw.get("phase", {}).get("source", "optimize"),
            phase_optimize_half=raw.get("phase", {}).get("optimizeHalf", False),
            phase_step=raw.get("phase", {}).get("step"),
            ep_enabled=ep.get("enabled", False),
            ep_csv_dir=ep_dir,
            ep_theta_range=(ep_theta[0], ep_theta[1]),
            ep_theta_step=ep_theta[2],
            ep_is_gain=ep.get("isGain", True),
            ep_in_db=ep.get("inDB", False),
            ep_aep_mode=ep.get("aepMode", False),
            opt_method=opt.get("method", "cma"),
            opt_sigma=opt.get("sigma", 0.5),
            opt_pop_size=opt.get("pop_size"),
            opt_max_iter=opt.get("max_iter", 500),
            opt_n_jobs=opt.get("n_jobs", -1),
            opt_verbose=opt.get("verbose", True),
            opt_stop_fitness=opt.get("stopFitness"),
            target_type=tgt.get("type", "sidelobe"),
            target_psll=tgt.get("targetPSLL"),
            target_hpbw=tgt.get("targetHPBW"),
            target_gain_drop_limit_db=tgt.get("gainDropLimitDb"),
            target_directivity_limit_dbi=tgt.get("directivityLimitDbi"),
            target_pointing_penalty=tgt.get("mainLobePointingPenalty", True),
        )
        # 保存原始 JSON 供子类使用
        cfg._raw = raw

        # 验证 step 能整除 range
        if cfg.amplitude_step:
            _check_step(cfg.amplitude_step, amp_bounds[1] - amp_bounds[0], "amplitude")
        # if cfg.phase_step:
        #     _check_step(cfg.phase_step, 2 * 3.141592653589793, "phase")

        return cfg

    @property
    def theta_range(self):
        return (self.theta_start, self.theta_end, self.theta_step)


@dataclass
class SparseConfig(BaseConfig):
    """线阵稀布幅相优化配置 (继承 Base, 无额外字段)。"""
    pass


@dataclass
class ASMConfig(SparseConfig):
    """渐进空间映射配置: + ASM 迭代参数、PE 子优化器。"""

    asm_max_iterations: int = 10
    asm_target_fine_psll: float = -30.0
    asm_target_res_norm: float = 0.001
    asm_pe_method: str = "cma"
    asm_pe_sigma: float = 0.3
    asm_pe_pop_size: int = None
    asm_pe_max_iter: int = 500
    asm_pe_n_jobs: int = -1
    asm_pe_verbose: bool = True
    asm_pe_stop_fitness: float = 0.001

    @classmethod
    def from_json(cls, path, base_dir=None):
        base = super().from_json(path, base_dir)
        raw = getattr(base, "_raw", {})
        asm = raw.get("asm", {})
        pe = asm.get("pe", {})

        # 基类字段 (排除 ASM 特有字段, 避免重复赋值)
        asm_fields = {f for f in cls.__dataclass_fields__ if f.startswith("asm_")}
        base_kwargs = {f: getattr(base, f) for f in base.__dataclass_fields__
                       if f not in asm_fields}

        return cls(
            **base_kwargs,
            asm_max_iterations=asm.get("maxIterations", 10),
            asm_target_fine_psll=asm.get("targetFinePSLL", -30.0),
            asm_target_res_norm=asm.get("targetResNorm", 0.001),
            asm_pe_method=pe.get("method", "cma"),
            asm_pe_sigma=pe.get("sigma", 0.3),
            asm_pe_pop_size=pe.get("pop_size"),
            asm_pe_max_iter=pe.get("max_iter", 500),
            asm_pe_n_jobs=pe.get("n_jobs", -1),
            asm_pe_verbose=pe.get("verbose", True),
            asm_pe_stop_fitness=pe.get("stopFitness", 0.001),
        )

    @property
    def pe_config(self):
        """返回 PE 参数字典 (供 space_mapping 使用)。"""
        return {
            "method": self.asm_pe_method,
            "sigma": self.asm_pe_sigma,
            "pop_size": self.asm_pe_pop_size,
            "max_iter": self.asm_pe_max_iter,
            "n_jobs": self.asm_pe_n_jobs,
            "verbose": self.asm_pe_verbose,
            "stopFitness": self.asm_pe_stop_fitness,
        }


@dataclass
class MSMConfig(SparseConfig):
    """流形空间映射配置 (预留 MSM 特有参数)。"""
    pass
