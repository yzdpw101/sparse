"""新版优化器 — 统一接口, 单目标最小化, 可配置超参数。

用法:
  from antopt.optimizer2 import minimize

  # CMA-ES
  result = minimize(fitness, n_vars, method="cma", sigma=0.5,
                    max_iter=500, n_jobs=-1, verbose=True)

  # DE (DMDE)
  result = minimize(fitness, n_vars, method="de", CR=0.9,
                    max_iter=500, n_jobs=-1)

  # GWO
  result = minimize(fitness, n_vars, method="gwo", max_iter=300)

超参数:
  CMA — sigma, CMA_diagonal, CMA_active, CMA_elitist
  DE  — CR, F_min_early, F_max_early, F_min_late, F_max_late, stall_limit
  GWO — (暂无特殊超参数)
"""

import os, json, warnings
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import Optional, Callable, Union
import numpy as np


def _is_positions_symmetric(positions, tol=1e-6):
    """检查位置是否关于原点对称 (x[i] + x[N-1-i] ≈ 0)。"""
    n = len(positions)
    for i in range(n // 2):
        if abs(positions[i] + positions[n - 1 - i]) > tol:
            return False
    return True


# ═══════════════════════════════════════════════
def _workers(n_jobs):
    if n_jobs is None or n_jobs == 0 or n_jobs == 1:
        return 0
    m = os.cpu_count() or 4
    if n_jobs < 0:
        return m
    return min(n_jobs, m)


def _vprint(verbose, max_iter):
    if verbose is None or verbose is False:
        return 0
    if verbose is True:
        return max(1, max_iter // 10)
    v = int(verbose)
    return 0 if v <= 0 else v


def _tent(D, seed):
    rng = np.random.default_rng(seed)
    x = np.empty(D)
    z = rng.uniform(0, 1)
    for i in range(D):
        z = 2.0 * min(z, 1.0 - z)
        z = np.clip(z, 1e-10, 1.0 - 1e-10)
        x[i] = z
    return x


def _init(D, seed, x0, init_mode, bounds):
    if x0 is not None:
        return np.asarray(x0, dtype=float).copy()
    rng = np.random.default_rng(seed)
    lo, hi = (-1.0, 1.0) if bounds is None else bounds
    base = _tent(D, seed) if init_mode == "chaos" else rng.uniform(0, 1, size=D)
    return lo + (hi - lo) * base


# ═══════════════════════════════════════════════
from .mapping import LMMapper
from .pattern import Pattern
from .analysis import get_psll, _find_peaks

TWO_PI = 2 * np.pi
_SIGMOID_CLIP = 20.0


class BeamformingProblem:
    """波束成形优化基类 — 统一处理位置/相位/幅度的解码与 AF 计算。

    子类只需实现 fitness(x) 来定义具体的优化目标。
    """

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        *,
        position_source: Union[str, bool] = True,
        amplitude_source: Union[str, bool] = "default",
        amplitude_bounds: tuple[float, float] = (0.0, 1.0),
        amplitude_in_db: bool = False,
        amplitude_step: Optional[float] = None,
        phase_step_deg: Optional[float] = None,
        phase_source: Union[str, bool] = "default",
        optimize_half_amp: bool = False,
        optimize_half_phase: bool = False,
        init_positions: Optional[np.ndarray] = None,
        init_phases_deg: Optional[np.ndarray] = None,
        init_amplitudes: Optional[np.ndarray] = None,
        element_patterns: Optional[list] = None,
        use_pointing_penalty: bool = True,
    ):
        self.mapper = mapper
        self.pattern = pattern
        self.element_patterns = element_patterns
        self.use_pointing_penalty = use_pointing_penalty
        self.amplitude_lower, self.amplitude_upper = amplitude_bounds
        self.amplitude_in_db = amplitude_in_db
        self._amp_step = amplitude_step  # dB (if in_db) 或线性步长
        self._phs_step_rad = np.deg2rad(phase_step_deg) if phase_step_deg else None

        # ── 解析 position_source ──
        if position_source is True:
            self._p_mode = 1  # 优化
            self.init_positions = None
        elif isinstance(position_source, str):
            self._p_mode = 2  # 导入
            self.init_positions = init_positions
        elif isinstance(position_source, (np.ndarray, list)):
            # 直接传入位置数组, 固定不动
            self._p_mode = 2  # 导入 (不优化位置)
            self.init_positions = np.asarray(position_source, dtype=float)
        else:
            raise ValueError(f"position.source 必须是 true/文件路径/数组, 得到 {position_source!r}")

        # ── 解析 amplitude_source ──
        if amplitude_source is True or amplitude_source == "optimize":
            self._a_mode = 1  # 优化
            self.init_amplitudes = None
        elif amplitude_source == "default":
            self._a_mode = 0  # 默认全1
            self.init_amplitudes = None
        elif isinstance(amplitude_source, str):
            self._a_mode = 2  # 导入
            self.init_amplitudes = init_amplitudes
        else:
            raise ValueError(f"amplitude.source 必须是 optimize/default/文件路径, 得到 {amplitude_source!r}")

        # ── 解析 phase_source ──
        if phase_source is True or phase_source == "optimize":
            self._h_mode = 1  # 优化
            self.init_phases_deg = None
        elif phase_source == "default":
            self._h_mode = 0  # 默认全0
            self.init_phases_deg = None
        elif isinstance(phase_source, str):
            self._h_mode = 2  # 导入
            self.init_phases_deg = init_phases_deg
        else:
            raise ValueError(f"phase.source 必须是 optimize/default/文件路径, 得到 {phase_source!r}")

        # ── 对称半优化校验 ──
        _pos_sym = self._check_position_symmetry(mapper, init_positions, self._p_mode)

        if optimize_half_amp:
            if not _pos_sym:
                warnings.warn("optimizeHalfAmp=True 但位置不对称，已忽略")
                optimize_half_amp = False
            elif self._a_mode != 1:
                warnings.warn("optimizeHalfAmp=True 但无幅度变量可优化，已忽略")
                optimize_half_amp = False

        if optimize_half_phase:
            if not _pos_sym:
                warnings.warn("optimizeHalfPhase=True 但位置不对称，已忽略")
                optimize_half_phase = False
            elif self._h_mode != 1:
                warnings.warn("optimizeHalfPhase=True 但无相位变量可优化，已忽略")
                optimize_half_phase = False

        self.optimize_half_amp = optimize_half_amp
        self.optimize_half_phase = optimize_half_phase
        # 对称 AF 公式要求幅度和相位都用 half
        self._use_symmetric_af = (self.optimize_half_amp and self.optimize_half_phase)

        # ── 变量维度 ──
        self.n_pos = mapper.n_vars if self._p_mode == 1 else 0
        self.n_phase = self._half_n() if self._h_mode == 1 and self.optimize_half_phase else (
            mapper.Ne if self._h_mode == 1 else 0)
        self.n_amp = self._half_n() if self._a_mode == 1 and self.optimize_half_amp else (
            mapper.Ne if self._a_mode == 1 else 0)
        self.n_vars = self.n_pos + self.n_phase + self.n_amp

        # ── 主瓣指向索引 ──
        self._main_idx = int(np.argmin(np.abs(self.pattern.theta_deg - self.pattern.theta0s_deg[0])))

    def _half_n(self) -> int:
        """对称半阵元变量数: halfNe + has_center。"""
        return self.mapper._halfNe + (1 if self.mapper.has_center else 0)

    @staticmethod
    def _check_position_symmetry(mapper, init_positions, p_mode):
        """检查位置是否对称。"""
        if p_mode == 1:
            return mapper.is_symmetric
        elif p_mode == 2:
            if init_positions is not None:
                return _is_positions_symmetric(init_positions)
            return mapper.is_symmetric
        return False

    def _expand_half(self, half_vals):
        """半阵元变量 → 全阵元数组 (镜像左半边)。"""
        hN = self.mapper._halfNe
        has_c = self.mapper.has_center
        full = np.empty(self.mapper.Ne)
        if has_c:
            full[hN] = half_vals[0]
            right = half_vals[1:]
        else:
            right = half_vals
        full[hN + (1 if has_c else 0):] = right
        full[:hN] = right[::-1]
        return full

    def _decode(self, x: np.ndarray):
        """解码变量向量 → (pos, phases_rad, amplitudes)。"""
        x = np.asarray(x, dtype=float)
        Ne = self.mapper.Ne

        # 位置
        if self._p_mode == 1:
            pos = self.mapper.synthesize(x[:self.n_pos])
        else:
            pos = self.init_positions

        # 相位
        if self._h_mode == 1:
            raw = x[self.n_pos:self.n_pos + self.n_phase]
            if self._phs_step_rad is not None:
                raw = np.round(raw / self._phs_step_rad) * self._phs_step_rad
            phases = self._expand_half(raw) if self.optimize_half_phase else raw
        elif self._h_mode == 2:
            phases = np.deg2rad(self.init_phases_deg)
        else:
            phases = np.zeros(Ne)

        # 幅度
        if self._a_mode == 1:
            raw = x[-self.n_amp:]
            if self._amp_step is not None:
                raw = np.round(raw / self._amp_step) * self._amp_step
            amps = self._expand_half(raw) if self.optimize_half_amp else raw
            if self.amplitude_in_db:
                amps = np.power(10.0, -amps / 20.0)
        elif self._a_mode == 2:
            amps = self.init_amplitudes
        else:
            amps = np.ones(Ne)

        return pos, phases, amps

    def _compute_af(self, pos, amps, phases):
        """计算 AF × 单元方向图, 返回复数 AF。"""
        if self.pattern.is_planar:
            af = self.pattern.planar_af(pos, np.zeros_like(pos), amps, phases)
        elif self._use_symmetric_af:
            halfNe = self.mapper._halfNe
            offset = 1 if self.mapper.has_center else 0
            kwargs = dict(has_center=self.mapper.has_center,
                          amplitudes=amps[halfNe + offset:],
                          phases=phases[halfNe + offset:])
            if self.mapper.has_center:
                kwargs["center_amplitude"] = amps[halfNe]
                kwargs["center_phase"] = phases[halfNe]
            af = self.pattern.linear_af_symmetric(pos[halfNe + offset:], **kwargs)
        else:
            af = self.pattern.linear_af(pos, amps, phases)

        if self.element_patterns is not None:
            ep = self.element_patterns
            # AEP 模式检测: ep[0] 是 list 时为逐元方向图
            _is_aep = isinstance(ep, list) and len(ep) > 0 and isinstance(ep[0], list)

            if _is_aep:
                # AEP: E(θ) = Σ Fe_n(θ) · A_n · exp(j·k·x_n·Δsin)
                # ep[freq_idx][elem_idx] → Fe_n(θ), shape (Nθ,)
                pos_arr = np.asarray(pos, dtype=float)
                amps_arr = np.asarray(amps, dtype=float)
                phases_arr = np.asarray(phases, dtype=float)
                Ne = len(pos_arr)

                if af.ndim == 1:
                    af_new = np.zeros_like(af, dtype=complex)
                    for n in range(Ne):
                        exc_n = amps_arr[n] * np.exp(1j * phases_arr[n])
                        phase_n = TWO_PI * pos_arr[n] * self.pattern._delta_sin
                        af_new += ep[0][n] * exc_n * np.exp(1j * phase_n)
                    af = af_new
                else:
                    af_new = np.zeros_like(af, dtype=complex)
                    for n in range(Ne):
                        exc_n = amps_arr[n] * np.exp(1j * phases_arr[n])
                        phase_n = TWO_PI * pos_arr[n] * self.pattern._delta_sin
                        af_new += ep[0][n][np.newaxis, :] * exc_n * np.exp(1j * phase_n)
                    af = af_new
            else:
                # 传统模式: |AF(θ)| × Fe(θ), 所有单元同一个方向图
                af_mag = np.abs(af)
                if af.ndim == 1:
                    af = af_mag * ep[0]
                else:
                    for i in range(len(ep)):
                        af[i] = af_mag[i] * ep[i]
        return af

    def _get_af_db(self, x):
        """解码变量 + 计算 AF + 转 dB, 返回 af_db (可能多维)。"""
        pos, phases, amps = self._decode(x)
        af = self._compute_af(pos, amps, phases)
        return Pattern.to_dB(af)

    def _flatten_af_db(self, af_db):
        """将 af_db 展为 (Nsub, Nθ)。"""
        if af_db.ndim > 1:
            return af_db.reshape(-1, af_db.shape[-1])
        return af_db[None, :]

    def fitness(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def get_result(self, x_opt: np.ndarray) -> dict:
        x = np.asarray(x_opt, dtype=float)
        pos, phases, amps = self._decode(x)
        r = {"positions": pos}
        r["phases_deg"] = np.rad2deg(phases)
        r["amplitudes"] = amps  # 始终是线性值 (供 AF 计算)

        if self.amplitude_in_db and self._a_mode == 1:
            # dB 值单独保存 (需量化到网格)
            raw_amp = x[-self.n_amp:]
            if self._amp_step is not None:
                raw_amp = np.round(raw_amp / self._amp_step) * self._amp_step
            if self.optimize_half_amp:
                raw_amp = self._expand_half(raw_amp)
            r["amplitudes_dB"] = raw_amp

        return r


class SidelobeProblem(BeamformingProblem):
    """副瓣电平优化问题。

    适应度: psll + pointing + hpbw + gain_drop + directivity

    limit 参数格式:
      - None: 不限制
      - float: 绝对值限制
      - [abs, rel]: 绝对 + 相对限制 (None 表示该维度不限)
    """

    @staticmethod
    def _parse_limit(val):
        """解析 limit 参数 → (abs_val, rel_val)。

        格式:
          None → (None, None)
          ["abs", 17.0] → (17.0, None)  绝对值
          ["rel", 2.0] → (None, 2.0)    相对值 (和全1激励对比)
        """
        if val is None:
            return None, None
        if isinstance(val, (list, tuple)) and len(val) == 2:
            mode, value = val[0], val[1]
            if mode == "abs":
                return float(value), None
            elif mode == "rel":
                return None, float(value)
        return None, None

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        *,
        target_psll: Optional[float] = None,
        target_hpbw: float = 180.0,
        gain_drop_limit_db=None,  # float | [abs, rel] | None
        directivity_limit_dbi=None,  # float | [abs, rel] | None
        mainlobe_region: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(mapper, pattern, **kwargs)
        self.target_psll = target_psll
        self.target_hpbw = target_hpbw
        self.mainlobe_region = mainlobe_region

        # 解析 limit 格式: float → [float, None], list → [abs, rel], None → [None, None]
        self._gd_abs, self._gd_rel = self._parse_limit(gain_drop_limit_db)
        self._dir_abs, self._dir_rel = self._parse_limit(directivity_limit_dbi)

        # 计算参考值 (全1幅度, 全0相位)
        self._ref_main_gain_db = None
        self._ref_directivity_dbi = None
        if self._gd_rel is not None or self._dir_rel is not None:
            ref_amps = np.ones(mapper.Ne)
            ref_phases = np.zeros(mapper.Ne)
            ref_af = self._compute_af(self.init_positions if self._p_mode == 2 else mapper.synthesize(np.zeros(mapper.n_vars)),
                                      ref_amps, ref_phases)
            ref_abs = np.abs(ref_af)
            ref_flat = ref_abs.reshape(-1, ref_abs.shape[-1]) if ref_abs.ndim > 1 else ref_abs[None, :]
            if self._gd_rel is not None:
                self._ref_main_gain_db = float(20 * np.log10(np.max(ref_flat[0]) + 1e-30))
            if self._dir_rel is not None:
                from .analysis import compute_directivity
                # 参考方向性系数: 用 _compute_af (含单元方向图)
                ref_af_full = self._compute_af(
                    self.init_positions if self._p_mode == 2 else mapper.synthesize(np.zeros(mapper.n_vars)),
                    ref_amps, ref_phases)
                ref_abs_full = np.abs(ref_af_full)
                ref_flat_full = ref_abs_full.reshape(-1, ref_abs_full.shape[-1]) if ref_abs_full.ndim > 1 else ref_abs_full[None, :]
                t_step = float(abs(self.pattern.theta_deg[1] - self.pattern.theta_deg[0]))
                self._ref_directivity_dbi, _ = compute_directivity(
                    ref_flat_full[0].transpose(),
                    [float(self.pattern.theta_deg[0]), float(self.pattern.theta_deg[-1]), t_step],
                    [0, 0, 1])

    def fitness(self, x: np.ndarray) -> float:
        af_db = self._get_af_db(x)
        af_flat = self._flatten_af_db(af_db)

        # PSLL
        if self.pattern.is_planar:
            pslls, _ = get_psll(af_db, self.pattern.theta_deg, self.pattern.phi_deg,
                                mainlobe_region=self.mainlobe_region)
            valid = ~np.isinf(pslls)
            psll_val = float(np.max(pslls[valid])) if valid.any() else 0.0
        else:
            mr_1d = None
            if (isinstance(self.mainlobe_region, tuple) and len(self.mainlobe_region) == 2
                    and isinstance(self.mainlobe_region[0], (int, float, np.floating))):
                mr_1d = self.mainlobe_region
            if af_db.ndim == 1:
                psll_val, _ = get_psll(af_db, self.pattern.theta_deg, mainlobe_region=mr_1d)
            else:
                psll_val = -np.inf
                for sub in range(af_flat.shape[0]):
                    v, _ = get_psll(af_flat[sub], self.pattern.theta_deg, mainlobe_region=mr_1d)
                    if not np.isinf(v) and v > psll_val:
                        psll_val = v

        # 主瓣指向惩罚
        pointing = 0.0
        if self.use_pointing_penalty and not self.pattern.is_planar:
            from .analysis import _find_peaks
            for sub in range(af_flat.shape[0]):
                pattern_1d = af_flat[sub]
                scan_idx = sub % len(self.pattern.theta0s_deg)
                target_idx = int(np.argmin(
                    np.abs(self.pattern.theta_deg - self.pattern.theta0s_deg[scan_idx])))
                indices, extrema = _find_peaks(pattern_1d)
                main_idx = indices[0]
                for j in range(1, len(indices)):
                    if abs(extrema[j]) < 1e-6:
                        cur = indices[j]
                        if abs(cur - target_idx) < abs(main_idx - target_idx):
                            main_idx = cur
                    else:
                        break
                pointing = max(pointing, abs(main_idx - target_idx) * 100.0)

        # HPBW 惩罚
        hpbw_pen = 0.0
        if self.target_hpbw < 180.0:
            hpbw = self._compute_hpbw(af_db)
            if hpbw > self.target_hpbw:
                hpbw_pen = (hpbw - self.target_hpbw) * 100.0

        # 主瓣增益下降惩罚
        gain_drop_pen = 0.0
        if (self._gd_abs is not None or self._gd_rel is not None) and self._ref_main_gain_db is not None:
            pos, phases, amps = self._decode(x)
            af_abs = np.abs(self._compute_af(pos, amps, phases))
            af_abs_flat = af_abs.reshape(-1, af_abs.shape[-1]) if af_abs.ndim > 1 else af_abs[None, :]
            cur_main_gain = float(20 * np.log10(np.max(af_abs_flat[0]) + 1e-30))
            drop = self._ref_main_gain_db - cur_main_gain
            # 绝对值: drop 不能超过 limit
            if self._gd_abs is not None and drop > self._gd_abs:
                gain_drop_pen += (drop - self._gd_abs) * 100.0
            # 相对: drop 不能超过 ref_gain - limit
            if self._gd_rel is not None and drop > self._gd_rel:
                gain_drop_pen += (drop - self._gd_rel) * 100.0

        # 方向性系数约束
        dir_pen = 0.0
        if self._dir_abs is not None or self._dir_rel is not None:
            from .analysis import compute_directivity
            pos_d, phases_d, amps_d = self._decode(x)
            af_abs_d = np.abs(self._compute_af(pos_d, amps_d, phases_d))
            af_abs_d = af_abs_d.reshape(-1, af_abs_d.shape[-1]) if af_abs_d.ndim > 1 else af_abs_d[None, :]
            t_step = float(abs(self.pattern.theta_deg[1] - self.pattern.theta_deg[0]))
            d_val, _ = compute_directivity(af_abs_d[0],
                (float(self.pattern.theta_deg[0]), float(self.pattern.theta_deg[-1]), t_step),
                [0, 0.5, 1])
            # 绝对值: D 不能低于 limit
            if self._dir_abs is not None and d_val < self._dir_abs:
                dir_pen += (self._dir_abs - d_val) * 100.0
            # 相对: D 不能低于 ref_D - limit
            if self._dir_rel is not None and self._ref_directivity_dbi is not None:
                rel_drop = self._ref_directivity_dbi - d_val
                if rel_drop > self._dir_rel:
                    dir_pen += (rel_drop - self._dir_rel) * 100.0

        return float(psll_val) + pointing + hpbw_pen + gain_drop_pen + dir_pen

    def _compute_hpbw(self, af_db):
        if self.pattern.is_planar:
            j0 = np.argmin(np.abs(self.pattern.phi_deg))
            pattern_1d = af_db[:, j0]
        else:
            pattern_1d = af_db
        peak_idx = int(np.argmax(pattern_1d))
        left = pattern_1d[:peak_idx + 1]
        right = pattern_1d[peak_idx:]
        left_idx = int(np.argmin(np.abs(left - (-3.0))))
        right_idx = int(np.argmin(np.abs(right - (-3.0)))) + peak_idx
        dtheta = abs(self.pattern.theta_deg[1] - self.pattern.theta_deg[0])
        return (right_idx - left_idx) * dtheta


class NullSteeringProblem(BeamformingProblem):
    """零陷优化问题。

    适应度: w_ml·pointing + w_null·depth + w_valley·valley + w_sll·psll
    """

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        *,
        null_angle_deg: float = 80.0,
        null_target_db: float = -40.0,
        sll_target_db: float = -15.0,
        null_window_half_deg: float = 5.0,
        null_margin_db: float = 0.0,
        weights: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(mapper, pattern, **kwargs)

        self.null_angle_deg = float(null_angle_deg)
        self.null_target_db = float(null_target_db)
        self.sll_target_db = float(sll_target_db)
        self.null_window_half_deg = float(null_window_half_deg)
        self.null_margin_db = float(null_margin_db)

        w = weights or {}
        self.w_ml = float(w.get("pointing", 0.5))
        self.w_null = float(w.get("null_depth", 3.0))
        self.w_valley = float(w.get("valley", 1.0))
        self.w_sll = float(w.get("sll", 5.0))

        # 零陷窗口索引
        self._null_idx = int(np.argmin(np.abs(self.pattern.theta_deg - self.null_angle_deg)))
        hw = int(round(self.null_window_half_deg / abs(self.pattern.theta_deg[1] - self.pattern.theta_deg[0])))
        self._null_win = slice(max(0, self._null_idx - hw),
                               min(len(self.pattern.theta_deg), self._null_idx + hw + 1))

    def fitness(self, x: np.ndarray) -> float:
        af_db = self._get_af_db(x)
        af_flat = self._flatten_af_db(af_db)

        total_fit = 0.0
        for sub in range(af_flat.shape[0]):
            p = af_flat[sub]

            # PSLL
            psll_val, _ = get_psll(p, self.pattern.theta_deg)
            psll_pt = max(0.0, psll_val - self.sll_target_db)

            # 主瓣指向惩罚
            pointing = 0.0
            if self.use_pointing_penalty:
                from .analysis import _find_peaks
                indices, extrema = _find_peaks(p)
                main_idx = indices[0]
                for j in range(1, len(indices)):
                    if abs(extrema[j]) < 1e-6:
                        cur = indices[j]
                        if abs(cur - self._main_idx) < abs(main_idx - self._main_idx):
                            main_idx = cur
                    else:
                        break
                pointing = abs(main_idx - self._main_idx) * 0.5

            # 零陷深度惩罚
            min_null = np.max(p[self._null_win])
            depth_pen = max(0.0, min_null - self.null_target_db)

            # 谷形质量惩罚
            win = self._null_win
            edge_val = min(p[win.start] if win.start > 0 else p[0],
                           p[win.stop - 1] if win.stop < len(p) else p[-1])
            valley_depth = p[self._null_idx] - edge_val
            valley_pen = max(0.0, valley_depth + self.null_margin_db)

            total_fit += (self.w_ml * pointing
                          + self.w_null * depth_pen
                          + self.w_valley * valley_pen
                          + self.w_sll * psll_pt)

        return float(total_fit)


class TripleNullProblem(BeamformingProblem):
    """三零陷优化问题。

    适应度: pointing + 2·depth_penalty + 2·PSLL_penalty + edge_dip_penalty
    参考: f5.m (孟师兄三零陷验证)
    """

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        *,
        null_angles_deg: list = None,
        null_target_db: float = -40.0,
        sll_target_db: float = -15.0,
        null_window_half_deg: float = 5.0,
        null_margin_db: float = 0.0,
        weights: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(mapper, pattern, **kwargs)

        self.null_angles_deg = [float(a) for a in (null_angles_deg or [55.0, 70.0, 85.0])]
        self.null_target_db = float(null_target_db)
        self.sll_target_db = float(sll_target_db)
        self.null_window_half_deg = float(null_window_half_deg)
        self.null_margin_db = float(null_margin_db)

        w = weights or {}
        self.w_ml = float(w.get("pointing", 0.5))
        self.w_null = float(w.get("null_depth", 2.0))
        self.w_sll = float(w.get("sll", 2.0))
        self.w_edge = float(w.get("edge_dip", 1.0))

        # 预计算每个零陷的窗口索引
        step = abs(self.pattern.theta_deg[1] - self.pattern.theta_deg[0])
        hw = int(round(self.null_window_half_deg / step))
        self._null_infos = []
        for ang in self.null_angles_deg:
            idx = int(np.argmin(np.abs(self.pattern.theta_deg - ang)))
            win = slice(max(0, idx - hw), min(len(self.pattern.theta_deg), idx + hw + 1))
            self._null_infos.append((idx, win))

    def fitness(self, x: np.ndarray) -> float:
        af_db = self._get_af_db(x)
        af_flat = self._flatten_af_db(af_db)

        total_fit = 0.0
        for sub in range(af_flat.shape[0]):
            p = af_flat[sub]

            # PSLL (副瓣电平)
            psll_val, _ = get_psll(p, self.pattern.theta_deg)
            psll_pt = max(0.0, psll_val - self.null_target_db)

            # 主瓣指向惩罚
            pointing = 0.0
            if self.use_pointing_penalty:
                from .analysis import _find_peaks
                indices, extrema = _find_peaks(p)
                main_idx = indices[0]
                for j in range(1, len(indices)):
                    if abs(extrema[j]) < 1e-6:
                        cur = indices[j]
                        if abs(cur - self._main_idx) < abs(main_idx - self._main_idx):
                            main_idx = cur
                    else:
                        break
                pointing = abs(main_idx - self._main_idx) * 0.5

            # 三个零陷的深度惩罚 + 边缘下陷惩罚
            total_depth_pen = 0.0
            total_edge_pen = 0.0
            for null_idx, null_win in self._null_infos:
                # 深度惩罚: 窗口内最大值 - 目标
                min_null = np.max(p[null_win])
                depth_pen = max(0.0, min_null - self.null_target_db)
                total_depth_pen = max(total_depth_pen, depth_pen)

                # 边缘下陷惩罚: 零陷中心是否比窗口边缘低
                edge_val = min(p[null_win.start] if null_win.start > 0 else p[0],
                               p[null_win.stop - 1] if null_win.stop < len(p) else p[-1])
                edge_pen = max(0.0, p[null_idx] - edge_val + self.null_margin_db)
                total_edge_pen += edge_pen

            total_fit += (self.w_ml * pointing
                          + self.w_null * total_depth_pen
                          + self.w_sll * psll_pt
                          + self.w_edge * total_edge_pen)

        return float(total_fit)


# 向后兼容别名
SparseArrayProblem = SidelobeProblem
class Base(ABC):
    """优化器基类。子类在 __init__ 中接收自己的超参数。"""

    def __init__(self, fitness, n_vars, bounds=None, n_jobs=0, verbose=False,
                 seed=42, stop_fitness=None, max_iter=500, pop_size=None,
                 x0=None, init="chaos", checkpoint=None, resume=None):
        self._fit = fitness
        self._D = n_vars
        self._bounds = bounds
        self._nw = _workers(n_jobs)
        self._vi = _vprint(verbose, max_iter)
        self._seed = seed
        self._stop = stop_fitness
        self._G = max_iter
        self._ps = pop_size
        self._x0 = x0
        self._init = init
        self._ckpt = checkpoint
        self._resume = resume
        self._pool = None

    def _eval(self, X):
        if self._nw > 0:
            if self._pool is None:
                self._pool = ThreadPool(self._nw)
            return np.array(self._pool.map(self._fit, X))
        return np.array([self._fit(x) for x in X])

    def _log(self, t, bf, **kw):
        if self._vi > 0 and (t + 1) % self._vi == 0:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in kw.items())
            print(f"Iter {t+1:>4d}: best={bf:.4f}  {parts}".strip())

    def _done(self, bf):
        return self._stop is not None and bf <= self._stop

    def _save(self, t, bx, bf):
        if self._ckpt is None:
            return
        with open(self._ckpt, "w") as f:
            json.dump({"iter": t, "x": bx.tolist(), "f": float(bf)}, f)

    def _load(self):
        if self._resume is None:
            return None
        try:
            with open(self._resume) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _cleanup(self):
        if self._pool:
            self._pool.terminate()
            self._pool = None

    @abstractmethod
    def run(self) -> dict: ...


# ═══════════════════════════════════════════════
class CMA(Base):
    """C++ 风格 CMA-ES。

    超参数 (通过 **kwargs 传入):
      sigma=0.3         初始步长
      CMA_diagonal=True D>30 时用对角协方差
      CMA_active=True    激活 CMA
      CMA_elitist=False  精英策略
    """

    def __init__(self, fitness, n_vars, **kwargs):
        self._sigma = kwargs.pop("sigma", 0.3)
        self._cma_diag = kwargs.pop("CMA_diagonal", True)
        self._cma_active = kwargs.pop("CMA_active", True)
        self._cma_elitist = kwargs.pop("CMA_elitist", False)
        super().__init__(fitness, n_vars, **kwargs)

    def _dp(self):
        if self._ps is None or self._ps <= 0:
            return 2 * (4 + int(3.0 * np.log(self._D)))
        n = int(self._ps)
        return n + 1 if n % 2 == 1 else n  # CMA 要求偶数

    def run(self) -> dict:
        D = self._D
        lam = self._dp()
        mu = lam // 2
        rng = np.random.default_rng(self._seed)
        bounded = self._bounds is not None

        ckpt = self._load()
        if ckpt:
            xm = np.array(ckpt["x"]); bx, bf = xm.copy(), ckpt["f"]
            s0 = ckpt["iter"] + 1
        else:
            xm = _init(D, self._seed, self._x0, self._init, self._bounds)
            bx, bf = xm.copy(), self._fit(xm)
            s0 = 0

        w = np.log(mu + 0.5) - np.log(np.arange(1.0, mu + 1.0)); w /= w.sum()
        mueff = 1.0 / (w**2).sum()
        cc = (4 + mueff/D) / (D + 4 + 2*mueff/D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2.0 / ((D + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((D + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff - 1)/(D + 1)) - 1) + cs
        pc = np.zeros(D); ps = np.zeros(D)
        B = np.eye(D); dd = np.ones(D); C = np.eye(D); iC = np.eye(D)
        chiN = np.sqrt(D)*(1 - 1/(4*D) + 1/(21*D*D))
        sigma = self._sigma; ee = ce = 0

        try:
            for t in range(s0, self._G):
                arx = np.empty((lam, D))
                for k in range(lam):
                    arx[k] = xm + sigma * (B @ (dd * rng.standard_normal(D)))
                    if bounded:
                        arx[k] = np.clip(arx[k], *self._bounds)
                af = self._eval(arx); ce += lam
                idx = np.argsort(af)
                if af[idx[0]] < bf: bf = af[idx[0]]; bx = arx[idx[0]].copy()
                self._log(t, bf, sigma=sigma)
                self._save(t, bx, bf)
                if self._done(bf): break

                xo = xm.copy(); xm.fill(0)
                for j in range(mu): xm += w[j] * arx[idx[j]]
                ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (iC @ (xm-xo)) / sigma
                pc = (1-cc)*pc + np.sqrt(cc*(2-cc)*mueff) * (xm-xo) / sigma
                at = np.array([(arx[idx[j]] - xo)/sigma for j in range(mu)]).T
                C = (1-c1-cmu)*C + c1*np.outer(pc,pc) + cmu*(at @ np.diag(w) @ at.T)
                sigma *= np.exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
                if (ce - ee) > (lam/(c1+cmu)/D/10):
                    ee = ce; Cs = (C + C.T)*0.5
                    try:
                        ev, eig = np.linalg.eigh(Cs); ev = np.maximum(ev, 0)
                        fl = max(1e-15, 1e-8*max(1, np.max(ev)))
                        dd = np.sqrt(np.maximum(ev, fl))
                        iD = np.where(dd > fl, 1/dd, 1/fl)
                        B = eig; iC = B @ np.diag(iD) @ B.T; C = Cs
                    except np.linalg.LinAlgError:
                        B = np.eye(D); dd = np.ones(D); iC = np.eye(D)
        finally:
            self._cleanup()
        return {"x": bx, "f": bf, "method": "cma"}


# ═══════════════════════════════════════════════
class DE(Base):
    """DMDE 双变异 DE。

    超参数:
      CR=1.0             交叉概率
      F_min_early=0.65   前期 F 下限
      F_max_early=0.90   前期 F 上限
      F_min_late=0.40    后期 F 下限
      F_max_late=0.65    后期 F 上限
      stall_limit=5      停滞代数触发 Cauchy-Gaussian
    """

    def __init__(self, fitness, n_vars, **kwargs):
        self._CR = kwargs.pop("CR", 1.0)
        self._Fe = (kwargs.pop("F_min_early", 0.65), kwargs.pop("F_max_early", 0.90))
        self._Fl = (kwargs.pop("F_min_late", 0.40), kwargs.pop("F_max_late", 0.65))
        self._sl = kwargs.pop("stall_limit", 5)
        super().__init__(fitness, n_vars, **kwargs)

    def _dp(self):
        return self._ps or 50

    def run(self) -> dict:
        D = self._D; G = self._G; NP = self._dp()
        rng = np.random.default_rng(self._seed)
        bounded = self._bounds is not None
        pop = np.array([_init(D, self._seed + i,
                             self._x0 if i == 0 else None,
                             self._init, self._bounds) for i in range(NP)])
        try:
            fit = self._eval(pop)
            bi = int(np.argmin(fit)); bx = pop[bi].copy(); bf = fit[bi]; stall = 0
            for t in range(G):
                Fm, FM = self._Fe if t < G//2 else self._Fl
                fm, fM = fit.min(), fit.max(); d = max(fM - fm, 1e-15)
                np_new = np.empty((NP, D))
                for i in range(NP):
                    Fi = np.clip(Fm + (FM-Fm)*(fit[i]-fm)/d, Fm, FM)
                    cand = [j for j in range(NP) if j != i]
                    r1, r2 = rng.choice(cand, 2, replace=False)
                    v = bx + Fi * (pop[r1] - pop[r2])
                    u = pop[i].copy(); jr = rng.integers(0, D)
                    for j in range(D):
                        if rng.uniform() <= self._CR or j == jr: u[j] = v[j]
                    np_new[i] = np.clip(u, *self._bounds) if bounded else u
                nf = self._eval(np_new)
                improved = False
                for i in range(NP):
                    if nf[i] <= fit[i]:
                        pop[i] = np_new[i]; fit[i] = nf[i]
                        if nf[i] < bf: bf = nf[i]; bx = np_new[i].copy(); improved = True
                stall = 0 if improved else stall + 1
                if stall >= self._sl:
                    stall = 0; b1 = 1 - t/G; b2 = t/G
                    for i in range(NP):
                        tr = pop[i] * (1 + b1*rng.standard_cauchy(D) + b2*rng.standard_normal(D))
                        if bounded: tr = np.clip(tr, *self._bounds)
                        tf = self._eval(tr.reshape(1, -1))[0]
                        if tf < fit[i]: pop[i] = tr; fit[i] = tf
                        if tf < bf: bf = tf; bx = tr.copy()
                self._log(t, bf, stall=stall)
                self._save(t, bx, bf)
                if self._done(bf): break
        finally:
            self._cleanup()
        return {"x": bx, "f": bf, "method": "de"}


# ═══════════════════════════════════════════════
class GWO(Base):
    """Grey Wolf Optimizer。"""

    def _dp(self):
        return self._ps or (2 * (4 + int(3.0 * np.log(self._D))))

    def run(self) -> dict:
        D = self._D; G = self._G; N = self._dp()
        rng = np.random.default_rng(self._seed)
        bounded = self._bounds is not None
        wolves = np.array([_init(D, self._seed + i,
                                 self._x0 if i == 0 else None,
                                 self._init, self._bounds) for i in range(N)])
        try:
            fit = self._eval(wolves)
            idx = np.argsort(fit)
            a_pos = wolves[idx[0]].copy(); b_pos = wolves[idx[1]].copy()
            d_pos = wolves[idx[2]].copy(); af = fit[idx[0]]; bf = af
            for t in range(G):
                a = 2.0 * (1.0 - t/G)
                for i in range(N):
                    X1 = X2 = X3 = np.zeros(D)
                    for j in range(D):
                        r1, r2 = rng.uniform(0, 1, 2)
                        A1 = 2*a*r1 - a; C1 = 2*r2
                        X1[j] = a_pos[j] - A1*abs(C1*a_pos[j] - wolves[i, j])
                        r1, r2 = rng.uniform(0, 1, 2)
                        A2 = 2*a*r1 - a; C2 = 2*r2
                        X2[j] = b_pos[j] - A2*abs(C2*b_pos[j] - wolves[i, j])
                        r1, r2 = rng.uniform(0, 1, 2)
                        A3 = 2*a*r1 - a; C3 = 2*r2
                        X3[j] = d_pos[j] - A3*abs(C3*d_pos[j] - wolves[i, j])
                    wolves[i] = (X1 + X2 + X3) / 3.0
                    if bounded: wolves[i] = np.clip(wolves[i], *self._bounds)
                fit = self._eval(wolves)
                idx = np.argsort(fit)
                if fit[idx[0]] < af: af = fit[idx[0]]; a_pos = wolves[idx[0]].copy()
                bf = fit[idx[0]]; b_pos = wolves[idx[1]].copy(); d_pos = wolves[idx[2]].copy()
                self._log(t, bf, a=a)
                self._save(t, a_pos, af)
                if self._done(af): break
        finally:
            self._cleanup()
        return {"x": a_pos, "f": af, "method": "gwo"}


# ═══════════════════════════════════════════════
_REG = {"cma": CMA, "de": DE, "gwo": GWO}


def minimize(fitness: Callable, n_vars: int,
             method: Union[str, type] = "cma",
             bounds: Optional[tuple] = None,
             n_jobs: int = 0, verbose: Union[bool, int] = False,
             seed: int = 42, stop_fitness: Optional[float] = None,
             max_iter: int = 500, pop_size: Optional[int] = None,
             x0: Optional[np.ndarray] = None,
             init: str = "chaos",
             checkpoint: Optional[str] = None,
             resume: Optional[str] = None,
             **kwargs) -> dict:
    """单目标最小化统一入口。

    通用参数:
        fitness     — f(x) → float
        n_vars      — 变量维度
        method      — "cma" | "de" | "gwo" 或 Base 子类
        bounds      — (lb, ub), None=无约束
        n_jobs      — -1=全部CPU, 其他=指定核数, 0=串行
        verbose     — True=自适应, -9=pycma风格, False=静默
        seed        — 随机种子
        stop_fitness— 达到即停
        max_iter    — 最大迭代
        pop_size    — 种群大小, None=自动
        x0          — 初始最优解 (优先于 init)
        init        — "chaos" | "uniform"
        checkpoint  — 定期保存路径
        resume      — 断点续跑路径

    CMA 超参数 (**kwargs):
        sigma=0.3        初始步长

    DE 超参数 (**kwargs):
        CR=1.0           交叉概率
        F_min_early=0.65 前期F下限
        F_max_early=0.90 前期F上限
        F_min_late=0.40  后期F下限
        F_max_late=0.65  后期F上限

    Returns:
        {"x": best_x, "f": best_f, "method": "..."}
    """
    if isinstance(method, str):
        method = _REG[method]
    base_kw = dict(fitness=fitness, n_vars=n_vars, bounds=bounds,
                   n_jobs=n_jobs, verbose=verbose, seed=seed,
                   stop_fitness=stop_fitness, max_iter=max_iter,
                   pop_size=pop_size, x0=x0, init=init,
                   checkpoint=checkpoint, resume=resume)
    # 合并超参数 (sigma, CR 等)
    base_kw.update(kwargs)
    return method(**base_kw).run()
