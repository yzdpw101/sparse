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

import os, json
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import Optional, Callable, Union
import numpy as np


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

class SparseArrayProblem:
    """稀布阵列优化问题。"""

    def __init__(
        self,
        mapper: LMMapper,
        pattern: Pattern,
        mode: int = 100,
        init_positions: Optional[np.ndarray] = None,
        init_phases_deg: Optional[np.ndarray] = None,
        init_amplitudes: Optional[np.ndarray] = None,
        amplitude_bounds: tuple[float, float] = (0.0, 1.0),
        target_hpbw: float = 180.0,
        mainlobe_region: Optional[tuple] = None,
        element_patterns: Optional[list] = None,
        use_pointing_penalty: bool = True,
    ):
        # 解码 mode: 百位=位置, 十位=相位, 个位=幅度
        p_mode = (mode // 100) % 10
        h_mode = (mode // 10) % 10
        a_mode = mode % 10

        if p_mode not in (1, 2):
            raise ValueError(f"mode={mode}: 百位(位置)必须是 1 或 2")
        if h_mode not in (0, 1, 2) or a_mode not in (0, 1, 2):
            raise ValueError(f"mode={mode}: 十位/个位必须是 0/1/2")
        if 1 not in (p_mode, h_mode, a_mode):
            raise ValueError(f"mode={mode}: 至少一位为 1")

        self.mapper = mapper
        self.pattern = pattern
        self.mode = mode
        self._p_mode = p_mode
        self._h_mode = h_mode
        self._a_mode = a_mode
        self.init_positions = init_positions
        self.init_phases_deg = init_phases_deg
        self.init_amplitudes = init_amplitudes
        self.amplitude_lower, self.amplitude_upper = amplitude_bounds
        self.target_hpbw = target_hpbw
        self.mainlobe_region = mainlobe_region
        self.element_patterns = element_patterns
        self.use_pointing_penalty = use_pointing_penalty

        # 变量维度
        self.n_pos = mapper.n_vars if p_mode == 1 else 0
        self.n_phase = mapper.Ne if h_mode == 1 else 0
        self.n_amp = mapper.Ne if a_mode == 1 else 0
        self.n_vars = self.n_pos + self.n_phase + self.n_amp

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -_SIGMOID_CLIP, _SIGMOID_CLIP)
        return 1.0 / (1.0 + np.exp(-x))

    def fitness(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        Ne = self.mapper.Ne

        # 1. 位置
        if self._p_mode == 1:
            pos = self.mapper.synthesize(x[:self.n_pos])
        else:  # _p_mode == 2
            pos = self.init_positions

        # 2. 相位
        if self._h_mode == 1:
            frac = self._sigmoid(x[self.n_pos:self.n_pos + self.n_phase])
            phases = frac * TWO_PI
        elif self._h_mode == 2:
            phases = np.deg2rad(self.init_phases_deg)
        else:
            phases = np.zeros(Ne)

        # 3. 幅度
        if self._a_mode == 1:
            frac = self._sigmoid(x[-self.n_amp:])
            amps = self.amplitude_lower + (self.amplitude_upper - self.amplitude_lower) * frac
        elif self._a_mode == 2:
            amps = self.init_amplitudes
        else:
            amps = np.ones(Ne)

        # 4. 计算 AF
        if self.pattern.is_planar:
            af = self.pattern.planar_af(pos, np.zeros_like(pos), amps, phases)
        elif self.mapper.is_symmetric:
            halfNe = self.mapper._halfNe
            offset = 1 if self.mapper.has_center else 0  # 奇对称跳过中心
            kwargs = dict(has_center=self.mapper.has_center,
                          amplitudes=amps[halfNe + offset:],
                          phases=phases[halfNe + offset:])
            if self.mapper.has_center:
                kwargs["center_amplitude"] = amps[halfNe]
                kwargs["center_phase"] = phases[halfNe]
            af = self.pattern.linear_af_symmetric(pos[halfNe + offset:], **kwargs)
        else:
            af = self.pattern.linear_af(pos, amps, phases)

        # 单元方向图乘积: pattern = Fe * |AF|  (C++ readFeMultiFreqFromCsvs)
        if self.element_patterns is not None:
            af_mag = np.abs(af)
            if af.ndim == 1:  # 单频线阵 (Nθ,)
                af = af_mag * self.element_patterns[0]
            else:             # 多频线阵 (Nf, Nθ)
                for i in range(len(self.element_patterns)):
                    af[i] = af_mag[i] * self.element_patterns[i]
        af_db = Pattern.to_dB(af)

        # 5. PSLL — 线阵多频/多角度时取各子方向图最差 PSLL
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
                # 多频/多角度线阵: 展平遍历
                af_flat = af_db.reshape(-1, af_db.shape[-1])
                psll_val = -np.inf
                for sub in range(af_flat.shape[0]):
                    v, _ = get_psll(af_flat[sub], self.pattern.theta_deg, mainlobe_region=mr_1d)
                    if not np.isinf(v) and v > psll_val:
                        psll_val = v

        # 6. 主瓣指向惩罚 (C++ mainBeamPointPunishment)
        #    多频/多角度时遍历每个子方向图取最大惩罚
        pointing_penalty = 0.0
        if self.use_pointing_penalty and not self.pattern.is_planar:
            from .analysis import _find_peaks  # noqa
            # 将 af_db 展为 (Nsub, Nθ) 以便遍历
            af_db_flat = af_db.reshape(-1, af_db.shape[-1])
            for sub in range(af_db_flat.shape[0]):
                pattern_1d = af_db_flat[sub]
                # 该子方向图对应的指向角索引
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
                pointing_penalty = max(pointing_penalty,
                                       abs(main_idx - target_idx) * 100.0)

        # 7. HPBW 惩罚
        penalty = 0.0
        if self.target_hpbw < 180.0:
            hpbw = self._compute_hpbw(af_db)
            if hpbw > self.target_hpbw:
                penalty = (hpbw - self.target_hpbw) * 100.0

        return float(psll_val) + pointing_penalty + penalty

    def _compute_hpbw(self, af_db: np.ndarray) -> float:
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

    def get_result(self, x_opt: np.ndarray) -> dict:
        x = np.asarray(x_opt, dtype=float)
        r = {}
        if self._p_mode == 1:
            r["positions"] = self.mapper.synthesize(x[:self.n_pos])
        else:  # _p_mode == 2
            r["positions"] = self.init_positions

        if self._h_mode == 1:
            frac = self._sigmoid(x[self.n_pos:self.n_pos + self.n_phase])
            r["phases_deg"] = np.rad2deg(frac * TWO_PI)
        elif self._h_mode == 2:
            r["phases_deg"] = self.init_phases_deg
        else:
            r["phases_deg"] = np.zeros_like(r["positions"])

        if self._a_mode == 1:
            frac = self._sigmoid(x[-self.n_amp:])
            r["amplitudes"] = self.amplitude_lower + (self.amplitude_upper - self.amplitude_lower) * frac
        elif self._a_mode == 2:
            r["amplitudes"] = self.init_amplitudes
        else:
            r["amplitudes"] = np.ones_like(r["positions"])

        return r



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
