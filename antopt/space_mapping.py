"""渐进空间映射 (ASM) — 粗模型(方向图乘积) + 细模型(HFSS) 迭代对齐。

对应 C++ ASM.cpp + Main2_new.cpp 的 PE + Broyden 流程。
不再使用回调框架，直接在 ℝ^D 变量空间操作。
"""

import numpy as np
from .optimizer import SparseArrayProblem
from .pattern import Pattern

TWO_PI = 2 * np.pi


def calc_response_error(Yc_list, Yf_list, ignore_db=-45.0):
    """计算粗/细模型归一化 dB 方向图之间的误差。

    C++ calcError: e = weighted MSE + HPBW mismatch + pointing mismatch.
    简化版: 仅 MSE + HPBW，避免过度耦合导致 Broyden 不稳定。

    Args:
        Yc_list: 粗模型响应列表 [pattern_1d, ...]
        Yf_list: 细模型响应列表 (HFSS CSV dB 值)
        ignore_db: 低于此值的采样点忽略

    Returns:
        float: 总误差
    """
    total = 0.0
    for Yc, Yf in zip(Yc_list, Yf_list):
        # 归一化 dB
        yc_db = 20.0 * np.log10(np.maximum(Yc, 1e-30) / np.max(Yc))
        yf_db = Yf.copy()  # HFSS 导出的 GainTotal 已是线性

        # 细模型: 10*log10 (功率→dB), 归一化
        yf_max = np.max(yf_db)
        if yf_max > 0:
            yf_db = 10.0 * np.log10(np.maximum(yf_db, 1e-30) / yf_max)

        # 仅比较有效区域
        mask = yc_db > ignore_db
        if mask.any():
            diff = yc_db[mask] - yf_db[mask]
            total += np.mean(diff * diff)

        # HPBW 差 (粗)
        peak_idx = int(np.argmax(yc_db))
        hp = -3.0
        left = np.argmin(np.abs(yc_db[:peak_idx] - hp)) if peak_idx > 0 else 0
        right = peak_idx + np.argmin(np.abs(yc_db[peak_idx:] - hp))
        hpbw_c = right - left

        peak_idx_f = int(np.argmax(yf_db))
        left_f = np.argmin(np.abs(yf_db[:peak_idx_f] - hp)) if peak_idx_f > 0 else 0
        right_f = peak_idx_f + np.argmin(np.abs(yf_db[peak_idx_f:] - hp))
        hpbw_f = right_f - left_f

        total += abs(hpbw_c - hpbw_f) * 0.5

    return total


def parameter_extraction(
    Xf, Yf_list, mapper, pattern, fe_patterns,
    pop_size=50, max_iter=200, sigma=0.3, seed=0, verbose=False,
):
    """CMA-ES 找 Xe 使粗模型响应逼近细模型 (对应 C++ PE)。

    Args:
        Xf: 当前细模型变量 (ℝ^D)
        Yf_list: 细模型方向图列表
        mapper: LM 映射器
        pattern: Pattern 实例
        fe_patterns: 单元方向图或 None

    Returns:
        Xe: 参数提取结果 (ℝ^D)
    """
    import cma
    n_vars = mapper.n_vars

    def objective(x):
        pos = mapper.synthesize(x)
        # 计算粗模型方向图
        if pattern.is_planar or not mapper.is_symmetric:
            af = pattern.linear_af(pos)
        else:
            halfNe = mapper._halfNe
            af = pattern.linear_af_symmetric(pos[halfNe:], has_center=mapper.has_center)
        af_abs = np.abs(af)
        if fe_patterns is not None:
            af_abs = af_abs * fe_patterns[0]
        Yc_list = [af_abs]  # 简化: 单频单角度
        return calc_response_error(Yc_list, Yf_list)

    opts = {
        "seed": seed,
        "popsize": pop_size or (4 + int(3 * np.log(n_vars))),
        "maxfevals": int(max_iter * (pop_size or 50)),
        "verbose": 1 if verbose else -9,
        "ftarget": 0.001,
        "tolfun": 1e-8,
        "tolx": 1e-8,
    }
    res = cma.fmin(objective, Xf, sigma, opts)
    return np.array(res[0])


def run_space_mapping(
    Xc_star, mapper, pattern, fe_patterns,
    hfss_func,               # f(Xf) -> list[Yf]
    max_asm_iter=10,
    target_obj=-30.0,
    target_res_norm=1e-4,
    pop_size=50,
    pe_max_iter=200,
    sigma=0.3,
    seed=0,
    verbose=True,
):
    """渐进空间映射主循环 (对应 C++ ASMSolver::run)。

    Args:
        Xc_star: 粗模型最优变量 (ℝ^D)
        mapper: LM 映射器
        pattern: Pattern
        fe_patterns: 单元方向图或 None
        hfss_func: 细模型函数, 返回 [Yf1, Yf2, ...] (线性方向图列表)
        max_asm_iter: 最大 ASM 迭代次数
        target_obj: 目标细模型 PSLL (达到即停止)
        target_res_norm: 残差范数阈值
        pop_size, pe_max_iter, sigma: PE 阶段的 CMA-ES 参数
        seed: 随机种子
        verbose: 打印进度

    Returns:
        dict: {"Xf": 最优细模型变量, "history": [...]}
    """
    n_vars = mapper.n_vars
    theta = pattern.theta_deg

    # 辅助: 计算粗模型响应 (归一化 dB)
    def _yc_db(x):
        pos = mapper.synthesize(x)
        if pattern.is_planar or not mapper.is_symmetric:
            af = pattern.linear_af(pos)
        else:
            hN = mapper._halfNe
            af = pattern.linear_af_symmetric(pos[hN:], has_center=mapper.has_center)
        af_abs = np.abs(af)
        if fe_patterns is not None:
            af_abs = af_abs * fe_patterns[0]
        return 20.0 * np.log10(np.maximum(af_abs, 1e-30) / np.max(af_abs))

    # 细模型 dB
    def _yf_db(yf):
        return 10.0 * np.log10(np.maximum(yf, 1e-30) / np.max(yf))

    # 初始细模型响应
    Yf_list = hfss_func(Xc_star)
    fine_psll = _compute_fine_psll(Yf_list)
    if verbose:
        print(f"ASM Iter 0: PSLL={fine_psll:.2f} dB (coarse optimum)")

    # PE: 对齐初始响应
    if verbose:
        print("  PE (initial):", end="", flush=True)
    t0 = __import__('time').perf_counter()
    Xe = parameter_extraction(Xc_star, Yf_list, mapper, pattern, fe_patterns,
                               pop_size, pe_max_iter, sigma, seed, verbose=verbose)
    if verbose:
        print(f"  PE done ({__import__('time').perf_counter()-t0:.1f}s), |f|={np.linalg.norm(Xe-Xc_star):.4f}")
    f = Xe - Xc_star  # 残差
    B = np.eye(n_vars)  # Broyden Jacobian
    Xf = Xc_star.copy()
    history = [{"iter": 0, "Xf": Xf.copy(), "PSLL": fine_psll,
                "res_norm": float(np.linalg.norm(f)),
                "yc_db": _yc_db(Xe).tolist(), "yf_db": _yf_db(Yf_list[0]).tolist(),
                "xe": Xe.tolist()}]

    if np.linalg.norm(f) < target_res_norm or fine_psll <= target_obj:
        if verbose:
            print("  Initial design meets criteria. Stopping.")
        return {"Xf": Xf, "history": history}

    for k in range(1, max_asm_iter + 1):
        # 1) Broyden 步: B * h = -f
        try:
            h = np.linalg.solve(B, -f)
        except np.linalg.LinAlgError:
            B_reg = B + np.eye(n_vars) * 1e-8
            h = np.linalg.solve(B_reg, -f)
        if verbose:
            print(f"  Broyden step norm: {np.linalg.norm(h):.4f}")

        # 2) 更新细模型变量 (ℝ^D 无约束) → HFSS
        Xf_new = Xf + h
        Yf_list = hfss_func(Xf_new)
        fine_psll = _compute_fine_psll(Yf_list)

        # 3) 参数提取
        if verbose:
            print(f"  PE (iter {k}):", end="", flush=True)
        t_pe = __import__('time').perf_counter()
        Xe_new = parameter_extraction(Xf_new, Yf_list, mapper, pattern, fe_patterns,
                                       pop_size, pe_max_iter, sigma, seed, verbose=verbose)
        if verbose:
            print(f"  PE done ({__import__('time').perf_counter()-t_pe:.1f}s), |f|={np.linalg.norm(Xe_new-Xc_star):.4f}")

        f_new = Xe_new - Xc_star

        # 4) Broyden 更新
        s = h
        y = f_new - f
        denom = np.dot(s, s)
        if denom > 1e-16:
            B += np.outer(y - B @ s, s) / denom

        res_norm = float(np.linalg.norm(f_new))
        history.append({"iter": k, "Xf": Xf_new.copy(), "PSLL": fine_psll,
                         "res_norm": res_norm,
                         "yc_db": _yc_db(Xe_new).tolist(),
                         "yf_db": _yf_db(Yf_list[0]).tolist(),
                         "xe": Xe_new.tolist()})

        if verbose:
            print(f"  ASM Iter {k}: PSLL={fine_psll:.2f} dB, res_norm={res_norm:.4f}")

        Xf = Xf_new
        f = f_new

        if fine_psll <= target_obj or res_norm < target_res_norm:
            if verbose:
                print("  Converged!")
            break

    yc_star_db = _yc_db(Xc_star).tolist()
    return {"Xf": Xf, "history": history, "yc_star_db": yc_star_db}


def _compute_fine_psll(Yf_list):
    """计算细模型 (HFSS) PSLL: 10*log10 归一化后取副瓣峰值。"""
    from .analysis import _find_peaks
    worst = -np.inf
    for yf in Yf_list:
        yf_max = np.max(yf)
        if yf_max <= 0:
            continue
        db = 10.0 * np.log10(np.maximum(yf, 1e-30) / yf_max)
        _, extrema = _find_peaks(db)
        if len(extrema) >= 2:
            worst = max(worst, extrema[1])
    return float(worst)
