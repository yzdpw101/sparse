"""渐进空间映射 (ASM) — 粗模型(方向图乘积) + 细模型(HFSS) 迭代对齐。

对应 C++ ASM.cpp + Main2_new.cpp 的 PE + Broyden 流程。
"""

import numpy as np
from .base import calc_response_error, compute_fine_psll


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
        if pattern.is_planar or not mapper.is_symmetric:
            af = pattern.linear_af(pos)
        else:
            halfNe = mapper._halfNe
            offset = 1 if mapper.has_center else 0
            af = pattern.linear_af_symmetric(pos[halfNe + offset:], has_center=mapper.has_center)
        af_abs = np.abs(af)
        if fe_patterns is not None:
            af_abs = af_abs * fe_patterns[0]
        Yc_list = [af_abs]
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
    hfss_func,
    max_asm_iter=10,
    target_obj=-30.0,
    target_res_norm=1e-4,
    pop_size=50,
    pe_max_iter=200,
    sigma=0.3,
    seed=0,
    verbose=True,
    step_norm_max=None,
    pe_config=None,
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
        pop_size, pe_max_iter, sigma: PE 阶段的 CMA-ES 参数 (pe_config 优先)
        seed: 随机种子
        verbose: 打印进度
        step_norm_max: Broyden 步长上限 (None=不限制)
        pe_config: PE 参数字典, 覆盖 pop_size/pe_max_iter/sigma

    Returns:
        dict: {"Xf": 最优细模型变量, "history": [...]}
    """
    import time as _time

    n_vars = mapper.n_vars
    theta = pattern.theta_deg

    # pe_config 覆盖单独参数
    if pe_config is not None:
        pop_size = pe_config.get("pop_size", pop_size)
        pe_max_iter = pe_config.get("max_iter", pe_max_iter)
        sigma = pe_config.get("sigma", sigma)
        seed_pe = pe_config.get("seed", seed)
        verbose_pe = pe_config.get("verbose", verbose)
    else:
        seed_pe = seed
        verbose_pe = verbose

    # 辅助: 计算粗模型响应 (归一化 dB)
    def _yc_db(x):
        pos = mapper.synthesize(x)
        if pattern.is_planar or not mapper.is_symmetric:
            af = pattern.linear_af(pos)
        else:
            hN = mapper._halfNe
            offset = 1 if mapper.has_center else 0
            af = pattern.linear_af_symmetric(pos[hN + offset:], has_center=mapper.has_center)
        af_abs = np.abs(af)
        if fe_patterns is not None:
            af_abs = af_abs * fe_patterns[0]
        return 20.0 * np.log10(np.maximum(af_abs, 1e-30) / np.max(af_abs))

    # 细模型 dB
    def _yf_db(yf):
        return 10.0 * np.log10(np.maximum(yf, 1e-30) / np.max(yf))

    # 初始细模型响应
    Yf_list = hfss_func(Xc_star)
    fine_psll = compute_fine_psll(Yf_list)
    if verbose:
        print(f"ASM Iter 0: PSLL={fine_psll:.2f} dB (coarse optimum)")

    # PE: 对齐初始响应
    if verbose:
        print("  PE (initial):", end="", flush=True)
    t0 = _time.perf_counter()
    Xe = parameter_extraction(Xc_star, Yf_list, mapper, pattern, fe_patterns,
                              pop_size, pe_max_iter, sigma, seed_pe, verbose=verbose_pe)
    if verbose:
        print(f"  PE done ({_time.perf_counter()-t0:.1f}s), |f|={np.linalg.norm(Xe-Xc_star):.4f}")
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

        # 步长截断
        if step_norm_max is not None:
            h_norm = np.linalg.norm(h)
            if h_norm > step_norm_max:
                h = h * step_norm_max / h_norm
                if verbose:
                    print(f"  Broyden step truncated: {h_norm:.4f} -> {step_norm_max}")

        if verbose:
            print(f"  Broyden step norm: {np.linalg.norm(h):.4f}")

        # 2) 更新细模型变量 → HFSS
        Xf_new = Xf + h
        Yf_list = hfss_func(Xf_new)
        fine_psll = compute_fine_psll(Yf_list)

        # 3) 参数提取
        if verbose:
            print(f"  PE (iter {k}):", end="", flush=True)
        t_pe = _time.perf_counter()
        Xe_new = parameter_extraction(Xf_new, Yf_list, mapper, pattern, fe_patterns,
                                      pop_size, pe_max_iter, sigma, seed_pe, verbose=verbose_pe)
        if verbose:
            print(f"  PE done ({_time.perf_counter()-t_pe:.1f}s), |f|={np.linalg.norm(Xe_new-Xc_star):.4f}")

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
