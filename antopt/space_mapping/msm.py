"""流形空间映射 (MSM/OMM) — 替代模型 Rs(x) = Yf + S·(Yc - Yc_ref)。"""

import numpy as np
from .base import compute_fine_psll, TWO_PI


def run_manifold_mapping(
    Xc_star, coarse_model_func, n_vars,
    hfss_func,
    max_iter=10,
    target_fine_psll=-30.0,
    cma_iter=300,
    sigma=0.3,
    seed=0,
    verbose=True,
    on_iter_end=None,
):
    """流形空间映射 (MSM/OMM) 主循环。

    核心思想: 构造替代模型 Rs(x) = Yf(x^(i)) + S·(Yc(x) - Yc(x^(i))),
    用 CMA-ES 直接优化 Rs 得到下一次细模型参数, 再通过 Broyden rank-1 更新 S 矩阵。
    与 ASM 的区别: 无 PE 过程, S 在响应空间 (Nθ×Nθ), 不锚定粗模型最优。

    Args:
        Xc_star: 粗模型最优变量 (ℝ^D)
        coarse_model_func: 粗模型函数, f(x) -> 1d dB 方向图 (归一化, clip)
        n_vars: 优化变量维度
        hfss_func: 细模型函数, 返回 [Yf1, Yf2, ...] (线性方向图列表)
        max_iter: 最大 MSM 迭代次数
        target_fine_psll: 目标细模型 PSLL (达到即停止)
        cma_iter: 替代模型优化的 CMA-ES 最大迭代
        sigma: CMA-ES 初始步长
        seed: 随机种子
        verbose: 打印进度
        on_iter_end: 每轮结束回调, f(iter_k, Xf, yc_db, yf_db, psll_fine)

    Returns:
        dict: {"Xf": 最优细模型变量, "history": [...]}
    """
    import cma as _cma
    from ..analysis import _find_peaks

    IGNORE_DB = -45.0

    def _yc_db(x):
        return coarse_model_func(x)

    n_theta = len(_yc_db(Xc_star))

    def _yf_db(yf_linear):
        yf = np.sqrt(np.maximum(yf_linear, 0))
        yf_norm = yf / np.max(yf) if np.max(yf) > 0 else yf
        yf_db = 20.0 * np.log10(np.maximum(yf_norm, 1e-30))
        return np.maximum(yf_db, IGNORE_DB)

    def _psll_from_db(db_arr):
        _, extrema = _find_peaks(db_arr)
        return float(extrema[1]) if len(extrema) >= 2 else -np.inf

    # ========== Step 0: 初始细模型仿真 ==========
    Yf_init_list = hfss_func(Xc_star)
    Yf_init = Yf_init_list[0]
    fine_psll_init = compute_fine_psll(Yf_init_list)
    Yf_k = _yf_db(Yf_init)
    Xf_k = Xc_star.copy()
    Yc_k = _yc_db(Xf_k)

    if verbose:
        print(f"MSM Iter 0: Fine PSLL={fine_psll_init:.2f} dB")

    S = np.eye(n_theta)

    history = [{"iter": 0, "PSLL": fine_psll_init, "deltaX_norm": float("nan"),
                "deltaC_norm": float("nan")}]

    if fine_psll_init <= target_fine_psll:
        if verbose:
            print("  Initial design meets criteria. Stopping.")
        return {"Xf": Xf_k, "history": history}

    # ========== MSM 迭代 ==========
    for k in range(1, max_iter + 1):
        if verbose:
            print(f"\n  --- MSM iter {k}/{max_iter} ---")

        Yf_ref = Yf_k.copy()
        Yc_ref = Yc_k.copy()
        Xf_ref = Xf_k.copy()

        def surrogate_fitness(x, _Yf_ref=Yf_ref, _Yc_ref=Yc_ref, _S=S):
            yc_x = _yc_db(x)
            delta_c = yc_x - _Yc_ref
            rs_db = _Yf_ref + _S @ delta_c
            return _psll_from_db(rs_db)

        lb = np.zeros(n_vars)
        ub = np.full(n_vars, TWO_PI)

        opts = _cma.CMAOptions()
        opts["verbose"] = -9
        opts["popsize"] = 4 + int(3 * np.log(n_vars))
        opts["maxfevals"] = int(cma_iter * opts["popsize"])
        opts["bounds"] = [lb.tolist(), ub.tolist()]
        opts["seed"] = seed + k
        opts["ftarget"] = -100.0

        res = _cma.fmin(surrogate_fitness, Xf_ref, sigma, opts)
        Xf_new = np.array(res[0])

        deltaX_norm = float(np.linalg.norm(Xf_new - Xf_ref))
        if verbose:
            print(f"  surrogate optimized, deltaX_norm={deltaX_norm:.6f}")

        if deltaX_norm < 1e-4:
            if verbose:
                print("  Converged (deltaX < 1e-4). Stopping.")
            break

        Yf_new_list = hfss_func(Xf_new)
        fine_psll_new = compute_fine_psll(Yf_new_list)
        Yf_new = _yf_db(Yf_new_list[0])

        deltaF = Yf_new - Yf_k
        Yc_new = _yc_db(Xf_new)
        deltaC = Yc_new - Yc_k

        deltaC_norm = float(np.linalg.norm(deltaC))
        if deltaC_norm > 1e-12:
            diff = deltaF - S @ deltaC
            S += np.outer(diff, deltaC) / (deltaC_norm ** 2)
        elif verbose:
            print("  [Warning] deltaC too small, skipping S update.")

        Xf_k = Xf_new
        Yf_k = Yf_new
        Yc_k = Yc_new

        history.append({"iter": k, "PSLL": fine_psll_new,
                        "deltaX_norm": deltaX_norm,
                        "deltaC_norm": deltaC_norm})

        if verbose:
            print(f"  Fine PSLL={fine_psll_new:.2f} dB, deltaC_norm={deltaC_norm:.6f}")

        if on_iter_end is not None:
            on_iter_end(k, Xf_new, Yc_new, Yf_new, fine_psll_new)

        if fine_psll_new <= target_fine_psll:
            if verbose:
                print("  Target PSLL reached!")
            break

    return {"Xf": Xf_k, "history": history}
