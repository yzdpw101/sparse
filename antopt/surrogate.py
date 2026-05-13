"""Kriging 代理模型 — 用高斯过程回归替代昂贵的 AF 计算。

流程:
  1. LHS 采样 → 真实 fitness 计算 → 训练 Kriging
  2. CMA-ES 在 Kriging 上搜索 → 候选解
  3. EI/UCB 选择填充点 → 真实 fitness 验证 → 加入训练集
  4. 重训练 Kriging, 重复直到收敛

需要安装: pip install scikit-learn>=1.3
"""

import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class KrigingSurrogate:
    """Kriging 代理模型 (sklearn GPR 封装)。"""

    def __init__(self, n_vars, alpha=1e-6):
        """
        参数:
            n_vars — 变量维度
            alpha  — GPR 噪声正则化
        """
        self.n_vars = n_vars
        kernel = Matern(nu=2.5, length_scale=np.ones(n_vars),
                        length_scale_bounds=(1e-5, 1e5))
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True,
            n_restarts_optimizer=5, random_state=None
        )

    def fit(self, X, y):
        """训练 GPR。

        参数:
            X — (n_samples, n_vars)
            y — (n_samples,)
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        self.gpr.fit(X, y)

    def predict(self, X, return_std=True):
        """预测均值和标准差。

        返回: (mu, sigma) 或仅 mu
        """
        X = np.atleast_2d(X)
        return self.gpr.predict(X, return_std=return_std)

    def ei(self, X, f_best):
        """Expected Improvement (最小化场景, 越小越好)。

        EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
        Z  = (f_best - mu) / sigma
        """
        mu, sigma = self.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-30)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)

    def lcb(self, X, kappa=2.0):
        """Lower Confidence Bound (最小化场景, 越小越好)。

        LCB = mu - kappa * sigma
        """
        mu, sigma = self.predict(X, return_std=True)
        return mu - kappa * sigma


def lhs_sample(n_vars, n_samples, bounds, seed=42):
    """拉丁超立方采样 (LHS)。

    参数:
        n_vars    — 变量维度
        n_samples — 采样数量
        bounds    — (lb, ub), 各为 (n_vars,) 数组
        seed      — 随机种子

    返回: (n_samples, n_vars) 采样点
    """
    sampler = qmc.LatinHypercube(d=n_vars, seed=seed)
    unit = sampler.random(n=n_samples)  # [0,1]^d
    lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])
    return lb + (ub - lb) * unit


def surrogate_optimize(fitness, n_vars, bounds,
                       n_initial=None, n_infill=5, max_iter=20,
                       patience=3, acquisition="ei", kappa=2.0,
                       method="cma", sigma=0.3, seed=42,
                       verbose=True):
    """Kriging 代理模型优化主入口。

    参数:
        fitness     — 目标函数 f(x) → float
        n_vars      — 变量维度
        bounds      — (lb, ub)
        n_initial   — 初始 LHS 采样数 (默认 10*n_vars)
        n_infill    — 每轮填充采样数
        max_iter    — 最大填充迭代轮数
        patience    — 连续无改善轮数后停止
        acquisition — "ei" 或 "lcb"
        kappa       — LCB 探索权重
        method      — 代理搜索优化器 ("cma"/"de"/"gwo")
        sigma       — CMA-ES 初始步长
        seed        — 随机种子
        verbose     — 打印进度

    返回: {"x", "f", "n_evals", "history"}
    """
    from .optimizer import minimize as _minimize

    if n_initial is None:
        n_initial = max(10 * n_vars, 20)

    lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])

    # ── 1. 初始设计 (DoE) ──
    X_all = lhs_sample(n_vars, n_initial, bounds, seed=seed)
    Y_all = np.array([fitness(x) for x in X_all])

    best_idx = np.argmin(Y_all)
    best_x = X_all[best_idx].copy()
    best_f = Y_all[best_idx]

    history = [{"iter": 0, "best_f": float(best_f), "n_evals": len(Y_all)}]
    if verbose:
        print(f"[代理] 初始采样 {n_initial} 个, 最优 PSLL = {best_f:.4f}")

    no_improve = 0

    # ── 2. 填充循环 ──
    for k in range(1, max_iter + 1):
        # 2a. 训练 Kriging
        surr = KrigingSurrogate(n_vars)
        surr.fit(X_all, Y_all)

        # 2b. CMA-ES 在代理上搜索 (exploitation)
        def _surrogate_pred(x):
            mu, _ = surr.predict(x.reshape(1, -1), return_std=True)
            return mu[0]

        res = _minimize(_surrogate_pred, n_vars, method=method,
                        bounds=bounds, sigma=sigma, seed=seed + k,
                        max_iter=200, verbose=False)
        candidates = [res["x"]]

        # 2c. 多次随机启动最大化 EI (exploration)
        if acquisition == "ei":
            for r in range(max(n_infill - 1, 0)):
                def _neg_ei(x, _fb=best_f):
                    return -surr.ei(x.reshape(1, -1), _fb)[0]

                ei_res = _minimize(_neg_ei, n_vars, method="cma",
                                   bounds=bounds, sigma=sigma,
                                   seed=seed + k * 100 + r,
                                   max_iter=200, verbose=False)
                candidates.append(ei_res["x"])

        # 2d. 去重 + 真实 fitness 评估
        X_cand = np.array(candidates)
        new_fvals = np.array([fitness(x) for x in X_cand])

        improved = False
        for i in range(len(X_cand)):
            if new_fvals[i] < best_f:
                best_f = new_fvals[i]
                best_x = X_cand[i].copy()
                improved = True

        X_all = np.vstack([X_all, X_cand])
        Y_all = np.concatenate([Y_all, new_fvals])

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        history.append({"iter": k, "best_f": float(best_f),
                        "n_evals": len(Y_all)})

        if verbose:
            print(f"[代理] 轮 {k}/{max_iter}: "
                  f"最优 PSLL = {best_f:.4f}, "
                  f"总评估 = {len(Y_all)}")

        if no_improve >= patience:
            if verbose:
                print(f"[代理] 连续 {patience} 轮无改善, 提前停止")
            break

    return {"x": best_x, "f": float(best_f),
            "n_evals": len(Y_all), "history": history}
