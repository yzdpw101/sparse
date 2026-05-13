"""Kriging 代理模型测试。"""

import numpy as np
import pytest

try:
    from antopt.surrogate import KrigingSurrogate, surrogate_optimize, lhs_sample
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

pytestmark = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn 未安装")


def test_lhs_sample():
    """LHS 采样数量和边界正确。"""
    n_vars, n_samples = 4, 50
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    X = lhs_sample(n_vars, n_samples, (lb, ub), seed=0)

    assert X.shape == (n_samples, n_vars)
    assert np.all(X >= lb - 1e-10)
    assert np.all(X <= ub + 1e-10)
    # 无完全重复行
    assert len(np.unique(X, axis=0)) == n_samples


def test_lhs_sample_scaled():
    """LHS 采样缩放到非 [0,1] 边界。"""
    lb = np.array([1.0, -2.0])
    ub = np.array([5.0, 3.0])
    X = lhs_sample(2, 30, (lb, ub), seed=42)

    assert X.shape == (30, 2)
    assert np.all(X[:, 0] >= 1.0 - 1e-10)
    assert np.all(X[:, 0] <= 5.0 + 1e-10)
    assert np.all(X[:, 1] >= -2.0 - 1e-10)
    assert np.all(X[:, 1] <= 3.0 + 1e-10)


def test_kriging_fit_predict():
    """GPR 拟合后预测合理。"""
    rng = np.random.RandomState(0)
    X_train = rng.uniform(0, 1, (20, 3))
    y_train = np.sum(X_train ** 2, axis=1)  # f(x) = sum(x^2)

    surr = KrigingSurrogate(n_vars=3)
    surr.fit(X_train, y_train)

    X_test = rng.uniform(0, 1, (10, 3))
    mu, sigma = surr.predict(X_test, return_std=True)

    assert mu.shape == (10,)
    assert sigma.shape == (10,)
    assert np.all(sigma >= 0)  # 标准差非负


def test_ei_positive_at_unexplored():
    """未探索区域 EI > 0。"""
    rng = np.random.RandomState(1)
    X_train = rng.uniform(0, 1, (15, 2))
    y_train = np.sum(X_train ** 2, axis=1)

    surr = KrigingSurrogate(n_vars=2)
    surr.fit(X_train, y_train)

    f_best = np.min(y_train)
    # 远离训练点的区域
    X_new = np.array([[0.5, 0.5], [0.1, 0.9]])
    ei_vals = surr.ei(X_new, f_best)

    assert np.all(ei_vals >= 0)


def test_ei_zero_at_known_point():
    """已知最优点处 EI ≈ 0。"""
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    y_train = np.array([0.0, 2.0, 0.5])  # 最优在 (0,0)

    surr = KrigingSurrogate(n_vars=2)
    surr.fit(X_train, y_train)

    ei_at_best = surr.ei(np.array([[0.0, 0.0]]), f_best=0.0)
    assert ei_at_best[0] < 0.1  # 几乎为零


def test_surrogate_optimize_toy():
    """代理优化求解 2D sum(x^2)，最优解在原点。"""
    def toy(x):
        return np.sum(x ** 2)

    n_vars = 2
    lb = np.full(n_vars, -1.0)
    ub = np.full(n_vars, 1.0)

    result = surrogate_optimize(
        toy, n_vars, bounds=(lb, ub),
        n_initial=20, n_infill=3, max_iter=5,
        patience=3, seed=42, verbose=False
    )

    assert result["f"] < 0.5  # 应接近 0
    assert result["n_evals"] >= 20
    assert len(result["history"]) >= 2
