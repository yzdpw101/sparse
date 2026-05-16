"""多模态测试函数 — 用于评估优化器性能。"""

import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin 函数 — 大量局部极小, 全局最小在原点 (f=0)。"""
    A = 10.0
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray) -> float:
    """Ackley 函数 — 中心有深坑, 周围大量局部极小。"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock 函数 — 狭长弯曲山谷, 全局最小在 (1,1,...,1)。"""
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
               for i in range(len(x) - 1))


def griewank(x: np.ndarray) -> float:
    """Griewank 函数 — 大量规则分布的局部极小。"""
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1


def schwefel(x: np.ndarray) -> float:
    """Schwefel 函数 — 全局极小远离原点, 难以搜索。"""
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


# 所有测试函数: (函数, 搜索范围, 全局最优位置, 全局最优值)
BENCHMARKS = {
    "rastrigin":  (rastrigin,  [-5.12, 5.12],   np.zeros,   0.0),
    "ackley":     (ackley,     [-32.768, 32.768], np.zeros,  0.0),
    "rosenbrock": (rosenbrock, [-5.0, 10.0],     lambda n: np.ones(n), 0.0),
    "griewank":   (griewank,   [-600, 600],      np.zeros,   0.0),
    "schwefel":   (schwefel,   [-500, 500],      lambda n: np.full(n, 420.9687), 0.0),
}
