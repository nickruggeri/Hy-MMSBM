from itertools import product

import numpy as np

from src.model._linear_ops import bf, bf_and_sum, qf, qf_and_sum


def make_symmetric(w):
    return (w + w.T) / 2


def test_bf_one_dim(u, v, w):
    def correct_bf(u, v, w):
        K = w.shape[0]
        return sum(u[i] * w[i, j] * v[j] for i in range(K) for j in range(K))

    efficient = bf(u, v, w)
    correct = correct_bf(u, v, w)
    assert efficient.shape == correct.shape, (efficient.shape, correct.shape)
    assert np.allclose(efficient, correct)


def test_bf(u, v, w):
    def correct_bf(u, v, w):
        K = w.shape[0]
        res = np.zeros((u.shape[0], v.shape[0]))
        for i in range(K):
            for j in range(K):
                for n1 in range(u.shape[0]):
                    for n2 in range(v.shape[0]):
                        res[n1, n2] += u[n1, i] * w[i, j] * v[n2, j]
        return res

    efficient = bf(u, v, w)
    correct = correct_bf(u, v, w)
    assert efficient.shape == correct.shape, (efficient.shape, correct.shape)
    assert np.allclose(efficient, correct)


def test_qf(u, w):
    def correct_qf(u, w):
        K = w.shape[0]
        return sum(u[..., i] * w[i, j] * u[..., j] for i in range(K) for j in range(K))

    efficient = qf(u, w)
    correct = correct_qf(u, w)
    assert efficient.shape == correct.shape, (efficient.shape, correct.shape)
    assert np.allclose(efficient, correct)


def test_qf_and_sum(u, w):
    def correct_qf_and_sum(u, w):
        N = u.shape[0]
        res = np.zeros(1)
        for i in range(N):
            res += qf(u[i], w)
        return res

    assert np.allclose(correct_qf_and_sum(u, w), qf_and_sum(u, w))


def test_bf_and_sum(u, w):
    def correct_bf_and_sum(u, w):
        N = u.shape[0]
        res = np.zeros(1)
        for i in range(N):
            for j in range(i + 1, N):
                res += bf(u[i], u[j], w)
        return res

    def correct_bf_and_sum_2(u, w):
        N = u.shape[0]
        res = np.zeros(1)
        for i in range(N):
            for j in range(N):
                if i != j:
                    res += bf(u[i], u[j], w)
        return 0.5 * res

    def correct_bf_and_sum_3(u, w):
        N = u.shape[0]
        res = np.zeros(1)
        for i in range(N):
            for j in range(N):
                res += bf(u[i], u[j], w)
        for i in range(N):
            res -= qf(u[i], w)
        return 0.5 * res

    assert np.allclose(correct_bf_and_sum(u, w), bf_and_sum(u, w))
    assert np.allclose(correct_bf_and_sum_2(u, w), bf_and_sum(u, w))
    assert np.allclose(correct_bf_and_sum_3(u, w), bf_and_sum(u, w))


if __name__ == "__main__":
    # Test bf.
    N_vals = [2, 5, 7, 8, 1]
    k_vals = [3, 5, 6, 13, 1]

    for K in k_vals:
        u = np.random.rand(K)
        v = np.random.rand(K)
        w = np.random.rand(K, K)
        test_bf_one_dim(u, v, w)

    for u_dims, K, v_dims in product(N_vals, k_vals, N_vals):
        for _ in range(10):
            u = np.random.rand(u_dims, K)
            v = np.random.rand(v_dims, K)
            w = np.random.rand(K, K)
            test_bf(u, v, w)

    # Test qf.
    N_vals = [2, 5, 7, 8, 1]
    k_vals = [3, 5, 6, 13, 1]
    for u_dims, K in product(N_vals, k_vals):
        for _ in range(10):
            u = np.random.rand(u_dims, K)
            w = np.random.rand(K, K)
            test_qf(u, w)

    # Test sum operations.
    N_vals = [2, 5, 7, 8, 1]
    k_vals = [3, 5, 6, 13, 1]
    for u_dims, K in product(N_vals, k_vals):
        for _ in range(10):
            u = np.random.rand(u_dims, K) * 10
            w = make_symmetric(np.random.rand(K, K))
            test_qf_and_sum(u, w)
            test_bf_and_sum(u, w)
