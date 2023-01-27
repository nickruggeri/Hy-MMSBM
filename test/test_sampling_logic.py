"""Test the logic of the sampling functions of HyMMSBM and HyMMSBMSampler."""

import logging
from collections import Counter

import numpy as np

from src.data.data_io import load_real_hypergraph
from src.model.model import HyMMSBM


def test1(K, u, w, default_deg_seq, default_dim_seq, verbose=False):
    """TEST 1: everything runs without errors"""
    if verbose:
        print("TEST 1")
    model = HyMMSBM(K, u, w, assortative, max_hye_size=max_hye_size)
    for exact_binary_sampling in [True, False]:
        for dim_seq in [None, default_dim_seq]:
            for deg_seq in [None, default_deg_seq]:
                samples = model.sample(
                    deg_seq=deg_seq,
                    dim_seq=dim_seq,
                    exact_binary_sampling=exact_binary_sampling,
                )
                if verbose:
                    print(
                        "\tEXACT BINARY",
                        exact_binary_sampling,
                        "DIM SEQ is None",
                        dim_seq is None,
                        "DEG SEQ is None",
                        deg_seq is None,
                    )
                for i in range(10):
                    if verbose:
                        print("\t\t", i, end="")
                    next(samples)
                print("")


def test2(K, u, w, default_deg_seq, default_dim_seq, verbose=False):
    """TEST 2: if we provide a dimension sequence, that is preserved during sampling
    regardless of the degree sequence.
    """
    if verbose:
        print("TEST 2")
    model = HyMMSBM(K, u, w, assortative, max_hye_size=max_hye_size)
    for exact_binary_sampling in [True, False]:
        for deg_seq in [None, default_deg_seq]:
            samples = model.sample(
                deg_seq=deg_seq,
                dim_seq=default_dim_seq,
                exact_binary_sampling=exact_binary_sampling,
            )
            if verbose:
                print(
                    "\tEXACT BINARY",
                    exact_binary_sampling,
                    "DEG SEQ is None",
                    deg_seq is None,
                )
            for i in range(10):
                if verbose:
                    print("\t\t", i, end="")
                hyg, sampler = next(samples)
                binary_incidence = hyg.get_binary_incidence_matrix()
                sampled_dim_seq = dict(Counter(binary_incidence.sum(axis=0)))

                assert sampled_dim_seq == default_dim_seq
            print("")


def test3(K, u, w, default_deg_seq, verbose=False):
    """TEST 3: if we provide a degree sequence, that is preserved. At most one node can
    be different from the specified degree sequence.
    """
    if verbose:
        print("TEST 3")
    model = HyMMSBM(K, u, w, assortative, max_hye_size=max_hye_size)
    for exact_binary_sampling in [True, False]:
        samples = model.sample(
            deg_seq=default_deg_seq,
            dim_seq=None,
            exact_binary_sampling=exact_binary_sampling,
        )
        if verbose:
            print("\tEXACT BINARY", exact_binary_sampling)
        for i in range(10):
            if verbose:
                print("\t\t", i, end="")
            hyg, sampler = next(samples)
            binary_incidence = hyg.get_binary_incidence_matrix()
            sampled_deg_seq = binary_incidence.sum(axis=1)

            assert np.sum(sampled_deg_seq != default_deg_seq) <= 1
        print("")


def test4(dataset, K, verbose=False):
    """TEST 4: if we provide degree and dimension sequences that are feasible
    (extracting it from a real hypergraph), then these are preserved during sampling.
    """
    if verbose:
        print("TEST 4")

    real_hyg = load_real_hypergraph(dataset)
    binary_adj = real_hyg.get_binary_incidence_matrix()

    real_deg_seq = binary_adj.sum(axis=1)
    real_dim_seq = dict(Counter(binary_adj.sum(axis=0)))

    N = real_hyg.N
    u = np.random.rand(N, K)
    w = np.random.rand(K, K)
    w = w + w.T
    model = HyMMSBM(K, u, w, assortative, max_hye_size=real_hyg.max_hye_size)

    for exact_binary_sampling in [True, False]:
        samples = model.sample(
            deg_seq=real_deg_seq,
            dim_seq=real_dim_seq,
            exact_binary_sampling=exact_binary_sampling,
            burn_in_steps=10,
            intermediate_steps=10,
        )
        if verbose:
            print("\tEXACT BINARY", exact_binary_sampling)
        for i in range(10):
            if verbose:
                print("\t\t", i, end="")
            hyg, sampler = next(samples)
            binary_incidence = hyg.get_binary_incidence_matrix()
            sampled_deg_seq = binary_incidence.sum(axis=1)
            sampled_dim_seq = dict(Counter(binary_incidence.sum(axis=0)))

            assert np.all(sampled_deg_seq == real_deg_seq)
            assert sampled_dim_seq == real_dim_seq
        print("")


def test5(K, u, w, default_deg_seq, verbose=False):
    """TEST 5: check that everything runs smoothly also when the expected degree is
    provided."""
    if verbose:
        print("TEST 5")
    model = HyMMSBM(K, u, w, assortative, max_hye_size=max_hye_size)
    for exact_binary_sampling in [True, False]:
        for deg_seq in [None, default_deg_seq]:
            samples = model.sample(
                deg_seq=deg_seq,
                dim_seq=None,
                avg_deg=3.0,
                exact_binary_sampling=exact_binary_sampling,
            )
            if verbose:
                print(
                    "\tEXACT BINARY",
                    exact_binary_sampling,
                    "DEG SEQ is None",
                    deg_seq is None,
                )
            for i in range(10):
                if verbose:
                    print("\t\t", i, end="")
                next(samples)
            print("")


if __name__ == "__main__":
    configs = [
        {
            "N": 100,
            "K": 3,
            "assortative": False,
            "max_hye_size": 10,
            "default_dim_seq": {2: 5, 3: 2, 5: 2, 10: 2},
            "default_deg_seq": np.random.randint(0, 10, size=100, dtype=int),
        },
        {
            "N": 100,
            "K": 3,
            "assortative": False,
            "max_hye_size": 10,
            "default_dim_seq": {2: 5, 3: 2, 5: 2, 10: 2},
            "default_deg_seq": np.random.randint(0, 100, size=100, dtype=int),
        },
        {
            "N": 100,
            "K": 3,
            "assortative": False,
            "max_hye_size": 10,
            "default_dim_seq": {2: 10, 3: 4, 5: 10, 10: 6},
            "default_deg_seq": np.random.randint(0, 10, size=100, dtype=int),
        },
        {
            "N": 100,
            "K": 5,
            "assortative": False,
            "max_hye_size": 10,
            "default_dim_seq": {2: 10, 3: 4, 5: 10, 10: 6},
            "default_deg_seq": np.random.randint(0, 10, size=100, dtype=int),
        },
    ]

    verbose = True
    logger = logging.getLogger()
    logger.disabled = True
    for i, config in enumerate(configs):
        print(
            "\n\n",
            "#" * 30,
            "\n",
            "#" * 10,
            f" CONFIG {i} ",
            "#" * 10,
            "\n",
            "#" * 30,
            sep="",
        )
        N = config["N"]
        K = config["K"]
        assortative = config["assortative"]
        max_hye_size = config["max_hye_size"]
        default_dim_seq = config["default_dim_seq"]
        default_deg_seq = config["default_deg_seq"]

        u = np.random.rand(N, K)

        w = np.random.rand(K, K)
        w = w + w.T

        test1(K, u, w, default_deg_seq, default_dim_seq, verbose)
        test2(K, u, w, default_deg_seq, default_dim_seq, verbose)
        test3(K, u, w, default_deg_seq, verbose)
        test5(K, u, w, default_deg_seq, verbose)
    test4(dataset="high_school", K=5, verbose=verbose)
