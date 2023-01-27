from __future__ import annotations

import logging
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import yaml

from src.argparse_types import bool_type, float_or_none, float_or_str, int_or_none
from src.data.data_io import load_data
from src.model.model import HyMMSBM

if __name__ == "__main__":
    parser = ArgumentParser()
    # Model parameters
    parser.add_argument(
        "--K",
        type=int,
        help="Number of communities in the model.",
    )
    parser.add_argument(
        "--assortative",
        type=bool_type,
        default=False,
        help="Whether to use a diagonal or full affinity matrix.",
    )
    # Dataset
    parser.add_argument(
        "--real_dataset",
        type=str,
        default="",
        help="The name of a real dataset.",
    )
    parser.add_argument(
        "--hyperedge_file",
        type=str,
        default="",
        help="The path to a txt file containing the hyperedges.",
    )
    parser.add_argument(
        "--weight_file",
        type=str,
        default="",
        help="The path to a txt file containing the weights of the hyperedges.",
    )
    parser.add_argument(
        "--pickle_file",
        type=str,
        default="",
        help="The path to a pickle file containing a hypergraph representation.",
    )
    parser.add_argument(
        "--max_hye_size",
        type=int_or_none,
        default=None,
        help=(
            "Utilize hyperedges up to a maximum size. If None, use all the hyperedges."
        ),
    )
    # Training parameters
    parser.add_argument("--seed", type=int_or_none, default=None, help="Random seed.")
    parser.add_argument(
        "--em_rounds", type=int, default=100, help="Maximum number of EM iterations."
    )
    parser.add_argument(
        "--tolerance",
        type=float_or_none,
        default=None,
        help="Tolerance for training stopping criterion.",
    )
    parser.add_argument(
        "--check_convergence_every",
        type=int,
        default=10,
        help="EM iterations between consecutive checks for convergence.",
    )
    parser.add_argument(
        "--u_prior",
        type=float_or_str,
        default=0.0,
        help=(
            "Prior for u. It can be a float, or the path to a numpy file. "
            "A value of 0. corresponds to no prior on u. "
            "If a path to a file is provided, it is going to be loaded with numpy.load "
            "and the array contained must have shape NxK, specifying the element-wise "
            "value of the exponential rate in the prior."
        ),
    )
    parser.add_argument(
        "--w_prior",
        type=float_or_str,
        default=1.0,
        help=(
            "Prior for w. It can be a float, or the path to a numpy file. "
            "A value of 0. corresponds to no prior on w. "
            "If a path to a file is provided, it is going to be loaded with numpy.load "
            "and the array contained must have shape KxK, specifying the element-wise "
            "value of the exponential rate in the prior. "
            "By definition, the matrix prior must be symmetric."
        ),
    )
    # Validation parameters
    parser.add_argument(
        "--training_rounds",
        type=int,
        default=10,
        help="Number of models to train. The best among these is chosen and saved.",
    )
    # Result saving
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Directory where results are saved."
    )
    args = parser.parse_args()

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load hypergraph
    hypergraph = load_data(
        args.real_dataset,
        args.hyperedge_file,
        args.weight_file,
        args.pickle_file,
    )

    # Keep only hyperedges with a given max degree
    if args.max_hye_size is not None:
        if args.max_hye_size > hypergraph.max_hye_size:
            logging.warning(
                "max_hye_size provided in argparse argument is greater than the max "
                "size in the input hypergraph. No hyperedge will be removed."
            )
        else:
            hye_idx = np.array(
                [
                    idx
                    for idx, hye_size in zip(
                        range(hypergraph.E), (len(hye) for hye, _ in hypergraph)
                    )
                    if hye_size <= args.max_hye_size
                ]
            )
            hypergraph = hypergraph.sub_hyg(hye_idx)

    # Prior parameters
    u_prior = args.u_prior
    if isinstance(u_prior, str):
        u_prior = np.load(u_prior)
    w_prior = args.w_prior
    if isinstance(w_prior, str):
        w_prior = np.load(w_prior)

    # Train some models with different random initializations,
    # choose the best one in terms of likelihood.
    best_model = None
    best_loglik = float("-inf")
    for j in range(args.training_rounds):
        model = HyMMSBM(
            K=args.K,
            assortative=args.assortative,
            max_hye_size=args.max_hye_size,
            u_prior=u_prior,
            w_prior=w_prior,
        )
        model.fit(
            hypergraph,
            n_iter=args.em_rounds,
            tolerance=args.tolerance,
            check_convergence_every=args.check_convergence_every,
        )

        log_lik = model.log_likelihood(hypergraph)
        if log_lik > best_loglik:
            best_model = model
            best_loglik = log_lik

    # Save results.
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "args.yaml", "w") as file:
            yaml.dump(vars(args), file)

        np.savetxt(out_dir / "inferred_w.txt", best_model.w)
        np.savetxt(out_dir / "inferred_u.txt", best_model.u)
