import pickle as pkl
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import yaml

from src.argparse_types import bool_type, int_or_none
from src.data.data_io import load_data
from src.model.model import HyMMSBM

if __name__ == "__main__":
    parser = ArgumentParser()
    # Sampling configuration
    parser.add_argument(
        "--w", type=str, help="Path to the file containing the affinity matrix."
    )
    parser.add_argument(
        "--u", type=str, help="Path to the file containing the community assignments."
    )
    parser.add_argument(
        "--deg_seq",
        type=str,
        default="",
        help="Path to the file containing the degree sequence.",
    )
    parser.add_argument(
        "--dim_seq",
        type=str,
        default="",
        help="Path to the file containing the dimension sequence.",
    )
    parser.add_argument(
        "--max_hye_size",
        type=int_or_none,
        default=None,
        help="Maximum hyperedge size in the generated samples.",
    )
    # Arguments related an existing dataset to start the MCMC.
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
        "--burn_in_steps",
        type=int,
        default=1000,
        help="Burn in steps for MCMC sampling.",
    )
    parser.add_argument(
        "--intermediate_steps",
        type=int,
        default=1000,
        help="Burn in steps for MCMC sampling.",
    )
    parser.add_argument(
        "--exact_dyadic_sampling",
        type=bool_type,
        default=True,
        help=(
            "Whether to perform exact or approximate sampling "
            "for the binary interactions."
        ),
    )
    parser.add_argument(
        "--allow_rescaling",
        type=bool_type,
        default=True,
        help=(
            "In case the degree and/or the dimension sequence is provided, "
            "whether to rescale the model parameters to match the provided "
            "sequences in expected value."
        ),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of hypergraphs to sample from the same Markov chain.",
    )
    # Random seed
    parser.add_argument("--seed", type=int_or_none, default=None, help="Random seed.")
    # Results saving
    parser.add_argument(
        "--out_dir", type=str, help="Directory where to save the sampling results."
    )
    args = parser.parse_args()

    # Random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load input parameters.
    w = np.loadtxt(args.w)
    u = np.loadtxt(args.u)
    N, K = u.shape
    if not w.shape == (K, K):
        raise ValueError("Shapes of w and u are incompatible.")

    if args.max_hye_size is not None:
        max_hye_size = args.max_hye_size
    else:
        max_hye_size = N

    # Load input degree and dimension sequences.
    if args.deg_seq:
        deg_seq = np.loadtxt(args.deg_seq).astype(int)
    else:
        deg_seq = None

    if args.dim_seq:
        with open(args.dim_seq, "r") as file:
            dim_seq = dict(map(int, line.split(" ")) for line in file.readlines())
    else:
        dim_seq = None

    if dim_seq is not None and max(dim_seq) > max_hye_size:
        raise ValueError(
            "The dimension sequences specifies hyperedges bigger than max_hye_size."
        )
    if deg_seq is not None and len(deg_seq) != N:
        raise ValueError(
            f"The degree sequence has length {len(deg_seq)} different from N={N}."
        )

    # Load pre-existing dataset to condition the MCMC.
    if any(
        (args.real_dataset, args.hyperedge_file, args.weight_file, args.pickle_file)
    ):
        hypergraph = load_data(
            args.real_dataset, args.hyperedge_file, args.weight_file, args.pickle_file
        )
        if hypergraph.N != N:
            raise ValueError(
                f"The conditioning hypergraph contains N={hypergraph.N} nodes, "
                f"which is different from the shape of u {u.shape}."
            )
        if hypergraph.max_hye_size > max_hye_size:
            raise ValueError(
                "The input hypergraph contains hyperedges "
                f"up to size {hypergraph.max_hye_size}, "
                f"while the max_hye_size specified is {max_hye_size}."
            )
        initial_config = [set(hye) for hye, _ in hypergraph]
        del hypergraph
    else:
        initial_config = None

    # Sample.
    model = HyMMSBM(K, u, w, assortative=None, max_hye_size=max_hye_size)
    samples = model.sample(
        deg_seq=deg_seq,
        dim_seq=dim_seq,
        initial_config=initial_config,
        allow_rescaling=args.allow_rescaling,
        exact_dyadic_sampling=args.exact_dyadic_sampling,
        burn_in_steps=args.burn_in_steps,
        intermediate_steps=args.intermediate_steps,
    )

    # Save generated data and input arguments.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        sample, _ = next(samples)
        with open(out_dir / f"sample_{i}.pkl", "wb") as file:
            pkl.dump(sample, file)

    with open(out_dir / "args.yaml", "w") as file:
        yaml.dump(vars(args), file)
