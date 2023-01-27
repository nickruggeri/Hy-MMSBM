import pickle as pkl
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import sparse

from src.data.representation.abhye_hypergraph import ABHyeHypergraph
from src.data.representation.hypergraph import Hypergraph

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"

PREPROCESSED_DATASETS = [
    "arxiv",
    "amazon_5core",
    "curated_gene_disease_associations",
    "enron-email",
    "high_school",
    "hospital",
    "house-bills",
    "house-committees",
    "justice",
    "primary_school",
    "senate-bills",
    "senate-committees",
    "trivago-clicks_2core",
    "trivago-clicks_5core",
    "trivago-clicks_10core",
    "walmart-trips_2core",
    "walmart-trips_3core",
    "walmart-trips_4core",
    "workspace1",
]


def load_real_hypergraph(
    dataset: str,
    **kwargs: Optional[Any],
) -> ABHyeHypergraph:
    """Load a real-world hypergraph.

    Parameters
    ----------
    dataset: name of the hypergraph: It can be one of the available preprocessed
        datasets or a synthetic graph model.
    kwargs: keyword arguments to be passed to the hypergraph instance created.

    Returns
    -------
    hypergraph: the loaded or generated hypergraph.
    """
    if dataset not in PREPROCESSED_DATASETS:
        raise ValueError(
            f"Dataset unknown: {dataset}."
            f"\nThe available datasets are: \n{PREPROCESSED_DATASETS}"
        )

    filename = DEFAULT_DATA_DIR / "preprocessed_real_data" / f"{dataset}.npz"
    data = np.load(filename, allow_pickle=True)
    A = data["A"]
    B = data["B"]
    hye = data["hyperedges"]

    # When saving sparse arrays via np.savez, they are stored inside a numpy array with
    # null shape. Manage these cases for sparse incidence B.
    if not B.shape:
        B = B.reshape(1)[0]
        assert isinstance(B, sparse.spmatrix)

    return ABHyeHypergraph(A, B, hye, **kwargs)


def load_data(
    real_dataset: str = "",
    hye_file: str = "",
    weight_file: str = "",
    pickle_file: str = "",
) -> Hypergraph:
    """Load a hypergraph dataset.
    Utility function for loading hypergraph data provided in various formats.
    Currently three formats are supported:
    - a string with the name of a real dataset
    - a pair (hye_file, weight_file) specifying the hyperedges and relative weights
    - the path to a serialized hypergraph, to be loaded via the pickle package.

    The function raises an error if more than one of the options above is given as
    input.

    Parameters
    ----------
    real_dataset: name of one the supported real datasets
    hye_file: txt file containing the hyperedges in the dataset.
        If provided, also weight_file needs to be provided.
    weight_file:  txt file containing the hyperedge weights in the dataset.
        If provided, also hye_file needs to be provided.
    pickle_file: path to a .pkl file to be loaded via the pickle package.

    Returns
    -------
    The loaded hypergraph.
    """
    # Check that the data is provided exactly in one format:
    # - as a real real_dataset name
    # - in the form of two files, specifying the hyperedges and relative weights
    # - in the form of a pickle file, containing a serialized hypergraph
    inputs = (
        bool(real_dataset) + (bool(hye_file) or bool(weight_file)) + bool(pickle_file)
    )
    if inputs == 0:
        raise ValueError("no input hypergraph has been provided.")
    if inputs >= 2:
        raise ValueError("Provide only one valid input hypergraph format.")

    if real_dataset:
        if real_dataset in PREPROCESSED_DATASETS:
            return load_real_hypergraph(real_dataset, force_sparse=True)
        raise ValueError("Real real_dataset unknown:", real_dataset)

    if pickle_file:
        with open(pickle_file, "rb") as file:
            return pkl.load(file)

    if hye_file or weight_file:
        if not hye_file and weight_file:
            raise ValueError("Provide both the hyperedge and weight files.")
        return ABHyeHypergraph.load(hye_file, weight_file)
