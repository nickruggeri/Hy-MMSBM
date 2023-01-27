"""Convenience functions for changing the data representation format."""
from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse


def hye_list_to_binary_incidence(
    hye_list: Union[List[List[int]], List[Tuple[int]]],
    shape: Optional[Tuple[int]],
) -> sparse.spmatrix:
    """Convert a list of hyperedges into a scipy sparse csc array.

    Parameters
    ----------
    hye_list: the list of hyperedges.
        Every hyperedge is represented as either a tuple or list of nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    """
    # See docs:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html
    len_list = [0] + [len(hye) for hye in hye_list]
    indptr = np.cumsum(len_list)

    type_ = type(hye_list[0])
    indices = type_(chain(*hye_list))

    data = np.ones_like(indices)

    return sparse.csc_array((data, indices, indptr), shape=shape)
