from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse

from ..data_conversion import hye_list_to_binary_incidence
from ._custom_typing import hyperedge_type
from .hypergraph import Hypergraph


class ABHyeHypergraph(Hypergraph):
    """Representation of the hypergraph via three objects:
    - the array A, of size E. It contains the hyperedge weights.
    - the incidence matrix B, of size N x E, with N number of nodes in the hypergraph.
        It contains the mapping from a node to the hyperedges it belongs to.
    - the hyperedge list, of length E, where hyperedges are stored as tuples of nodes.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: Union[np.ndarray, sparse.spmatrix],
        hye: Optional[List[hyperedge_type]] = None,
        force_sparse: bool = False,
        input_check: bool = False,
    ):
        """
        Parameters
        ----------
        A: the hyperedge weights as an array of shape (E,).
        B: weighted incidence matrix, of shape (N, E).
        hye: hyperedge list.
            If not provided, it is inferred from B.
        force_sparse: whether to force the incidence matrix to be sparse or not.
            If the input is dense, it is converted to sparse.
        input_check: whether to check the coherence of the inputs.
            This procedure can be expensive for heavy inputs.
        """

        self.force_sparse = force_sparse

        if input_check:
            self._check_inputs(A, B, hye)

        self.A, self.B = self._cast_inputs_to_types(A, B)
        if hye is not None:
            self.hye = hye
        else:
            self.hye = create_hye_list(self.B)

        self.N, self.E = self.B.shape

        hye_lengths = map(len, self.hye)
        hye_counter = dict(Counter(hye_lengths))
        self.hye_count = hye_counter
        self.max_hye_size = max(hye_counter.keys())

    def _cast_inputs_to_types(
        self,
        A: np.ndarray,
        B: Union[np.ndarray, sparse.base.spmatrix],
    ) -> Tuple[np.ndarray, Union[np.ndarray, sparse.base.spmatrix]]:
        if isinstance(B, np.ndarray) and self.force_sparse:
            B = sparse.csr_matrix(B)
        return A, B

    @staticmethod
    def _check_inputs(
        A: np.ndarray,
        B: Union[np.ndarray, sparse.base.spmatrix],
        hye: Optional[List[hyperedge_type]],
    ) -> None:
        # check shapes
        N, E = B.shape
        assert A.shape == (E,), "Shapes of A and B don't match correctly."
        if hye is not None:
            assert len(hye) == E, "Shape of B and len of hye don't match correctly."

        if hye is not None:
            # infer hyperedges from B and check that they are the same as hye
            inferred_hye = set(create_hye_list(B))
            input_hye = set(hye)
            assert len(input_hye) == len(
                hye
            ), "There are repeated elements in the input hye."
            assert sorted(inferred_hye) == sorted(
                input_hye
            ), "The hyperedge list inferred from B is different from the input one."

        # check that weights in B and A coincide
        if sparse.issparse(B):
            assert np.all(
                (B > 0).multiply(A[None, :]).data == B.data
            ), "Hyperedge weights in A and B don't coincide."
        else:
            assert (
                (B > 0) * A[None, :] == B
            ).all(), "Hyperedge weights in A and B don't coincide."

    def get_repr(
        self,
    ) -> Union[
        np.ndarray, Union[np.ndarray, sparse.base.spmatrix], List[hyperedge_type]
    ]:
        return self.A, self.B, self.hye

    def get_binary_incidence_matrix(self) -> Union[np.ndarray, sparse.base.spmatrix]:
        return self.B > 0

    def get_incidence_matrix(self) -> Union[np.ndarray, sparse.base.spmatrix]:
        return self.B

    def get_hye_weights(self) -> np.ndarray:
        return self.A

    def sub_hyg(
        self,
        hyperedge_idx: Optional[np.ndarray] = None,
    ) -> ABHyeHypergraph:
        """Produce a sub-hypergraph where only the specified hyperedges are present.

        Parameters
        ----------
        hyperedge_idx: the list of the hyperedges to keep, specified by their indices.

        Returns
        -------
        The sub-hypergraph instance.
        """
        if hyperedge_idx is None:
            return self

        A = self.A[hyperedge_idx]
        B = self.B[:, hyperedge_idx]
        hye = [self.hye[idx] for idx in hyperedge_idx]

        return ABHyeHypergraph(A, B, hye)

    def __iter__(self):
        return zip(self.hye, self.A)

    def __str__(self):
        return f"{self.__class__.__name__} with N={self.N}, E={self.E}"

    @classmethod
    def load(
        cls,
        hye_file: Union[str, Path],
        weight_file: Union[str, Path],
        N: Optional[int] = None,
        **kwargs,
    ) -> ABHyeHypergraph:
        """Load a ABHyeHypergraph instance from two txt files, containing the list of
        hyperedges and weights.

        Parameters
        ----------
        hye_file: text file containing the hyperedges.
        weight_file: text file containing the hyperedges weights.
        N: number of nodes in the hypergraph.
        kwargs: keyword arguments to be passed to the initializer of ABHyeHypergraph.

        Returns
        -------
        An instance of ABHyeHypergraph.
        """
        with open(hye_file, "r") as file:
            hye = [tuple(map(int, line.split(" "))) for line in file.readlines()]

        A = np.loadtxt(weight_file)

        shape = (N, len(A)) if N else None
        B = (hye_list_to_binary_incidence(hye, shape=shape) * A).tocsr()

        return ABHyeHypergraph(A, B, hye, **kwargs)


def create_hye_list(B: Union[sparse.base.spmatrix, np.ndarray]) -> List[hyperedge_type]:
    """Create the hye list from the incidence matrix."""
    if sparse.issparse(B):
        if not isinstance(B, sparse.csr.csr_matrix):
            raise NotImplementedError(
                "Conversion to hye list not implemented for sparse type", type(B)
            )
        col_idx, row_idx = B.T.nonzero()
    else:
        nonzero_idx = np.argwhere(B.T > 0)
        row_idx = nonzero_idx[:, 1]
        col_idx = nonzero_idx[:, 0]

    hye = list()
    current_col = -1
    current_hye = list()
    for row, col in zip(row_idx, col_idx):
        if col == current_col:
            current_hye.append(row)
        else:
            if current_hye:
                hye.append(tuple(current_hye))
            current_hye = [row]
            current_col = col
    if current_hye:
        hye.append(tuple(current_hye))
    return hye
