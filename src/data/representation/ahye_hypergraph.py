from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse

from ._custom_typing import hyperedge_type
from .abhye_hypergraph import ABHyeHypergraph
from .hypergraph import Hypergraph
from ..data_conversion import hye_list_to_binary_incidence


class AHyeHypergraph(Hypergraph):
    """Representation of the hypergraph via two objects:
    - the array A, of size E. It contains the hyperedge weights.
    - the list hye, of length E, containing hyperedges represented as tuples of nodes.
    """

    def __init__(
        self,
        A: np.ndarray,
        hye: List[hyperedge_type] = None,
        N: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        A: the hyperedge weights as an array of shape (E,).
        hye: hyperedge list.
            If not provided, it's inferred from B.
        N: number of nodes in the hypergraph. If not provided, it is inferred from hye.
        """

        self.N = self._check_inputs_and_infer_N_if_needed(A, hye, N)
        self.A = A
        self.hye = hye
        self.E = len(self.hye)

        hye_lengths = map(len, self.hye)
        hye_counter = dict(Counter(hye_lengths))
        self.hye_count = hye_counter
        self.max_hye_size = max(hye_counter.keys()) if self.E > 0 else 0

    @staticmethod
    def _check_inputs_and_infer_N_if_needed(
        A: np.ndarray,
        hye: Optional[List[hyperedge_type]],
        N: Optional[int],
    ) -> int:
        E = len(hye)
        assert A.shape == (E,), "Shapes of A and hyperedge list don't match correctly."

        if E > 0:
            inferred_N = max(map(max, hye)) + 1  # Assuming 0 based indexing
            if N is not None:
                if N < inferred_N:
                    raise ValueError(
                        "The N provided is lower than some node indices "
                        "found in the hyperedge list."
                    )
                return N
            return inferred_N
        else:
            if N is None:
                raise ValueError(
                    "Cannot infer N from an empty input. Provide N explicitly."
                )
            return N

    def get_repr(self) -> Tuple[np.ndarray, List[hyperedge_type]]:
        return self.A, self.hye

    def get_binary_incidence_matrix(self) -> sparse.csr.csr_matrix:
        return self._create_binary_incidence_matrix()

    def get_incidence_matrix(self) -> sparse.csr.csr_matrix:
        binary_incidence = self._create_binary_incidence_matrix()

        incidence = binary_incidence.multiply(self.A).tocsr()
        assert incidence.shape == binary_incidence.shape
        assert isinstance(incidence, sparse.csr.csr_matrix)
        return incidence

    def get_hye_weights(self) -> np.ndarray:
        return self.A

    def sub_hyg(
        self,
        hyperedge_idx: Optional[np.ndarray] = None,
    ) -> AHyeHypergraph:
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
        hye = [self.hye[idx] for idx in hyperedge_idx]

        return AHyeHypergraph(A, hye, self.N)

    def to_abhye_hypergraph(self) -> ABHyeHypergraph:
        B = self.get_incidence_matrix()
        assert isinstance(B, sparse.csr.csr_matrix)
        return ABHyeHypergraph(self.A, B, self.hye, input_check=False)

    def __iter__(self):
        return zip(self.hye, self.A)

    def __str__(self):
        return f"{self.__class__.__name__} with N={self.N}, E={self.E}"

    def _create_binary_incidence_matrix(self) -> sparse.csr.csr_matrix:
        """Create incidence matrix from the hyperedge list."""
        return hye_list_to_binary_incidence(self.hye, shape=(self.N, self.E)).tocsr()

    @classmethod
    def load(
        cls,
        hye_file: Union[str, Path],
        weight_file: Union[str, Path],
        N: Optional[int] = None,
    ) -> AHyeHypergraph:
        """Load a AHyeHypergraph instance from two txt files, containing the list of
        hyperedges and weights.

        Parameters
        ----------
        hye_file: text file containing the hyperedges.
        weight_file: text file containing the hyperedges weights.
        N: number of nodes in the hypergraph.

        Returns
        -------
        An instance of AHyeHypergraph.
        """
        with open(hye_file, "r") as file:
            hye = [tuple(map(int, line.split(" "))) for line in file.readlines()]

        A = np.loadtxt(weight_file)

        return AHyeHypergraph(A, hye)
