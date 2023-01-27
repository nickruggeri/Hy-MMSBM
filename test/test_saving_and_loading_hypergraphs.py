from pathlib import Path

import numpy as np

from src.data.data_io import load_real_hypergraph
from src.data.representation.abhye_hypergraph import ABHyeHypergraph
from src.data.representation.ahye_hypergraph import AHyeHypergraph

tmp_savedir = Path("./tmp")
for dataset in ["justice", "workspace1", "high_school"]:
    hyg = load_real_hypergraph(dataset)

    (tmp_savedir / dataset).mkdir(parents=True, exist_ok=True)
    hyg.save_to_txt(tmp_savedir / dataset)

    np.savetxt(
        str(tmp_savedir / dataset / "np_weights.txt"), hyg.get_hye_weights().astype(int)
    )

    loaded_ahye = AHyeHypergraph.load(
        tmp_savedir / dataset / "hyperedges.txt",
        tmp_savedir / dataset / "weights.txt",
    )

    assert np.all(loaded_ahye.get_hye_weights() == hyg.get_hye_weights())
    assert loaded_ahye.N == hyg.N

    loaded_abhye = ABHyeHypergraph.load(
        tmp_savedir / dataset / "hyperedges.txt",
        tmp_savedir / dataset / "weights.txt",
        input_check=True,
    )
    assert (loaded_abhye.get_incidence_matrix() == hyg.get_incidence_matrix()).all()
