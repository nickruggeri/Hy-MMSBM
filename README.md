<h1 align="center">
Hy-MMSBM <br/>  
<i>Hypergraph Mixed-Membership Stochastic Block Model</i>
</h1>

<p align="center">
<i>Inference and benchmarking of higher-order community structure</i>
</p>

<p align="center">
<a href="https://github.com/nickruggeri/Hy-MMSBM/blob/main/LICENSE" target="_blank">
<img alt="License: MIT" src="https://img.shields.io/github/license/nickruggeri/Hy-MMSBM">
</a>

<a href="https://www.python.org/" target="_blank">
<img alt="Made with Python" src="https://img.shields.io/badge/made%20with-python-1f425f.svg">
</a>

<a href="https://github.com/psf/black" target="_blank">
<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</a>

<a href="https://arxiv.org/abs/2301.11226" target="_blank">
<img alt="ARXIV: 2301.11226" src="https://img.shields.io/badge/arXiv-2301.11226-red.svg">
</a>

<a href="https://arxiv.org/abs/2212.08593" target="_blank">
<img alt="ARXIV: 2212.08593" src="https://img.shields.io/badge/arXiv-2212.08593-red.svg">
</a>

<a href="https://www.treedom.net/en/user/nicolo-ruggeri-7568/trees/V36-Y75D" target="_blank">
<img alt="Treedom" src="https://img.shields.io/badge/CO2%20compensation%20-Treedom%20%F0%9F%8C%B4-brightgreen">
</a>

</p>

This repository contains the implementation of the <i>Hy-MMSBM</i> model presented in 

&nbsp;&nbsp; 
[1] <i> Community Detection in Large Hypergraphs</i><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Nicolò Ruggeri, Martina Contisciani, Federico Battiston and Caterina De Bacco<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[<a href="https://arxiv.org/abs/2301.11226" target="_blank">ArXiv</a>]
        
&nbsp;&nbsp; 
[2] <i>A framework to generate hypergraphs with community structure</i><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Nicolò Ruggeri, Federico Battiston and Caterina De Bacco <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[<a href="https://arxiv.org/abs/2212.08593" target="_blank">ArXiv</a>]


Respectively, these works present efficient inference and sampling methods based on 
<i>Hy-MMSBM</i>, a stochastic block model for higher-order interactions. <br/> 
This code is made available for the public, if you make use of it please cite our work 
in the form of the references above.

<h2>Code installation</h2>

The code was developed utilizing <b>Python 3.9</b>, and can be downloaded and used locally as-is. <br>
To install the necessary packages, run the following command

`pip install -r requirements.txt`


<h2>Inference of community structure</h2>

The inference of the affinity matrix <i>w</i> and community assignments <i>u</i> is 
performed by running the code in `main_inference.py`. 

The most basic run only needs a dataset, the number of communities <i>K</i>, and a path to store the results. <br/>
For example, to perform inference on the Senate Bills dataset with <i>K=2</i> 
communities, one can run the following command:

`
python 
main_inference.py 
--K 2 --out_dir ./out_inference/senate_bills --real_dataset senate-bills
`

<h3>Input dataset format</h3>

It is possible to provide the input dataset in three formats.

__1. Real preprocessed dataset__<br/> 
We make all the real datasets we utilize in the 
experiments in [1] available, see the [data release section](#data-release). <br/>
After the download, any of these datasets can be utilized for inference by running the 
command above and specifying the dataset name using the `--real_dataset` argument. <br/>
The complete list of such datasets is available at
`src.data.data_io.PREPROCESSED_DATASETS`.

__2. Text format__<br/> 
Alternatively, a hypergraph can be provided as input via two *.txt* files,
containing the list of hyperedges, and the relative weights. 
This allows the user to provide arbitrary datasets as inputs.
We provide examples for the format expected in [./data/examples/justice_dataset/hyperedges.txt](./data/examples/justice_dataset/hyperedges.txt)
and [./data/examples/justice_dataset/weights.txt](./data/examples/justice_dataset/weights.txt). 
To perform inference on a dataset specified in text format, provide the path to the two 
files as 
```python
python main_inference.py 
--K 2 
--out_dir ./out_inference/justice 
--hyperedge_file ./data/examples/justice_dataset/hyperedges.txt 
--weight_file ./data/examples/justice_dataset/weights.txt
```
In this case, the command above is equivalent to running with `--real_dataset justice`.

__3. Pickle format__<br/>
Finally, one can provide a `Hypergraph` instance, which is the main representation 
utilized internally in the code (see `src.data.representation`), serialized via the 
<a href="https://docs.python.org/3/library/pickle.html">pickle</a> Python library. <br/>
An example equivalent to the above is
```python
python main_inference.py 
--K 2 
--out_dir ./out_inference/justice 
--pickle_file ./data/examples/justice_dataset/justice.pkl
```
Similarly to the text format, this allows to provide arbitrary hypergraphs as input.

<h3>Additional options</h3>

Additional options can be specified, the full documentation is shown by running
        
`python main_inference.py --help`

Among the important ones we list:
- `--assortative` whether to run inference with a diagonal affinity matrix <i>w</i>.  
- `--max_hye_size` to keep only hyperedges up to a given size for inference. If `None`, all hyperedges are utilized.
- `--w_prior` and `--u_prior` the rates for the exponential priors on the parameters. A value of zero is equivalent to no prior, any positive value is utilized for MAP inference. <br/>
For non-uniform priors, the path to a file containing a NumPy array can be specified, which will be loaded via `numpy.load`.
- `--em_rounds` number of EM steps during optimization. It is sometimes useful when the model doesn't converge rapidly.
- `--training_rounds` the number of models to train with different random initializations. The one with the highest log-likelihood is returned and saved.
- `--seed` integer random seed.


<h2>Sampling of synthetic data</h2>

Sampling of synthetic data from the generative model can be performed by running the 
code in `main_sampling.py`. As explained in the reference paper [2], various inputs can 
be provided to condition the sampling procedure. <br/>
We also refer to the notebook for additional examples. 

<h3>Vanilla sampling</h3>

The simplest form of sampling, simply requires the affinity matrix <i>w</i> and 
community assignments <i>u</i>. These need to be provided in text files to be opened 
via `numpy.loadtxt`, for example
```python
python main_sampling.py 
--w ./data/examples/fixed_dimension_sequence_hard/K=2/w.txt 
--u ./data/examples/fixed_dimension_sequence_hard/K=2/u.txt 
--out_dir ./out_sampling
```

<h3>Degree and size sequences</h3>

Samples can also be conditioned on the degree sequence (i.e. the array specifying each 
node's degree) , size sequence (i.e. the count of hyperedges for any given size), or both.<br/>
These need to be stored in text files, the degree sequence via `numpy.savetxt`, and the 
size sequence in a text file specifying, for every line, the size of the hyperedges and 
the number of the hyperedges of such size. Examples for these formats are given in 
[./data/examples](./data/examples) at `fixed_dimension_sequence_hard` and `justic_dataset`. <br/>
The command line arguments are specified as follows:
```python
python main_sampling.py 
--w ./data/examples/fixed_dimension_sequence_hard/K=2/w.txt 
--u ./data/examples/fixed_dimension_sequence_hard/K=2/u.txt 
--out_dir ./out_sampling
--deg_seq ./data/examples/fixed_dimension_sequence_hard/K=2/deg_seq.txt
--dim_seq ./data/examples/fixed_dimension_sequence_hard/K=2/dim_seq.txt
```
Notice that, if conditioning on both the sequences, and the sequences are extracted from 
an already available dataset, it is more convenient and computationally faster to 
directly provide the dataset as explained in the following section.


<h3>Conditioning on a dataset</h3>

Conditioning the sampling on an existing dataset means having the MCMC procedure start 
from the dataset itself, and is statistically equivalent to conditioning on its degree
and size sequences. However, it is computationally faster since the sequences don't need
to be arranged into a first hypergraph proposal (see reference paper [2]).<br/>
The format to provide a dataset as input is the same as the one specified in the 
[relative section](#input-dataset-format) of the inference procedure.


<h3>Additional arguments</h3>

Additional arguments that can be provided are:
- `--max_hye_size` the maximum hyperedges size allowed.  
- `--burn_in_steps` and `--intermediate_steps` the number of MCMC steps to be performed 
before returning the first sample, and in between returned samples.
- `--exact_dyadic_sampling` whether to perform the sampling of dyadic interaction 
(i.e. standard edges between two nodes) exactly from their Poisson distribution, as opposed to approximate Central Limit Theorem-based sampling.
- `--allow_rescaling` in case the degree or dimension sequences are provided, whether to 
rescale the model parameters to match the expected sequences with the input ones.
- `--n_samples` number of generated samples.
- `--seed` integer random seed.


<h2>Data release</h2>

All the preprocessed real datasets utilized in [1, 2], as well as some of the synthetic 
data generated for various experiments, are publicly available. <br/>
The data is stored and distributed via <a href="https://edmond.mpdl.mpg.de/">Edmond</a>, 
the Open Research Data Repository of the Max Planck Society, and is available at the 
following 
<a href="https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.HRW0OE&version=1.0">link</a>.

Alternatively, it can be directly downloaded 
(approximately 1.7GB compressed, then uncompressed to 12.5GB), 
by running the relative script as <br/> 
`python download_data.py`
 
