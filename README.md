# clusters-in-language-evaluation

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/rycolab/clusters-in-language-evaluation/tree/main.svg?style=svg&circle-token=555ae1e5333644c90e3364a2959808171ff39d0d)](https://dl.circleci.com/status-badge/redirect/gh/rycolab/clusters-in-language-evaluation/tree/main)

Code accompanying the ICLR 2023 publication "On the Usefulness of Embeddings, Clusters and Strings for Text Generator Evaluation".

## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip install transformers
```

Then install the local version of mauve:
```bash
$ git submodule update --init --recursive
$ pip install -e ./mauve
```

Finally build the library in the kenlm submodule
```bash
$ cd kenlm
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j 4
```
And install it to be usable in python:
```bash
$ pip install https://github.com/kpu/kenlm/archive/master.zip
```

## Run Experiments

### Process Data

To download the text data that will be needed for the experiments run:
```bash
$ make get_text MODEL=human
$ make get_text MODEL=<model>
$ make get_representations MODEL=human REPRESENTATIONS=<representations>
$ make get_representations MODEL=<model> REPRESENTATIONS=<representations>
```
where `<model>` is the model you want to analyse text from, and `<representations>` is the model from which you want the embeddings for evaluation: ['gpt2-small', 'gpt2', 'gpt2-large', 'gpt2-xl'].


### Train models


#### To estimate a language model

To estimate the language models `p_w` and `q_w` on the data:
```bash
$ make train_text MODEL=human LANGMODEL=<langmodel>
$ make train_text MODEL=<model> LANGMODEL=<langmodel>
$ make eval_text MODEL=<model> LANGMODEL=<langmodel>
```
where `<langmodel>` can be one of an: ['ngram', 'lstm'].

Then, to compute mauve and other divergences on text distributions:
```bash
$ make analyse_texts MODEL=<model> LANGMODEL=<langmodel>
```

#### To estimate a k-means model and get clusters

To fit k-means on the embeddings, and get clusters (for both the model and human text) run:
```bash
$ make get_clusters MODEL=<model> REPRESENTATIONS=<representations>
```

Then, to estimate the cluster distributions `p_c` and `q_c` and evaluate them on the data:
```bash
$ make train_clusters MODEL=<model> REPRESENTATIONS=<representations>
$ make eval_clusters MODEL=<model> REPRESENTATIONS=<representations>
```

To compute mauve and other divergences on cluster distributions:
```bash
$ make analyse_clusters MODEL=<model> REPRESENTATIONS=<representations>
```

### Compile and analyse resuls

Finally, after you are done evaluating text and clusters for all models of interest (`small-117M`, `small-117M-p0.9`, `medium-345M`, `medium-345M-p0.9`, `large-762M`, `large-762M-p0.95`, `xl-1542M`, `xl-1542M-p0.95`), compile all results by running:
```bash
$ make get_results
```

Now you can compare to human text by running:
````bash
$ make compare
$ make plot
````

## Extra Information

#### Citation

Here is the bibtex citation for our paper:
```json
@inproceedings{pimentel2023on,
	title={On the Usefulness of Embeddings, Clusters and Strings for Text Generation Evaluation},
	author={Tiago Pimentel and Clara Isabel Meister and Ryan Cotterell},
	booktitle={The Eleventh International Conference on Learning Representations },
	year={2023},
	url={https://openreview.net/forum?id=bvpkw7UIRdU}
}
```

#### Contact

To ask questions or report problems, please open an [issue](https://github.com/rycolab/clusters-in-language-evaluation/issues).
