# Forward Learning of Large Language Models on Consumer Devices

This repository contains supplementary material for the paper titled **"Forward Learning of Large Language Models by Consumer Devices"**. The paper explores memory and computational complexity of different learning algorithms applied to Transformer models and investigates their applicability to consumer edge devices.

## Overview

The paper evaluates various learning algorithms, namely Backpropagation (BP), PEPITA, and MEMPEPITA, in terms of their computational and memory complexity when applied to Transformer-based Large Language Models (LLMs).

## Repository Contents

- **LICENSE**: Contains the licensing information for this repository.
- **README.md**: This file, providing an overview of the repository.
- **decoder_only_model.py**: Code for the decoder-only Transformer model.
- **encoder_decoder_model.py**: Code for the encoder-decoder Transformer model.
- **encoder_only_model.py**: Code for the encoder-only Transformer model.
- **requirements.txt**: Lists the dependencies required to run the code.
- **results.py**: Scripts related to generating or analyzing results from the models.
- **blocks**: Directory containing additional scripts for modeling the memory consumption and computational complexity of transformer layers.

## Installation

To run the code and replicate the results from the paper, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Running the Memory and Complexity Model and Analyzing Results

To analyze the results from the models, use the following command:

```bash
python results.py

## Citation
If you find "Forward Learning of Large Language Models by Consumer Devices" helpful for your research, please consider citing the paper.

```
@Article{electronics13020402,
AUTHOR = {Pau, Danilo Pietro and Aymone, Fabrizio Maria},
TITLE = {Forward Learning of Large Language Models by Consumer Devices},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {2},
ARTICLE-NUMBER = {402},
URL = {https://www.mdpi.com/2079-9292/13/2/402},
ISSN = {2079-9292},
DOI = {10.3390/electronics13020402}
}
```

A more thorough mathematical description of the computational complexity metrics attributed to each operation involved in the Transformer training is provided in "Mathematical Formulation of Learning and Its Computational Complexity for Transformers’ Layers".

```
@Article{eng5010003,
AUTHOR = {Pau, Danilo Pietro and Aymone, Fabrizio Maria},
TITLE = {Mathematical Formulation of Learning and Its Computational Complexity for Transformers’ Layers},
JOURNAL = {Eng},
VOLUME = {5},
YEAR = {2024},
NUMBER = {1},
PAGES = {34--50},
URL = {https://www.mdpi.com/2673-4117/5/1/3},
ISSN = {2673-4117},
ABSTRACT = {Transformers are the cornerstone of natural language processing and other much more complicated sequential modelling tasks. The training of these models, however, requires an enormous number of computations, with substantial economic and environmental impacts. An accurate estimation of the computational complexity of training would allow us to be aware in advance about the associated latency and energy consumption. Furthermore, with the advent of forward learning workloads, an estimation of the computational complexity of such neural network topologies is required in order to reliably compare backpropagation with these advanced learning procedures. This work describes a mathematical approach, independent from the deployment on a specific target, for estimating the complexity of training a transformer model. Hence, the equations used during backpropagation and forward learning algorithms are derived for each layer and their complexity is expressed in the form of MACCs and FLOPs. By adding all of these together accordingly to their embodiment into a complete topology and the learning rule taken into account, the total complexity of the desired transformer workload can be estimated.},
DOI = {10.3390/eng5010003}
}
```


