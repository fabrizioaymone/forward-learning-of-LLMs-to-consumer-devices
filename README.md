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
```

## Citation
If you find <em><strong>Forward Learning of Large Language Models by Consumer Devices</strong></em> helpful for your research, please consider citing the paper.

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
DOI = {10.3390/electronics13020402}}
```

A more thorough mathematical description of the computational complexity metrics attributed to each operation involved in the Transformer training is provided in <em><strong>Mathematical Formulation of Learning and Its Computational Complexity for Transformers’ Layers</strong></em>.

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
DOI = {10.3390/eng5010003}}
```


