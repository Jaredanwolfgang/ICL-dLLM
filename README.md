# ICL-dLLM

This repository is the official implementation of the paper: **In-Context Learning in Diffusion Models: A Comparative Analysis with Transformers**, which is forked from the repository of [In-Context Learning](https://github.com/dtsip/in-context-learning).

## Overview

This project investigates in-context learning capabilities in diffusion models and provides a comparative analysis with transformer-based approaches. We implement diffusion-based models (both encoder and decoder variants) for various regression and classification tasks, exploring how diffusion models can learn from context examples in a few-shot setting.

## Features

- **Diffusion Models for ICL**: Implementation of diffusion encoder and decoder models for in-context learning
- **Transformer Baselines**: GPT2-based transformer models for comparison
- **Multiple Task Support**: Linear regression, sparse linear regression, ReLU networks, decision trees, and more
- **Flexible Architecture**: Support for both encoder and decoder-style diffusion models

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd in-context-learning
   ```

2. Install dependencies using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate in-context-learning
   ```

3. [Optional] If you plan to train models, populate `src/conf/wandb.yaml` with your wandb information.

## Getting Started

### Training

Train a model using a configuration file from `src/conf/`:

```bash
cd src
python train.py --config conf/toy.yaml
```

Available configurations include:
| Configuration | Linear Regression | Sparse Linear Regression | ReLU 2-layer Neural Network Regression | Decision Tree |
|---------------|------------------|--------------------------|---------------------------------------|---------------|
| GPT-2 | `linear_regression.yaml`  | `sparse_linear_regression.yaml`  | `relu_2nn_regression.yaml`  | `decision_tree.yaml`  |
| Diffusion Encoder | `base_encoder.yaml` |  | `relu2nn_encoder.yaml` | `dt_encoder.yaml` |
| Diffusion Decoder | `base_decoder.yaml` |  | `relu2nn_decoder.yaml` | `dt_decoder.yaml` |

### Evaluation

The evaluation notebooks in `src/plot/` contain code to:
- Load pre-trained models
- Plot pre-computed metrics
- Evaluate models on new data

## Collaborators

- [Moxin Tang](moxintang@berkeley.edu) Student ID: 3041997936
- [Ruizhe Song](https://sites.google.com/d/1PL4Y2rzwtcqWJNr0NMBx6I85Kj3Ond1N/p/14yPj_OhCA7zwsBR_xmdJ8rJmr51V0c-z/edit) Student ID: 3042013171
- [Weiyi Zhang](https://v1zhang.github.io/) Student ID: 3042031814
- [Yicheng Xiao](https://jaredanwolfgang.github.io/about/) Student ID: 3042011144

## Acknowledgments

This work is based on the original [In-Context Learning](https://github.com/dtsip/in-context-learning) repository by Garg et al.
