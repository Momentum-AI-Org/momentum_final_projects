# Momentum Final Projects

This repo contains source code and colab notebooks for all Momentum AI final project templates. Projects targeted at 2 days in length and aim to guide students through the process of making a simple image classifier.


## Usage

There are 9 colabs for students to choose from in the `notebooks` folder.
Notebooks are made to be imported into and run on [Google Colab](https://colab.research.google.com/).
Datasets are hosted publicly on dropbox and will be automatically downloaded by the given scripts.


## Dev Environment Setup

First, create and activate a conda environment:
```bash
conda create -y -n momentum python="3.8.8"
conda activate momentum
```

Install requirements:
```bash
pip install -r requirements.txt && pip install -e .
```

Finally, if you wish to contribute to the repo, please install pre-commit hooks:
```
pip install pre-commit
pre-commit install
```
