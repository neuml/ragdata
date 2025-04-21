# ragdata: Build knowledge bases for RAG

<p align="center">
    <a href="https://github.com/neuml/ragdata/releases">
        <img src="https://img.shields.io/github/release/neuml/ragdata.svg?style=flat&color=success" alt="Version"/>
    </a>
    <a href="https://github.com/neuml/ragdata/releases">
        <img src="https://img.shields.io/github/release-date/neuml/ragdata.svg?style=flat&color=blue" alt="GitHub Release Date"/>
    </a>
    <a href="https://github.com/neuml/ragdata/issues">
        <img src="https://img.shields.io/github/issues/neuml/ragdata.svg?style=flat&color=success" alt="GitHub issues"/>
    </a>
    <a href="https://github.com/neuml/ragdata">
        <img src="https://img.shields.io/github/last-commit/neuml/ragdata.svg?style=flat&color=blue" alt="GitHub last commit"/>
    </a>
</p>

`ragdata` builds knowledge bases for Retrieval Augmented Generation (RAG).

This project has processes to build [txtai](https://github.com/neuml/txtai) embeddings databases for common datasets.

The currently supported datasets are:

- [ArXiv](https://huggingface.co/NeuML/txtai-arxiv)
- [Wikipedia](https://huggingface.co/NeuML/txtai-wikipedia)

Each of the links above has full instructions on how to build those datasets, including using this project.

## Installation
The easiest way to install is via pip and PyPI

```
pip install ragdata
```

Python 3.10+ is supported. Using a Python [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

`ragdata` can also be installed directly from GitHub to access the latest, unreleased features.

```
pip install git+https://github.com/neuml/ragdata
```
