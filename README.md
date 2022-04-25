# tensortyping tutorial


## Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adama/torchtyping-tutorial/blob/master/torchtyping_tutorial.ipynb)

## Code Example

Install deps with conda. `conda env create -f environment.yml`.

Train an XOR classifier with `python try_torchtyping.py`.

Test the XOR classifier code with `pytest --torchtyping-patch-typeguard --tb=line` or `pytest --torchtyping-patch-typeguard --tb=short`.