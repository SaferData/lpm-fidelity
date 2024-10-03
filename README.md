# lpm-fidelity

This repo was forked from [here](https://github.com/inferenceql/lpm.fidelity).

## Disclaimer
This is pre-alpha software. We are currently testing it in real-world scenarios. In its present state, we discourage users from trying it.

## Overview over fidelity component

A library for assessing fidelity between different Polars data frames. 
Fidelity refers to *"[measures] that directly compare a synthetic dataset with a the real one. From
a high-level perspective, fidelity is how well the synthetic data "statistically"
matches the real data"* ([Jordan et al., 2022](https://arxiv.org/pdf/2205.03257)).
![schematic](images/fidelity-schematic.png)

## Installation

This library is packaged with [Poetry](https://python-poetry.org/). Add this
line to your `pyproject.toml` file:
```toml
lpm-fidelity = {git = "https://github.com/neeshjaa/lpm_fidelity.git", branch = "main"}
```

## Usage

:warning: this currently only works with categorical data frames files.

### Using fidelity as a Python library

```python
# Get dependencies.
import polars as pl

from lpm_fidelity.distances import bivariate_distances_in_data
from lpm_fidelity.distances import univariate_distances_in_data
from lpm_fidelity.two_sample_testing import univariate_two_sample_testing_in_data

# Read in two csv files.
df_foo = pl.read_csv("foo.csv")
df_bar = pl.read_csv("bar.csv")

# Compute univariate distance.
df_univariate_distance = univariate_distances_in_data(df_foo, df_bar, distance_metric="tvd")

# Compute bivariate distance.
df_bivariate_distance = bivariate_distances_in_data(df_foo, df_bar, distance_metric="tvd")

# Compute univariate two-sample hypothesis tests (currently only Chi^2).
df_univariate_two_sample_test = univariate_two_sample_testing_in_data(df_foo, df_bar)
```
## Test

Tests can be run with Poetry

```shell
poetry run pytest tests/ -vvv
```
