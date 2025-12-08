import polars as pl
import pytest

from lpm_fidelity.distances import (
    bivariate_distances_in_data,
    univariate_distances_in_data,
)


def _generate_test_data(
    n_cols: int, n_rows: int, n_categories: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate two dataframes with categorical data for performance testing."""
    categories = [f"cat_{i}" for i in range(n_categories)]
    data_a = {
        f"col_{i}": [categories[j % len(categories)] for j in range(n_rows)]
        for i in range(n_cols)
    }
    data_b = {
        f"col_{i}": [categories[(j + i) % len(categories)] for j in range(n_rows)]
        for i in range(n_cols)
    }
    return pl.DataFrame(data_a), pl.DataFrame(data_b)


@pytest.mark.parametrize(
    "n_cols,n_rows,n_categories",
    [
        (4, 5000, 2),
        (8, 20000, 16),
        (12, 50000, 4),
        (12, 50000, 24),
        (12, 500000, 120),
        (12, 500000, 240),
    ],
)
def test_univariate_distances_in_data_performance(
    benchmark, n_cols: int, n_rows: int, n_categories: int
):
    df_a, df_b = _generate_test_data(n_cols, n_rows, n_categories)

    result = benchmark(univariate_distances_in_data, df_a, df_b, distance_metric="tvd")
    assert len(result) == n_cols


@pytest.mark.parametrize(
    "n_cols,n_rows,n_categories",
    [
        (4, 5000, 2),
        (8, 20000, 16),
        (12, 50000, 4),
        (12, 50000, 24),
        (12, 500000, 120),
        (12, 500000, 240),
    ],
)
def test_bivariate_distances_in_data_performance(
    benchmark, n_cols: int, n_rows: int, n_categories: int
):
    df_a, df_b = _generate_test_data(n_cols, n_rows, n_categories)

    result = benchmark(bivariate_distances_in_data, df_a, df_b, distance_metric="tvd")

    n_pairs = n_cols * (n_cols - 1) // 2
    assert len(result) == n_pairs
