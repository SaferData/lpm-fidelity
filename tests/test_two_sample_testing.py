import numpy as np
import polars as pl

from lpm_fidelity.two_sample_testing import chi_squared
from lpm_fidelity.two_sample_testing import univariate_two_sample_testing_in_data


def test_chi_squared_smoke():
    column = ["a"] * 10
    p, chi2, problem = chi_squared(column, column)
    assert isinstance(p, float)
    assert isinstance(chi2, float)
    assert problem in [True, False]


def test_chi_squared_insignificance():
    column_a = ["a"] * 100 + ["b"] * 100
    column_b = ["b"] * 100 + ["a"] * 100
    p, chi2, problem = chi_squared(column_a, column_b)
    assert p > 0.05
    assert not problem


def test_chi_squared_significance():
    column_a = ["a"] * 100 + ["b"] * 100
    column_b = ["b"] * 90 + ["a"] * 10
    p, chi2, problem = chi_squared(column_a, column_b)
    assert p < 0.05
    assert not problem


def test_chi_squared_problem():
    column_a = ["a"] * 20 + ["b"] * 20
    column_b = ["a", "b"]
    p, chi2, problem = chi_squared(column_a, column_b)
    assert p > 0.05
    assert problem


df_a = pl.DataFrame(
    {"column-1": ["a"] * 50 + ["b"] * 50, "column-2": ["x"] * 90 + ["y"] * 10}
)
df_b = pl.DataFrame(
    {"column-1": ["b"] * 50 + ["a"] * 50, "column-2": ["y"] * 50 + ["x"] * 50}
)


def univariate_two_sample_testing_in_data_smoke():
    df_result = univariate_distances_in_data(df_a, df_b, distance_metric="tvd")
    assert len(df_result) == 2
    assert len(df_result.columns) == 4


def univariate_two_sample_testing_in_data():
    df_result = univariate_distances_in_data(df_a, df_b, distance_metric="tvd")
    assert df_result["p-value"][0] < 0.05
    assert df_result["p-value"][1] > 0.05
