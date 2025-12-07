import polars as pl
import pytest

from lpm_fidelity.distances import (
    bivariate_distance,
    bivariate_distances_in_data,
    tvd,
    univariate_distance,
    univariate_distances_in_data,
)


@pytest.mark.parametrize(
    "P, Q",
    [
        (
            [1.0],
            [1.0],
        ),
        (
            [0.5, 0.5],
            [0.5, 0.5],
        ),
    ],
)
def test_tvd_0(P, Q):
    assert tvd(P, Q) == 0


@pytest.mark.parametrize(
    "P, Q",
    [
        (
            [1.0, 0.0],
            [0.0, 1.0],
        ),
        (
            [0.0, 1.0],
            [1.0, 0.0],
        ),
    ],
)
def test_tvd_1(P, Q):
    assert tvd(P, Q) == 1


def test_tvd_spot_check():
    assert tvd([0.5, 0.5], [0.9, 0.1]) == 0.4


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_univariate_distance_0(distance_metric):
    col = ["a", "b", "a"]
    assert univariate_distance(col, col, distance_metric=distance_metric) == 0.0


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_univariate_distance_transitivity(distance_metric):
    column_a = ["a", "a", "a", "a"]
    column_b = ["a", "a", "b", "b"]
    column_c = ["a", "a", "a", "b"]
    assert univariate_distance(
        column_a, column_b, distance_metric=distance_metric
    ) > univariate_distance(column_a, column_c, distance_metric=distance_metric)


def test_univariate_distance_spot():
    column_a = ["a"] * 5 + ["b"] * 5
    column_b = ["a"] * 9 + ["b"] * 1
    assert univariate_distance(column_a, column_b, distance_metric="tvd") == 0.4


def test_univariate_distance_spot_different_length():
    column_a = ["a", "b"]
    column_b = ["a"] * 9 + ["b"] * 1
    assert univariate_distance(column_a, column_b, distance_metric="tvd") == 0.4


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_univariate_distance_one_empty(distance_metric):
    column = ["a", "b"]
    with pytest.raises(AssertionError) as exc_info:
        univariate_distance(column, [], distance_metric=distance_metric)(column)
    assert exc_info.type == AssertionError

    with pytest.raises(AssertionError) as exc_info:
        univariate_distance([], column, distance_metric=distance_metric)(column)
    assert exc_info.type == AssertionError


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_univariate_distances_in_data_smoke(distance_metric):
    df = pl.DataFrame(
        {
            "column-1": ["foo", "bar"],
            "column-2": [42, 17],
        }
    )
    df_result = univariate_distances_in_data(df, df, distance_metric=distance_metric)
    assert len(df_result) == 2
    assert len(df_result.columns) == 2
    assert not df_result["column"].dtype.is_numeric()
    assert df_result[distance_metric].dtype.is_numeric()


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_univariate_distances_in_data_all_0(distance_metric):
    df = pl.DataFrame(
        {
            "column-1": ["foo", "bar"],
            "column-2": ["bar", "quagga"],
            "column-3": ["baz", "foo"],
        }
    )
    df_result = univariate_distances_in_data(df, df, distance_metric=distance_metric)
    assert set(df_result[distance_metric]) == {0}


def test_univariate_distances_in_data():
    df_a = pl.DataFrame({"column-1": ["a"] * 5 + ["b"] * 5, "column-2": ["x"] * 10})
    df_b = pl.DataFrame({"column-1": ["a"] * 9 + ["b"] * 1, "column-2": ["y"] * 10})
    df_result = univariate_distances_in_data(df_a, df_b, distance_metric="tvd")
    assert df_result["tvd"][0] == 0.4
    assert df_result["tvd"][1] == 1.0


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_bivariate_distance_smoke(distance_metric):
    col = ["a"]
    assert isinstance(bivariate_distance(col, col, col, col, distance_metric), float)


def test_bivariate_distance_spot():
    assert (
        bivariate_distance(
            pl.Series("foo", ["a", "a", "b", "b"]),
            pl.Series("bar", ["x", "x", "y", "y"]),
            pl.Series("foo", ["a", "a", "a", "b"]),
            pl.Series("bar", ["x", "x", "x", "y"]),
            distance_metric="tvd",
        )
        == 0.25
    )


def test_bivariate_distance_no_overlap_exception():
    with pytest.raises(AssertionError) as exc_info:
        bivariate_distance(
            pl.Series("foo", ["a", "a", None, None]),
            pl.Series("bar", [None, None, "y", "y"]),
            pl.Series("foo", ["a", "a", "a", "b"]),
            pl.Series("bar", ["x", "x", "x", "y"]),
            distance_metric="tvd",
        )
    assert exc_info.type == AssertionError


def test_bivariate_distance_no_overlap_no_exception():
    assert (
        bivariate_distance(
            pl.Series("foo", ["a", "a", None, None]),
            pl.Series("bar", [None, None, "y", "y"]),
            pl.Series("foo", ["a", "a", "a", "b"]),
            pl.Series("bar", ["x", "x", "x", "y"]),
            distance_metric="tvd",
            overlap_required=False,
        )
        == None
    )


def test_bivariate_distance_no_overlap_spot():
    assert (
        bivariate_distance(
            pl.Series("foo", ["a", "a", "b", "b"]),
            pl.Series("bar", ["x", "x", "y", "y"]),
            pl.Series("foo", ["a", "a", "a", "b"]),
            pl.Series("bar", ["x", "x", "x", "y"]),
            distance_metric="tvd",
            overlap_required=False,
        )
        == 0.25
    )


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_bivariate_distances_in_data_smoke(distance_metric):
    df = pl.DataFrame(
        {
            "column-1": ["foo", "bar"],
            "column-2": ["bar", "quagga"],
            "column-3": ["baz", "foo"],
        }
    )
    df_result = bivariate_distances_in_data(df, df, distance_metric=distance_metric)
    assert len(df_result) == 3
    assert len(df_result.columns) == 3
    assert not df_result["column-1"].dtype.is_numeric()
    assert not df_result["column-2"].dtype.is_numeric()
    assert df_result[distance_metric].dtype.is_numeric()


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_bivariate_distances_in_data_all_0(distance_metric):
    df = pl.DataFrame(
        {
            "column-1": ["foo", "bar"],
            "column-2": ["bar", "quagga"],
            "column-3": ["baz", "foo"],
        }
    )
    df_result = bivariate_distances_in_data(df, df, distance_metric=distance_metric)
    assert set(df_result[distance_metric]) == {0}


def test_bivariate_distances_in_data_spot():
    df_a = pl.DataFrame(
        {
            "column-1": ["a"] * 5 + ["b"] * 5,
            "column-2": ["x"] * 10,
            "column-3": range(10),
        }
    )
    df_b = pl.DataFrame(
        {
            "column-1": ["a"] * 9 + ["b"] * 1,
            "column-2": ["x"] * 10,
            "column-3": range(10),
        }
    )
    df_result = bivariate_distances_in_data(df_a, df_b, distance_metric="tvd")
    assert df_result["tvd"][0] == pytest.approx(0.0)
    assert df_result["tvd"][1] == pytest.approx(0.4)
    assert df_result["tvd"][2] == pytest.approx(0.4)
