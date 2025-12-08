import polars as pl
import pytest
from jax import numpy as jnp

from lpm_fidelity.counting import OrdinalDF
from lpm_fidelity.distances import (
    DistanceMetric,
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
    assert tvd([0.5, 0.5], [0.9, 0.1]) == pytest.approx(0.4)


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_univariate_distance_0(distance_metric):
    df = pl.DataFrame({"col": ["a", "b", "a"]})
    odf = OrdinalDF.from_dataframe(df)
    metric_enum = {
        "tvd": DistanceMetric.TVD,
        "kl": DistanceMetric.KL,
        "js": DistanceMetric.JS,
    }[distance_metric]
    n_unique = len(odf.encoders[0].categories_[0])
    assert univariate_distance(
        odf.data[:, 0], odf.data[:, 0], n_unique, metric_enum
    ) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_univariate_distance_transitivity(distance_metric):
    df_a = pl.DataFrame({"col": ["a", "a", "a", "a"]})
    df_b = pl.DataFrame({"col": ["a", "a", "b", "b"]})
    df_c = pl.DataFrame({"col": ["a", "a", "a", "b"]})
    odf_a, odf_b, odf_c = OrdinalDF.from_dataframes([df_a, df_b, df_c])
    metric_enum = {
        "tvd": DistanceMetric.TVD,
        "kl": DistanceMetric.KL,
        "js": DistanceMetric.JS,
    }[distance_metric]
    n_unique = len(odf_a.encoders[0].categories_[0])
    assert univariate_distance(
        odf_a.data[:, 0], odf_b.data[:, 0], n_unique, metric_enum
    ) > univariate_distance(odf_a.data[:, 0], odf_c.data[:, 0], n_unique, metric_enum)


def test_univariate_distance_spot():
    df_a = pl.DataFrame({"col": ["a"] * 5 + ["b"] * 5})
    df_b = pl.DataFrame({"col": ["a"] * 9 + ["b"] * 1})
    odf_a, odf_b = OrdinalDF.from_dataframes([df_a, df_b])
    n_unique = len(odf_a.encoders[0].categories_[0])
    assert univariate_distance(
        odf_a.data[:, 0], odf_b.data[:, 0], n_unique, DistanceMetric.TVD
    ) == pytest.approx(0.4)


def test_univariate_distance_spot_different_length():
    df_a = pl.DataFrame({"col": ["a", "b"]})
    df_b = pl.DataFrame({"col": ["a"] * 9 + ["b"] * 1})
    odf_a, odf_b = OrdinalDF.from_dataframes([df_a, df_b])
    n_unique = len(odf_a.encoders[0].categories_[0])
    assert univariate_distance(
        odf_a.data[:, 0], odf_b.data[:, 0], n_unique, DistanceMetric.TVD
    ) == pytest.approx(0.4)


def test_univariate_distance_one_empty():
    # Empty columns now raise ValueError at OrdinalDF encoding time
    df = pl.DataFrame({"col": []})
    with pytest.raises(ValueError):
        OrdinalDF.from_dataframe(df)


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
    assert df_result["tvd"][0] == pytest.approx(0.4)
    assert df_result["tvd"][1] == pytest.approx(1.0)


@pytest.mark.parametrize("distance_metric", ["tvd", "kl", "js"])
def test_bivariate_distance_smoke(distance_metric):
    col = jnp.array([[0, 0]])
    metric_enum = {
        "tvd": DistanceMetric.TVD,
        "kl": DistanceMetric.KL,
        "js": DistanceMetric.JS,
    }[distance_metric]
    assert bivariate_distance(col, col, 1, 1, metric_enum) == pytest.approx(0.0)


def test_bivariate_distance_spot():
    assert bivariate_distance(
        jnp.column_stack([jnp.array([0, 0, 1, 1]), jnp.array([0, 0, 1, 1])]),
        jnp.column_stack([jnp.array([0, 0, 0, 1]), jnp.array([0, 0, 1, 1])]),
        2,
        2,
        DistanceMetric.TVD,
    ) == pytest.approx(0.25)


def test_bivariate_distance_no_overlap_exception():
    # no overlap because -1 sentinel value in either x or y for every row in A
    result = bivariate_distance(
        jnp.column_stack([jnp.array([-1, -1, 1, 1]), jnp.array([0, 0, -1, -1])]),
        jnp.column_stack([jnp.array([0, 0, 0, 1]), jnp.array([0, 0, 1, 1])]),
        2,
        2,
        DistanceMetric.TVD,
    )
    assert jnp.isnan(result)


def test_bivariate_distance_no_overlap_no_exception():
    # Test that empty arrays return NaN
    empty_data = jnp.array([]).reshape(0, 2).astype(jnp.int32)

    result = bivariate_distance(
        empty_data,
        empty_data,
        2,
        2,
        DistanceMetric.TVD,
    )
    assert jnp.isnan(result)


def test_bivariate_distance_no_overlap_spot():
    df_a = pl.DataFrame({"foo": ["a", "a", "b", "b"], "bar": ["x", "x", "y", "y"]})
    df_b = pl.DataFrame({"foo": ["a", "a", "a", "b"], "bar": ["x", "x", "x", "y"]})

    odf_a, odf_b = OrdinalDF.from_dataframes([df_a, df_b])

    assert bivariate_distance(
        odf_a.data,
        odf_b.data,
        len(odf_a.encoders[0].categories_[0]),
        len(odf_a.encoders[1].categories_[0]),
        DistanceMetric.TVD,
    ) == pytest.approx(0.25)


def test_bivariate_distances_in_data_with_nulls():
    # Test that bivariate_distances_in_data properly handles nulls as sentinels
    # Pairs containing nulls are filtered out during counting
    df_a = pl.DataFrame(
        {"foo": ["a", "a", "b", None, "b"], "bar": ["x", "x", "y", "y", None]}
    )
    df_b = pl.DataFrame(
        {"foo": ["a", None, "a", "a", "b"], "bar": ["x", "x", None, "x", "y"]}
    )

    result = bivariate_distances_in_data(df_a, df_b, distance_metric="tvd")

    # Should compute distance successfully, filtering out pairs with nulls
    assert len(result) == 1  # Only one pair: (foo, bar)
    assert result["column-1"][0] == "foo"
    assert result["column-2"][0] == "bar"
    assert isinstance(result["tvd"][0], float)
    assert result["tvd"][0] >= 0.0


def test_bivariate_distances_in_data_no_valid_pairs_exception():
    # Test the old "no overlap" case: columns have no valid (non-null, non-null) pairs
    # This replicates the old test case with ["a", "a", None, None] and [None, None, "y", "y"]
    df_a = pl.DataFrame({"foo": ["a", "a", None, None], "bar": [None, None, "y", "y"]})
    df_b = pl.DataFrame({"foo": ["a", "a", "a", "b"], "bar": ["x", "x", "x", "y"]})

    # Should raise because no valid pairs exist between foo and bar in df_a
    with pytest.raises(ValueError, match="no overlap"):
        bivariate_distances_in_data(
            df_a, df_b, distance_metric="tvd", overlap_required=True
        )


def test_bivariate_distances_in_data_no_valid_pairs_no_exception():
    # Test that no valid pairs returns None when overlap_required=False
    df_a = pl.DataFrame({"foo": ["a", "a", None, None], "bar": [None, None, "y", "y"]})
    df_b = pl.DataFrame({"foo": ["a", "a", "a", "b"], "bar": ["x", "x", "x", "y"]})

    result = bivariate_distances_in_data(
        df_a, df_b, distance_metric="tvd", overlap_required=False
    )

    # Should return a dataframe with None for the distance
    assert len(result) == 1
    assert result["tvd"][0] is None


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
