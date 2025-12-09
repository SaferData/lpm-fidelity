import itertools
from enum import IntEnum
from functools import partial

import jax.numpy as jnp
import polars as pl
from jax import Array, jit, vmap
from jax.lax import cond, switch
from jax.scipy.special import rel_entr
from jaxtyping import Integer

from lpm_fidelity.counting import (
    OrdinalDF,
    normalize_count_bivariate_memoized,
    normalize_count_memoized,
)


class DistanceMetric(IntEnum):
    """JAX-compatible integer enum for distance metrics."""

    TVD = 0
    KL = 1
    JS = 2


@jit
def fasttvd(P, Q):
    return 0.5 * jnp.sum(jnp.abs(P - Q))


@jit
def fastkl(P, Q):
    return jnp.sum(rel_entr(P, Q))


@jit
def fastjs(P, Q):
    # Jensen-Shannon divergence: sqrt(0.5 * (KL(P||M) + KL(Q||M))) where M = 0.5 * (P + Q)
    M = 0.5 * (P + Q)
    return jnp.sqrt(0.5 * (fastkl(P, M) + fastkl(Q, M)))


def tvd(P, Q):
    """
    Compute total variation distance between two probability vectors.

    Parameters:
    - P:  list of probabilities
    - Q:  list of probabilities

    Returns:
        Total variation distance.

    Examples:
    >>> tvd([0.5, 0.5], [0.9, 0.1])
        0.4
    """
    assert len(P) > 0
    assert len(P) == len(Q)
    return float(0.5 * sum([jnp.abs(p - q) for p, q in zip(P, Q)]))


def _fast_distance(ps_a, ps_b, distance_metric: DistanceMetric):
    return switch(
        distance_metric,
        [  # Order is determined by DistanceMetric enum values
            fasttvd,
            fastkl,
            fastjs,
        ],
        ps_a,
        ps_b,
    )


@partial(jit, static_argnums=(2, 3))
def univariate_distance(
    column_a: Integer[Array, "n"],
    column_b: Integer[Array, "m"],
    unique_vals: int,
    distance_metric: DistanceMetric = DistanceMetric.TVD,
):
    """
    Compute a set of distance metric for a pair of columns

    Parameters:
    - column_a (Integer[Array, "n"]): first column used in distance.
    - column_b (Integer[Array, "m"]): second column used in distance.
    - unique_vals (int): The number of unique values in the column.
    - distance_metric (str): Choose a distance metric. One of
                              "tvd", "kl", "js".

    Returns:
        A dict with distance metric and the columns names

    Examples:
    >>> univariate_distance(
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("foo", ["a", "b", "b", "b"]),
            distance_metric="tvd"
            )
        0.5
    >>> univariate_distance(
            ["a", "b", "a", "a"],
            ["a", "b", "b", "b"],
            distance_metric="tvd"
            )
        0.5
    """
    cs_a, a = normalize_count_memoized(column_a, unique_vals)
    cs_b, b = normalize_count_memoized(column_b, unique_vals)
    ps_a = cs_a / a.astype(jnp.float32)
    ps_b = cs_b / b.astype(jnp.float32)
    return _fast_distance(ps_a, ps_b, distance_metric)


def univariate_distances_in_data(df_a, df_b, distance_metric="tvd"):
    """
    Take two dataframes and compare a distance metric
    for all categorical_columns.

    Parameters:
    - df_a:  Polars Dataframe
    - df_b:  Polars Dataframe
    - distance_metric (str): Choose a distance metric. One of
                              "tvd", "kl", "js".

    Returns:
        A Polars Dataframe with a column "column" recording columns names
        and the distance metric used.

    Examples:
    >>> univariate_distances_in_data(df_a, df_b)
        ┌────────┬─────┐
        │ column ┆ tvd │
        │ ---    ┆ --- │
        │ str    ┆ f64 │
        ╞════════╪═════╡
        │ foo    ┆ 0.1 │
        │ bar    ┆ 0.2 │
        │ ...    ┆ ... │
        │ baz    ┆ 0.3 │
        └────────┴─────┘
       (Above is using examples values for the distance metric tvd)
    """
    assert set(df_a.columns) == set(df_b.columns)

    # Encode the dataframes' categoricals into integers,
    # using -1 as sentinel value for nulls
    odf_a, odf_b = OrdinalDF.from_dataframes([df_a, df_b])

    dm = DistanceMetric[distance_metric.upper()]

    result = [
        {
            "column": odf_a.columns[c],
            distance_metric: univariate_distance(
                odf_a.data[:, c],
                odf_b.data[:, c],
                len(odf_a.encoders[c].categories_[0]),
                dm,
            ),
        }
        for c in range(len(odf_a.columns))
    ]
    return pl.DataFrame(result).sort(distance_metric, descending=False)


@partial(jit, static_argnums=(2, 3, 4))
def bivariate_distance(
    columns_a: Integer[Array, "n 2"],
    columns_b: Integer[Array, "n 2"],
    c1_uniq_vals: int,
    c2_uniq_vals: int,
    distance_metric: DistanceMetric = DistanceMetric.TVD,
):
    """
    Compute a set of distance metric for a pair of columns

    Parameters:i
    - column_a_1 (List or Polars Series):  A column in dataframe a
    - column_a_2 (List or Polars Series):  Another column in dataframe a
    - column_b_1 (List or Polars Series):  A column in dataframe b
    - column_b_2 (List or Polars Series):  Another column in dataframe b
    - distance_metric (DistanceMetric): Choose a distance metric. One of
                              DistanceMetric.TVD, DistanceMetric.KL, DistanceMetric.JS.
    - overlap_required bool:  If  two columns don't have non-null overlap,
                              throw error

    Returns:
        A dict with a distance metric and both columns names

    Examples:
    >>> bivariate_distance(
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("bar", ["x", "y", "y", "y"]),
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("bar", ["x", "y", "y", "y"]),
            distance_metric="tvd"
            )
        0.0

    >>> bivariate_distance(
            pl.Series("foo", ["a", "b", "a", "a"]),
            pl.Series("bar", ["x", "y", "x", "x"]),
            pl.Series("foo", ["a", "a", "a", "b"]),
            pl.Series("bar", ["x", "x", "x", "y"]),
            distance_metric="tvd"
            )
        0.5
    """
    cs_a, a = normalize_count_bivariate_memoized(columns_a, c1_uniq_vals, c2_uniq_vals)
    cs_b, b = normalize_count_bivariate_memoized(columns_b, c1_uniq_vals, c2_uniq_vals)

    ps_a = jnp.ravel(cs_a / a.astype(jnp.float32))
    ps_b = jnp.ravel(cs_b / b.astype(jnp.float32))

    return cond(
        (a > 0) * (b > 0),
        _fast_distance,
        lambda _ps_a, _ps_b, _dm: jnp.nan,  # Return NaN if no overlap -- becomes None
        ps_a,
        ps_b,
        distance_metric,
    )


def bivariate_distances_in_data(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    distance_metric: str = "tvd",
    overlap_required: bool = True,
):
    """
    Take two dataframes, create all pairs categorical columns.  For each pair,
    compute a probability vector of all possible events for this pair.
    Compare a distance metric for the probabilites of these events between
    the two dataframes.

    Parameters:
    - df_a:  Polars Dataframe
    - df_b:  Polars Dataframe
    - distance_metric (str): Choose a distance metric. One of
                              "tvd", "kl", "js".
    - overlap_required bool:  If  two columns don't have non-null overlap,
                              throw error

    Returns:
        A Polars Dataframe with two columns ("column-1", "column-2")
        recording columns names and the distance metric used.

    Examples:
    >>> bivariate_distances_in_data(df_a, df_b)
        ┌──────────┬──────────┬─────┐
        │ column-1 ┆ column-2 ┆ tvd │
        │ ---      ┆ ---      ┆ --- │
        │ str      ┆ str      ┆ f64 │
        ╞══════════╪══════════╪═════╡
        │ foo      ┆ bar      ┆ 1.0 │
        │ foo      ┆ baz      ┆ 2.0 │
        │ ...      ┆ ...      ┆ ... │
        │ bar      ┆ baz      ┆ 3.0 │
        └──────────┴──────────┴─────┘
       (Above is using examples values for the distance metric)
    """
    assert set(df_a.columns) == set(df_b.columns)

    # Encode the dataframes' categoricals into integers,
    # using -1 as sentinel value for nulls
    odf_a, odf_b = OrdinalDF.from_dataframes([df_a, df_b])

    pairs = list(itertools.combinations(range(len(odf_a.columns)), 2))

    # Optimizations for memoization when many columns are similarly sized (unique vals):
    # 1. compute cardinalities for all columns
    cardinalities = [len(e.categories_[0]) for e in odf_a.encoders]

    # 2. separate out pairs by their cardinalities, to be vmapped
    cardwise_map = {}
    for i, j in pairs:
        key = (cardinalities[i], cardinalities[j])
        if key not in cardwise_map:
            cardwise_map[key] = []
        cardwise_map[key].append((i, j))

    def _helper(pair: Integer[Array, "2"], da, db, c1_uniq_vals, c2_uniq_vals) -> float:
        return bivariate_distance(
            da[:, pair],
            db[:, pair],
            c1_uniq_vals,
            c2_uniq_vals,
            DistanceMetric[distance_metric.upper()],
        )

    vh = jit(vmap(_helper, in_axes=(0, None, None, None, None)), static_argnums=(3, 4))

    # helper to encode post-vmap results
    def _row(index_1: int, index_2: int, d: float):
        if jnp.isnan(d):
            if overlap_required:
                raise ValueError("no overlap")
            d = None

        return {
            "column-1": odf_a.columns[index_1],
            "column-2": odf_a.columns[index_2],
            distance_metric: d,
        }

    result = []
    for (i, j), v in cardwise_map.items():
        ds = vh(jnp.array(v, dtype=jnp.int32), odf_a.data, odf_b.data, i, j)
        result.extend(
            _row(index_1, index_2, d) for (index_1, index_2), d in zip(v, ds.tolist())
        )

    return pl.DataFrame(result).sort(distance_metric, descending=False)
