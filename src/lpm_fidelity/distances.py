import itertools

import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon as js
from scipy.stats import entropy as scipy_entropy

from lpm_fidelity.counting import (
    harmonize_categorical_probabilities,
    normalize_count,
    normalize_count_bivariate,
)


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
    return 0.5 * sum([np.abs(p - q) for p, q in zip(P, Q)])


def _distance_from_maps(ps_a, ps_b, distance_metric, overlap_required=True):
    # If we don't require overlap between columns, return 0 if one map is empty
    if not overlap_required:
        if (not ps_a) or (not ps_b):
            return None
    ps_a, ps_b = harmonize_categorical_probabilities(ps_a, ps_b)
    # The previous line ensures that the keys are the same. So the following
    # is safe to do.
    P = [ps_a[k] for k in ps_a.keys()]
    Q = [ps_b[k] for k in ps_a.keys()]
    if distance_metric == "tvd":
        return tvd(P, Q)
    elif distance_metric == "kl":
        # If qk is not null, scipy_entropy computes KL.
        return scipy_entropy(P, qk=Q)
    elif distance_metric == "js":
        return js(P, Q)
    else:
        return ValueError(f"Unknown distance metric: {distance_metric}")


def univariate_distance(column_a, column_b, distance_metric="tvd"):
    """
    Compute a set of distance metric for a pair of columns

    Parameters:
    - column_a (List or Polars Series): first column used in distance.
    - column_b (List or Polars Series): second column used in distance.
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
    ps_a = normalize_count(column_a)
    ps_b = normalize_count(column_b)
    return _distance_from_maps(ps_a, ps_b, distance_metric)


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
    result = [
        {
            "column": c,
            distance_metric: univariate_distance(
                df_a[c], df_b[c], distance_metric=distance_metric
            ),
        }
        for c in df_a.columns
    ]
    return pl.DataFrame(result).sort(distance_metric, descending=False)


def bivariate_distance(
    column_a_1,
    column_a_2,
    column_b_1,
    column_b_2,
    distance_metric="tvd",
    overlap_required=True,
):
    """
    Compute a set of distance metric for a pair of columns

    Parameters:
    - column_a_1 (List or Polars Series):  A column in dataframe a
    - column_a_2 (List or Polars Series):  Another column in dataframe a
    - column_b_1 (List or Polars Series):  A column in dataframe b
    - column_b_2 (List or Polars Series):  Another column in dataframe b
    - distance_metric (str): Choose a distance metric. One of
                              "tvd", "kl", "js".
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
    ps_a = normalize_count_bivariate(
        column_a_1, column_a_2, overlap_required=overlap_required
    )
    ps_b = normalize_count_bivariate(
        column_b_1, column_b_2, overlap_required=overlap_required
    )
    return _distance_from_maps(
        ps_a, ps_b, distance_metric, overlap_required=overlap_required
    )


def bivariate_distances_in_data(
    df_a, df_b, distance_metric="tvd", overlap_required=True
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

    def _row(column_1, column_2):
        d = bivariate_distance(
            df_a[column_1],
            df_a[column_2],
            df_b[column_1],
            df_b[column_2],
            overlap_required=overlap_required,
        )
        return {"column-1": column_1, "column-2": column_2, distance_metric: d}

    result = [
        _row(column_1, column_2)
        for column_1, column_2 in itertools.combinations(df_a.columns, 2)
    ]
    return pl.DataFrame(result).sort(distance_metric, descending=False)
