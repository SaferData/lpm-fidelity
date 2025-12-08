import sys

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from lpm_fidelity.counting import (
    OrdinalDF,
    _is_none_or_nan,
    _probabilities_safe_as_denominator,
    bivariate_empirical_frequencies,
    contingency_table,
    normalize_count,
    normalize_count_bivariate_memoized,
)


@pytest.mark.parametrize("value", [None, np.nan])
def test_is_none_or_nan_with_null(value):
    assert _is_none_or_nan(value)


@pytest.mark.parametrize("value", ["x", 0, 1, 42.0, "abc"])
def test_is_none_or_nan_with_value(value):
    assert not _is_none_or_nan(value)


list_items_single = ["a"]


@pytest.mark.parametrize(
    "column",
    [list_items_single, np.array(list_items_single), pl.Series(list_items_single)],
)
def test_normalize_count_single_str(column):
    assert normalize_count(column) == {list_items_single[0]: 1.0}


@pytest.mark.parametrize(
    "column",
    [
        [],
        np.array([]),
        pl.Series([]),
    ],
)
def test_normalize_count_empty(column):
    # Empty columns raise ValueError from OrdinalEncoder
    with pytest.raises(ValueError):
        normalize_count(column)


def test_normalize_count_all_null_numpy_none():
    # np.array([None]) raises IndexError due to empty categories in sklearn
    with pytest.raises(IndexError):
        normalize_count(np.array([None]))


list_items_numbers = [1]


@pytest.mark.parametrize(
    "column",
    [list_items_numbers, np.array(list_items_numbers), pl.Series(list_items_numbers)],
)
def test_normalize_count_single_number(column):
    assert normalize_count(column) == {list_items_numbers[0]: 1.0}


list_two_items = ["a", "b"]


@pytest.mark.parametrize(
    "column",
    [
        list_two_items,
        np.array(list_two_items),
        pl.Series(list_two_items),
        list_two_items + list_two_items,
        np.array(list_two_items + list_two_items),
        pl.Series(list_two_items + list_two_items),
    ],
)
def test_normalize_count_balanced_categories(column):
    assert normalize_count(column) == {k: 0.5 for k in list_two_items}


list_items_inbalanced = ["a", "b", "b", "b"]


@pytest.mark.parametrize(
    "column",
    [
        list_items_inbalanced,
        np.array(list_items_inbalanced),
        pl.Series(list_items_inbalanced),
    ],
)
def test_normalize_count_inbalanced_categories(column):
    assert normalize_count(column) == {"a": 0.25, "b": 0.75}


list_items_nan = ["a", None, "b", "b", "b", None]


@pytest.mark.parametrize(
    "column", [list_items_nan, np.array(list_items_nan), pl.Series(list_items_nan)]
)
def test_normalize_count_with_nan(column):
    assert normalize_count(column) == {"a": 0.25, "b": 0.75}


def test_normalize_count_bivariate_memoized_single_entry():
    df = pl.DataFrame({"c1": ["a"], "c2": ["b"]})
    odf = OrdinalDF.from_dataframe(df)
    cols = odf.data[:, :2]
    memo, total = normalize_count_bivariate_memoized(cols, 1, 1)
    assert memo.shape == (1, 1)
    assert memo[0, 0] == 1
    assert total == 1.0


def test_normalize_count_bivariate_memoized_without_nan():
    # ["a", "b", "b", "b"] x ["x", "y", "y", "y"]
    # Expect: (a,x)=1, (b,y)=3, total=4
    # After encoding: a→0, b→1, x→0, y→1
    # memo[0,0]=1, memo[1,1]=3
    df = pl.DataFrame({"c1": ["a", "b", "b", "b"], "c2": ["x", "y", "y", "y"]})
    odf = OrdinalDF.from_dataframe(df)
    cols = odf.data[:, :2]
    n_c1 = len(odf.encoders[0].categories_[0])
    n_c2 = len(odf.encoders[1].categories_[0])
    memo, total = normalize_count_bivariate_memoized(cols, n_c1, n_c2)

    assert memo.shape == (2, 2)
    assert total == 4.0
    # Normalized: memo / total gives probabilities
    probs = memo / total
    assert float(probs[0, 0]) == pytest.approx(0.25)
    assert float(probs[1, 1]) == pytest.approx(0.75)
    assert float(probs[0, 1]) == pytest.approx(0.0)
    assert float(probs[1, 0]) == pytest.approx(0.0)


def test_normalize_count_bivariate_memoized_empty_raises():
    df = pl.DataFrame({"c1": [], "c2": []})
    with pytest.raises(ValueError):
        OrdinalDF.from_dataframe(df)


def test_normalize_count_bivariate_memoized_with_nan():
    # ["a", None, "b", "b", "b", None] x ["x", None, "y", "y", "y", "y"]
    # After filtering pairs with -1 (null sentinel): (a,x), (b,y), (b,y), (b,y)
    # Expect: (a,x)=1, (b,y)=3, total=4
    df = pl.DataFrame(
        {"c1": ["a", None, "b", "b", "b", None], "c2": ["x", None, "y", "y", "y", "y"]}
    )
    odf = OrdinalDF.from_dataframe(df)
    cols = odf.data[:, :2]
    n_c1 = len(odf.encoders[0].categories_[0])
    n_c2 = len(odf.encoders[1].categories_[0])
    memo, total = normalize_count_bivariate_memoized(cols, n_c1, n_c2)

    assert memo.shape == (2, 2)
    # Pairs with -1 sentinel are skipped, so total = 4 (not 6)
    assert total == 4.0
    probs = memo / total
    assert float(probs[0, 0]) == pytest.approx(0.25)
    assert float(probs[1, 1]) == pytest.approx(0.75)
    assert float(probs[0, 1]) == pytest.approx(0.0)
    assert float(probs[1, 0]) == pytest.approx(0.0)


def test_probabilities_safe_as_denominator():
    assert _probabilities_safe_as_denominator(
        {"a": 0, "b": 1.0}, constant=sys.float_info.min
    ) == {"a": sys.float_info.min, "b": 1.0}


def test_probabilities_safe_as_denominator_idenity():
    assert _probabilities_safe_as_denominator({"a": 0.3, "b": 0.7}) == {
        "a": 0.3,
        "b": 0.7,
    }


def test_contingency_table_2x1():
    assert (
        contingency_table(["a", "b"], ["a", "a"]).tolist()
        == np.asarray([[1, 2], [1, 0]]).tolist()
    )


def test_contingency_table_2x2():
    assert (
        contingency_table(["a", "b"], ["x", "y"]).tolist()
        == np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]).tolist()
    )


def test_contingency_table_2x2_with_None():
    assert (
        contingency_table(["a", "b", None], ["x", "y"]).tolist()
        == np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]).tolist()
    )


def test_contingency_table_2x2_with_nan():
    assert (
        contingency_table(["a", "b", np.nan], ["x", "y"]).tolist()
        == np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]).tolist()
    )


def test_count_in_dfs_single():
    dfs = {
        "quax": pl.DataFrame(
            {
                "foo": ["A", "A", "B", "B", "B", "B", "B", "B"],
                "bar": ["X", "Y", "X", "X", "Y", "Y", "Y", "Y"],
            }
        )
    }
    foo_vals = ["A", "A", "B", "B"]
    bar_vals = ["X", "Y", "X", "Y"]
    expected = pl.DataFrame(
        {
            "foo": foo_vals,
            "bar": bar_vals,
            "Normalized frequency": [0.125, 0.125, 0.25, 0.5],
            "Source": ["quax"] * 4,
        }
    )
    assert_frame_equal(expected, bivariate_empirical_frequencies(dfs, "foo", "bar"))


def test_count_in_dfs():
    dfs = {
        "quax": pl.DataFrame(
            {
                "foo": ["A", "A", "B", "B", "B", "B", "B", "B"],
                "bar": ["X", "Y", "X", "X", "Y", "Y", "Y", "Y"],
            }
        ),
        "quagga": pl.DataFrame(
            {
                "foo": ["A", "A"],
                "bar": ["X", "Y"],
            }
        ),
        "baz": pl.DataFrame(
            {
                "foo": ["A", "A", "B", "B", "B", "B", "B", "B"],
                "bar": ["X", "Y", "X", "X", "Y", "Y", "Y", "Y"],
            }
        ),
    }

    foo_vals = ["A", "A", "B", "B"]
    bar_vals = ["X", "Y", "X", "Y"]
    expected = pl.DataFrame(
        {
            "foo": foo_vals + foo_vals + foo_vals,
            "bar": bar_vals + bar_vals + bar_vals,
            "Normalized frequency": [
                0.125,
                0.125,
                0.25,
                0.5,
                0.5,
                0.5,
                0.0,
                0.0,
                0.125,
                0.125,
                0.25,
                0.5,
            ],
            "Source": ["quax"] * 4 + ["quagga"] * 4 + ["baz"] * 4,
        }
    ).sort(["foo", "bar"])
    assert_frame_equal(expected, bivariate_empirical_frequencies(dfs, "foo", "bar"))
