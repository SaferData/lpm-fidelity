import jax.numpy as jnp
import polars as pl
import pytest

from lpm_fidelity.counting import OrdinalDF


def test_ordinal_df_single_column_two_values():
    df = pl.DataFrame({"col1": ["a", "b", "a", "b"]})
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1",)
    assert len(odf.encoders) == 1
    assert odf.data.shape == (4, 1)
    assert set(odf.data[:, 0].tolist()) == {0, 1}


def test_ordinal_df_single_column_three_values():
    df = pl.DataFrame({"col1": ["x", "y", "z", "x", "y", "z"]})
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1",)
    assert len(odf.encoders) == 1
    assert odf.data.shape == (6, 1)
    assert set(odf.data[:, 0].tolist()) == {0, 1, 2}


def test_ordinal_df_single_column_four_values():
    df = pl.DataFrame({"col1": ["p", "q", "r", "s", "p", "q", "r", "s"]})
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1",)
    assert len(odf.encoders) == 1
    assert odf.data.shape == (8, 1)
    assert set(odf.data[:, 0].tolist()) == {0, 1, 2, 3}


def test_ordinal_df_two_columns_two_values():
    df = pl.DataFrame({"col1": ["a", "b", "a", "b"], "col2": ["x", "y", "x", "y"]})
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1", "col2")
    assert len(odf.encoders) == 2
    assert odf.data.shape == (4, 2)
    assert set(odf.data[:, 0].tolist()) == {0, 1}
    assert set(odf.data[:, 1].tolist()) == {0, 1}


def test_ordinal_df_two_columns_three_values():
    df = pl.DataFrame(
        {
            "col1": ["a", "b", "c", "a", "b", "c"],
            "col2": ["x", "y", "z", "x", "y", "z"],
        }
    )
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1", "col2")
    assert len(odf.encoders) == 2
    assert odf.data.shape == (6, 2)
    assert set(odf.data[:, 0].tolist()) == {0, 1, 2}
    assert set(odf.data[:, 1].tolist()) == {0, 1, 2}


def test_ordinal_df_three_columns_two_values():
    df = pl.DataFrame(
        {
            "col1": ["a", "b", "a", "b"],
            "col2": ["x", "y", "x", "y"],
            "col3": ["p", "q", "p", "q"],
        }
    )
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1", "col2", "col3")
    assert len(odf.encoders) == 3
    assert odf.data.shape == (4, 3)
    assert set(odf.data[:, 0].tolist()) == {0, 1}
    assert set(odf.data[:, 1].tolist()) == {0, 1}
    assert set(odf.data[:, 2].tolist()) == {0, 1}


def test_ordinal_df_four_columns_two_values():
    df = pl.DataFrame(
        {
            "col1": ["a", "b"],
            "col2": ["x", "y"],
            "col3": ["p", "q"],
            "col4": ["m", "n"],
        }
    )
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1", "col2", "col3", "col4")
    assert len(odf.encoders) == 4
    assert odf.data.shape == (2, 4)
    assert set(odf.data[:, 0].tolist()) == {0, 1}
    assert set(odf.data[:, 1].tolist()) == {0, 1}
    assert set(odf.data[:, 2].tolist()) == {0, 1}
    assert set(odf.data[:, 3].tolist()) == {0, 1}


def test_ordinal_df_data_is_jax_array():
    df = pl.DataFrame({"col1": ["a", "b", "c"]})
    odf = OrdinalDF.from_dataframe(df)

    assert isinstance(odf.data, jnp.ndarray)


def test_ordinal_df_encoding_consistency():
    df = pl.DataFrame({"col1": ["a", "b", "a", "b", "a"]})
    odf = OrdinalDF.from_dataframe(df)

    encoded_values = odf.data[:, 0]
    assert encoded_values[0] == encoded_values[2] == encoded_values[4]
    assert encoded_values[1] == encoded_values[3]
    assert encoded_values[0] != encoded_values[1]


def test_ordinal_df_independent_column_encoding():
    df = pl.DataFrame(
        {
            "col1": ["a", "b", "c"],
            "col2": ["a", "b", "c"],
        }
    )
    odf = OrdinalDF.from_dataframe(df)

    assert jnp.array_equal(odf.data[:, 0], odf.data[:, 1])


def test_ordinal_df_different_value_sets_per_column():
    df = pl.DataFrame(
        {
            "col1": ["a", "b", "a"],
            "col2": ["x", "y", "z"],
        }
    )
    odf = OrdinalDF.from_dataframe(df)

    assert set(odf.data[:, 0].tolist()) == {0, 1}
    assert set(odf.data[:, 1].tolist()) == {0, 1, 2}


def test_ordinal_df_numeric_strings():
    df = pl.DataFrame(
        {
            "col1": ["1", "2", "3", "1"],
            "col2": ["10", "20", "10", "20"],
        }
    )
    odf = OrdinalDF.from_dataframe(df)

    assert odf.columns == ("col1", "col2")
    assert odf.data.shape == (4, 2)
    assert set(odf.data[:, 0].tolist()) == {0, 1, 2}
    assert set(odf.data[:, 1].tolist()) == {0, 1}


def test_ordinal_df_mixed_length_values():
    df = pl.DataFrame(
        {
            "col1": ["a", "abc", "abcdef", "a"],
            "col2": ["x", "xyz", "x", "xyz"],
        }
    )
    odf = OrdinalDF.from_dataframe(df)

    assert odf.data.shape == (4, 2)
    assert set(odf.data[:, 0].tolist()) == {0, 1, 2}
    assert set(odf.data[:, 1].tolist()) == {0, 1}


def test_ordinal_df_from_dataframes_consistent_encoding():
    df1 = pl.DataFrame({"col1": ["a", "b"], "col2": ["x", "y"]})
    df2 = pl.DataFrame({"col1": ["b", "c"], "col2": ["y", "z"]})

    odfs = OrdinalDF.from_dataframes([df1, df2])

    assert len(odfs) == 2
    assert odfs[0].columns == odfs[1].columns == ("col1", "col2")

    # Both should have access to all categories (a,b,c) and (x,y,z)
    # even though they don't all appear in each dataframe
    assert len(odfs[0].encoders[0].categories_[0]) == 3  # a, b, c
    assert len(odfs[1].encoders[0].categories_[0]) == 3  # a, b, c
    assert len(odfs[0].encoders[1].categories_[0]) == 3  # x, y, z
    assert len(odfs[1].encoders[1].categories_[0]) == 3  # x, y, z


def test_ordinal_df_from_dataframes_missing_category():
    # df1 has only "a" and "b", df2 has "a" and "c"
    # The key is that "c" should still be encodable in df1's encoder
    df1 = pl.DataFrame({"col1": ["a", "b"]})
    df2 = pl.DataFrame({"col1": ["a", "c"]})

    odfs = OrdinalDF.from_dataframes([df1, df2])

    # Both encoders should know about all three categories
    assert len(odfs[0].encoders[0].categories_[0]) == 3
    assert len(odfs[1].encoders[0].categories_[0]) == 3

    # The encoders should be the same object
    assert odfs[0].encoders[0] is odfs[1].encoders[0]


def test_ordinal_df_from_dataframes_single_dataframe():
    df = pl.DataFrame({"col1": ["a", "b", "c"]})

    odfs = OrdinalDF.from_dataframes([df])

    assert len(odfs) == 1
    assert odfs[0].columns == ("col1",)
    assert odfs[0].data.shape == (3, 1)


def test_ordinal_df_from_dataframes_three_dataframes():
    df1 = pl.DataFrame({"col1": ["a"], "col2": ["x"]})
    df2 = pl.DataFrame({"col1": ["b"], "col2": ["y"]})
    df3 = pl.DataFrame({"col1": ["c"], "col2": ["z"]})

    odfs = OrdinalDF.from_dataframes([df1, df2, df3])

    assert len(odfs) == 3
    # All should share the same encoders
    assert odfs[0].encoders[0] is odfs[1].encoders[0] is odfs[2].encoders[0]
    assert odfs[0].encoders[1] is odfs[1].encoders[1] is odfs[2].encoders[1]


def test_ordinal_df_from_dataframes_mismatched_columns_raises():
    df1 = pl.DataFrame({"col1": ["a", "b"]})
    df2 = pl.DataFrame({"col2": ["x", "y"]})

    with pytest.raises(AssertionError, match="same columns"):
        OrdinalDF.from_dataframes([df1, df2])


def test_ordinal_df_from_dataframes_empty_list_raises():
    with pytest.raises(AssertionError, match="at least one"):
        OrdinalDF.from_dataframes([])
