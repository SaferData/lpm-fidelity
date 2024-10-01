import polars as pl
from collections import Counter
import sys
import numpy as np


def _is_none_or_nan(value):
    if value is None:
        return True
    # Try checking if value is np.nan, catching TypeError for non-numeric values
    try:
        if np.isnan(value):
            return True
    except TypeError:
        pass
    # Value is neither None nor np.nan
    return False


def _is_none_or_nan_bivariate(values):
    return _is_none_or_nan(values[0]) or _is_none_or_nan(values[1])


def normalize_count(column):
    """
    Count occurences of categories. This works on Polars'columns
    i.e. Polars Series.

    Parameters:
    - column (List or Polars Series): A column in a dataframe.

    Returns:
    - dict: A Python dictionary, where keys are categories and values are the
      normalized ([0,1]) counts.


    Examples:
    >>> normalize_count(pl.Series("foo", ["a", "b", "a", "a"]))
        {"a": 0.75, "b" 0.25}
    >>> normalize_count(["a", "b", "a", "a"])
        {"a": 0.75, "b" 0.25}
    """
    column = [val for val in column if not _is_none_or_nan(val)]
    assert len(column) > 0
    return {k: v / len(column) for k, v in pl.Series(column).value_counts().rows()}


def normalize_count_bivariate(column_1, column_2, overlap_required=True):
    """
    Count occurences of events between two categorical columns.
    This works on Polars'columns i.e. Polars Series.

    Parameters:
    - column_1 (List or Polars Series):  A column in a dataframe.
    - column_2 (List or Polars Series):  Another column in a dataframe.
    - overlap_required bool:  If the two columns don't have non-null overlap,
                              throw error

    Returns:
    - dict: A Python dictionary, where keys are typles of categories from the
      two columns and values are the normalized ([0,1]) counts.


    Examples:
    >>> normalize_count_bivariate(
            pl.Series("foo", ["a", "b", "a", "a"])
            pl.Series("foo", ["x", "y", "x", "y"]))

    {("a", "x",): 0.5, ("a", "y",): 0.25, ("b, "y",): 0.25}
    """
    assert len(column_1) == len(column_2)
    assert len(column_1) > 0
    assert len(column_2) > 0

    column_values = [
        vals for vals in zip(column_1, column_2) if not _is_none_or_nan_bivariate(vals)
    ]
    if overlap_required:
        assert len(column_values) > 0, "no overlap"
    counter = Counter(column_values)
    # Note that Polars doesn't like to count tuples.
    return {k: v / len(column_values) for k, v in dict(counter).items()}


def harmonize_categorical_probabilities(ps_a, ps_b):
    """
    Harmonize two categorical distributions. Ensure they have the same set of
    keys.

    Parameters:
    - ps_a (dict): A dict encoding a categorical probability distribution.
    - ps_b (dict): A dict encoding a categorical probability distribution.

    Returns:
    - ps_a_harmonzied (dict): A dict encoding a categorical
                              probability distribution.
    - ps_b_harmonzied (dict): A dict encoding a categorical
                              probability distribution.


    Examples:
        >>> harmonize_categorical_probabilities({"a": 0.1, "b": 0.9}, {"a": 1.0})
            {"a": 0.1, "b": 0.9}, {"a": 1.0, "b" 0.0}
        >>> harmonize_categorical_probabilities({"a": 1.0}, {"a": 0.1, "b": 0.9})
            {"a": 1.0, "b" 0.0}, {"a": 0.1, "b": 0.9}
    """
    # Get the union of keys from both dictionaries
    assert (len(ps_a) > 0) or (len(ps_b) > 0)
    all_keys = set(ps_a) | set(ps_b)
    # Update both dictionaries to contain all keys, setting default values to None for missing keys
    return {key: ps_a.get(key, 0.0) for key in all_keys}, {
        key: ps_b.get(key, 0.0) for key in all_keys
    }


def _probabilities_safe_as_denominator(ps, constant=sys.float_info.min):
    """
    Ensure all values in a categorical are larger than 0. Some distance metrics,
    like SciPy's JS distance require this.

    The Constant should be chosen so small that it does not affect any results.
    Other state-of-the-art-libraries do similar things,
    .e.g. here: https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py#L273

    Parameters:
    - ps (dict): A dict encoding a categorical probability distribution.
    - constant (float): constant to be added to zero values.

    Returns:
    - ps_larger_zero (dict): A dict encoding a categorical probability
      distribution. All values are larger than zero.


    Examples:
        >>> p_larger_zero({"a": 1.0, "b": 0.0}, constant=0.00000001)
            {"a": 1.0, "b" 0.00000001}
    """

    def _add_constant_if_zero(v):
        if v == 0.0:
            return v + constant
        return v

    return {k: _add_constant_if_zero(v) for k, v in ps.items()}


def contingency_table(column_a, column_b):
    """
    Compute the contigency table for two columns in a Polars dataframe.

    Parameters:
    - column_a (List or Polars Series):  A column in a dataframe.
    - column_b (List or Polars Series):  The same column from another dataframe.

    Returns:
    - contingency table (np.arrray): a 2-d Numpy array couting the contigencies.
    """
    # Sorting unique values here so it's testable. Otherwise, the set/filter
    # combinations causes for stochatic orderings.
    assert len(column_a) > 0
    assert len(column_b) > 0
    # Ensure columns are list without NaNs:
    column_a = [val for val in column_a if not _is_none_or_nan(val)]
    column_b = [val for val in column_b if not _is_none_or_nan(val)]
    unique_values = sorted(set(column_a + column_b))
    contingency_table = np.zeros((len(unique_values), 2))
    for i, value in enumerate(unique_values):
        contingency_table[i, 0] = column_a.count(value)
        contingency_table[i, 1] = column_b.count(value)
    return contingency_table


def bivariate_empirical_frequencies(dfs, column_name_a, column_name_b):
    """
    Computes the normalized bivariate empirical frequencies for two categorical variables
    across multiple data sources and returns the result as a Polars DataFrame.

    This function takes a dictionary of Polars dataframes (`dfs`), where each dataframe represents
    a dataset from a different source, and calculates the normalized joint frequency of
    occurrences for the two categorical columns `column_name_a` and `column_name_b`. The
    function ensures that all datasets are harmonized, meaning they contain the same
    categories for the specified columns, by matching the categories across all sources.

    Parameters:
    - dfs (dict of {str: pl.DataFrame}): A dictionary where keys are the source names
        (strings) and values are Polars DataFrames containing the data.
    - column_name_a (str): The name of the first categorical column to be used for
        computing bivariate frequencies.
    - column_name_b (str): The name of the second categorical column to be used for
        computing bivariate frequencies.

    Returns:
    - pl.DataFrame: A Polars DataFrame with the following columns:
            - `column_name_a`: The categories from the first column.
            - `column_name_b`: The categories from the second column.
            - `"Normalized frequency"`: The normalized joint frequency for the pair of categories.
            - `"Source"`: The source of the data (corresponding to the keys from `dfs`).

        The resulting DataFrame is sorted by `column_name_a` and `column_name_b`.
    """
    # First, get the empirically counted frequeencies for each df in dfs.
    counts = {
        source: normalize_count_bivariate(df[column_name_a], df[column_name_b])
        for source, df in dfs.items()
    }
    # Next, we have to make sure we have all categories; we create a reference
    # count.
    sources = list(counts.keys())
    ref_count = counts[sources[0]]
    for source in sources[1:]:
        ref_count, _ = harmonize_categorical_probabilities(ref_count, counts[source])

    # Loop over all empirical counts and record results.
    empirical_counts = {
        column_name_a: [],
        column_name_b: [],
        "Normalized frequency": [],
        "Source": [],
    }
    for source, count in counts.items():
        _, count = harmonize_categorical_probabilities(ref_count, count)
        for (x, y), v in count.items():
            empirical_counts[column_name_a].append(x)
            empirical_counts[column_name_b].append(y)
            empirical_counts["Normalized frequency"].append(v)
            empirical_counts["Source"].append(source)
    return pl.DataFrame(empirical_counts).sort([column_name_a, column_name_b])
