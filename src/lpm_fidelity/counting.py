import sys
from functools import partial

import equinox as eqx
import polars as pl
from jax import Array, jit
from jax import numpy as jnp
from jaxtyping import Integer
from sklearn.preprocessing import OrdinalEncoder


def _is_nan(value) -> bool:
    """Check if value is NaN or None (handles non-numeric types safely)."""
    match value:
        case None:
            return True
        case Array() | float():
            return bool(jnp.isnan(value))
        case _:
            return False


class OrdinalDF(eqx.Module):
    columns: tuple[str, ...]
    encoders: tuple[OrdinalEncoder, ...] = eqx.field(static=True)
    data: Array

    def __init__(self, df: pl.DataFrame, encoders: dict[str, OrdinalEncoder]):
        self.columns = tuple(encoders.keys())
        self.encoders = tuple(encoders.values())
        self.data = jnp.column_stack(
            [
                jnp.array(encoders[c].transform([[x] for x in df[c].to_list()]))
                for c in self.columns
            ]
        ).astype(jnp.int32)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> "OrdinalDF":
        return cls.from_dataframes([df])[0]

    @classmethod
    def from_dataframes(cls, dfs: list[pl.DataFrame]) -> list["OrdinalDF"]:
        """A method for creating multiple OrdinalDFs under the assumption
        that they have the same columns with the same possible categorical
        values; use this method to ensure they encode&decode consistently,
        and to protect against issues in the rare case that some value
        happens not to appear in some dataframes.
        """
        assert len(dfs) > 0, "Must provide at least one dataframe"

        # 1. check that all dataframes have same set() of columns
        #    n.b. this counting method allows a list of 1 df, to support shared impl
        first_columns = set(dfs[0].columns)
        for df in dfs[1:]:
            assert set(df.columns) == first_columns, (
                "All dataframes must have the same columns"
            )

        # 2. concatenate dataframes vertically
        concatenated = pl.concat(dfs, how="vertical")

        # 3. create an encoders dict from the concatenated dataframe
        # Use handle_unknown to map None -> -1 automatically
        encoders = {}
        for c in concatenated.columns:
            col_list = concatenated[c].to_list()
            col_data = [[x] for x in col_list]
            unique_vals_set = set(col_list)
            # Exclude None from categories - they will be handled as unknown -> -1
            non_null_vals = sorted([v for v in unique_vals_set if not _is_nan(v)])
            enc = OrdinalEncoder(
                categories=[non_null_vals],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            enc.fit(col_data)
            encoders[c] = enc

        # 4. create an OrdinalDF *for each input dataframe* using the encoders dict
        return [cls(df, encoders) for df in dfs]


@partial(jit, static_argnums=(1,))
def normalize_count_memoized(
    column: Integer[Array, "n"], unique_vals: int
) -> tuple[Array, int]:
    # Mask for valid (non-null) values
    valid = column != -1

    # Index; invalid rows map to 0 (harmless, masked out)
    idx = jnp.where(valid, column, 0)

    # Vectorized scatter-add: adds 1 for valid, 0 for invalid
    counts = (
        jnp.zeros(unique_vals, dtype=jnp.int32).at[idx].add(valid.astype(jnp.int32))
    )

    return counts, jnp.sum(counts)


def normalize_count(
    column: list | pl.Series,
) -> dict[str, float]:
    """
    Count occurences of categories. This works on Polars'columns
    i.e. Polars Series.

    NOTE: this function exists for backwards-compatibility and human
    readability; most users will use `distances.py` which calls the
    underlying accelerated implementation.

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
    df = pl.DataFrame({"col": column})
    odf = OrdinalDF.from_dataframe(df)
    categories = odf.encoders[0].categories_[0]
    n_unique = len(categories)
    counts, total = normalize_count_memoized(odf.data[:, 0], n_unique)
    assert total > 0
    return {cat: float(counts[i]) / float(total) for i, cat in enumerate(categories)}


@partial(jit, static_argnums=(1, 2))
def normalize_count_bivariate_memoized(
    cols_1_and_2: Integer[Array, "n 2"],
    c1_uniq_vals: int,
    c2_uniq_vals: int,
) -> tuple[Array, int]:
    """
    Count co-occurrences of category pairs from two ordinal-encoded columns.

    Uses vectorized scatter-add for efficient counting. Pairs containing -1
    (null sentinel) are excluded from the count.

    Parameters:
        cols_1_and_2: JAX array of shape (n, 2) with ordinal-encoded values.
            Values of -1 indicate null/missing.
        c1_uniq_vals: Number of unique categories in the first column.
        c2_uniq_vals: Number of unique categories in the second column.

    Returns:
        Tuple of (counts, total) where:
        - counts: JAX array of shape (c1_uniq_vals, c2_uniq_vals) with counts
        - total: Total number of valid (non-null) pairs counted
    """
    col1 = cols_1_and_2[:, 0]
    col2 = cols_1_and_2[:, 1]

    # Mask for valid (non-null) pairs
    valid = (col1 != -1) & (col2 != -1)

    # Linear index; invalid rows map to 0 (harmless, masked out)
    idx = jnp.where(valid, col1 * c2_uniq_vals + col2, 0)

    # Vectorized scatter-add: adds 1 for valid, 0 for invalid
    counts = (
        jnp.zeros(c1_uniq_vals * c2_uniq_vals, dtype=jnp.int32)
        .at[idx]
        .add(valid.astype(jnp.int32))
        .reshape(c1_uniq_vals, c2_uniq_vals)
    )

    return counts, jnp.sum(counts)


def normalize_count_bivariate(
    column_1: list | pl.Series,
    column_2: list | pl.Series,
    overlap_required: bool = True,
) -> dict[tuple[str, str], float]:
    """
    Count occurences of events between two categorical columns.
    This works on Polars'columns i.e. Polars Series.

    NOTE: this function exists for backwards-compatibility and human
    readability; most users will use `distances.py` which calls the
    underlying accelerated implementation.

    Parameters:
    - column_1 (List or Polars Series):  A column in a dataframe.
    - column_2 (List or Polars Series):  Another column in a dataframe.
    - overlap_required bool:  If the two columns don't have non-null overlap,
                              throw error

    Returns:
    - dict: A Python dictionary, where keys are tuples of categories from the
      two columns and values are the normalized ([0,1]) counts.


    Examples:
    >>> normalize_count_bivariate(
            pl.Series("foo", ["a", "b", "a", "a"])
            pl.Series("foo", ["x", "y", "x", "y"]))

    {("a", "x",): 0.5, ("a", "y",): 0.25, ("b, "y",): 0.25}
    """
    df = pl.DataFrame({"c1": column_1, "c2": column_2})
    odf = OrdinalDF.from_dataframe(df)
    cats_1 = odf.encoders[0].categories_[0]
    cats_2 = odf.encoders[1].categories_[0]
    n_1, n_2 = len(cats_1), len(cats_2)

    counts, total = normalize_count_bivariate_memoized(odf.data, n_1, n_2)

    if overlap_required:
        assert total > 0, "no overlap"

    if total == 0:
        return {}

    result = {}
    for i, cat_1 in enumerate(cats_1):
        for j, cat_2 in enumerate(cats_2):
            count = float(counts[i, j])
            if count > 0:
                result[(cat_1, cat_2)] = count / float(total)
    return result


def harmonize_categorical_probabilities(ps_a, ps_b):
    """
    Harmonize two categorical distributions. Ensure they have the same set of
    keys.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use OrdinalDF.from_dataframes() for consistent category encoding across
        multiple dataframes.

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
    import warnings

    warnings.warn(
        "harmonize_categorical_probabilities is deprecated. "
        "Use OrdinalDF.from_dataframes() for consistent category encoding.",
        DeprecationWarning,
        stacklevel=2,
    )
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


def contingency_table(column_a, column_b) -> Array:
    """
    Compute the contigency table for two columns in a Polars dataframe.

    Parameters:
    - column_a (List or Polars Series):  A column in a dataframe.
    - column_b (List or Polars Series):  The same column from another dataframe.

    Returns:
    - contingency table (Array): a 2-d JAX array counting the contingencies.
    """
    # Sorting unique values here so it's testable. Otherwise, the set/filter
    # combinations causes for stochastic orderings.
    assert len(column_a) > 0
    assert len(column_b) > 0
    # Ensure columns are list without NaNs:
    column_a = [val for val in column_a if not _is_nan(val)]
    column_b = [val for val in column_b if not _is_nan(val)]
    unique_values = sorted(set(column_a + column_b))
    # Build counts as a list of rows, then stack into array (JAX arrays are immutable)
    rows = [[column_a.count(value), column_b.count(value)] for value in unique_values]
    return jnp.array(rows, dtype=jnp.float32)


def bivariate_empirical_frequencies(
    dfs: dict[str, pl.DataFrame], column_name_a: str, column_name_b: str
) -> pl.DataFrame:
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
    sources = list(dfs.keys())

    # Select only the two columns we need from each dataframe
    column_dfs = [dfs[s].select([column_name_a, column_name_b]) for s in sources]

    # Encode all sources together to ensure consistent category mapping
    odfs = OrdinalDF.from_dataframes(column_dfs)

    # Get category labels from the shared encoders
    cats_a = odfs[0].encoders[0].categories_[0]
    cats_b = odfs[0].encoders[1].categories_[0]
    n_a, n_b = len(cats_a), len(cats_b)

    # Build result rows
    rows: list[tuple[str, str, float, str]] = []
    for source, odf in zip(sources, odfs):
        memo, total = normalize_count_bivariate_memoized(odf.data, n_a, n_b)
        # Convert matrix to probabilities; handle total=0 case
        if not (total > 0):
            raise ValueError("no overlap")
        probs = memo / total.astype(jnp.float32)
        for i, cat_a in enumerate(cats_a):
            for j, cat_b in enumerate(cats_b):
                rows.append((cat_a, cat_b, float(probs[i, j]), source))

    return pl.DataFrame(
        rows,
        schema=[column_name_a, column_name_b, "Normalized frequency", "Source"],
        orient="row",
    ).sort([column_name_a, column_name_b])
