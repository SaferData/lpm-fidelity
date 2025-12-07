import numpy as np
import polars as pl
from scipy.stats import chi2_contingency

from lpm_fidelity.counting import contingency_table


def chi_squared(column_a, column_b):
    """
    Compute the Chi^2 contigency table-based two samples test.
    First, create a sound contingency table. Then compute the test.
    Finally, record any problems.

    Parameters:
    - column_a (List or Polars Series): A list categories
    - column_b (List or Polars Series): Another list with same kinds of categories.

    Returns:
    - p (float):  A p-value assessing the null hypothesis that the columns record the same distribution.
    - chi2 (float): Chi^2 test statistic.
    - problem (boolean): Recording a problems with the test, e.g. when not
      enough data was recorded in one of the columns.


    Examples:
    >>> chi_squared(
            pl.Series("foo", ["a", "b", "a", "a",...]),
            pl.Series("foo", ["a", "b", "b", "a",...]))
        0.08, 42., false
    >>> chi_squared(
            ["a", "b", "a", "a",...],
            ["a", "b", "b", "a",...])
        0.08, 42., false
    """
    contingency_table_ab = contingency_table(column_a, column_b)
    chi2, p, dof, expected = chi2_contingency(contingency_table_ab)
    # Apply heuristic about when Chi-squared is not supposed to work.
    problem = (np.min(contingency_table_ab) == 1.0) and (dof <= 5)
    return p, chi2, problem


def univariate_two_sample_testing_in_data(df_a, df_b):
    """
    Take two dataframes and compute the Chi^2 contigency table-based two samples test
    for each column.
    Loop over all columns First, create a sound contingency tables for each.
    Then compute the test. Finally, record any problems.

    Parameters:
    - df_a:  Polars Dataframe
    - df_b:  Polars Dataframe

    Returns:
        A Polars Dataframe with a column "column" recording columns names,
        a column recording the uncorrected p-value and whether there
        was a problem with sparsity that scipy happily ignores.
    """
    assert set(df_a.columns) == set(df_b.columns)

    def _row(column):
        p, chi2, problem = chi_squared(df_a[column], df_b[column])
        return {
            "column": column,
            "p-value": p,
            "test statistic": chi2,
            "problem with sparsity": problem,
        }

    result = [_row(column) for column in df_a.columns]
    return pl.DataFrame(result).sort("p-value", descending=False)
