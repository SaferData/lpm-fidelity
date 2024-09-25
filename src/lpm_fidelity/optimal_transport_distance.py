import ot
import numpy as np
import polars as pl


def get_mapping(df: pl.DataFrame, columns=None):
    if columns is None:
        columns = df.columns
    mapping = {}
    for c in columns:
        mapping[c] = {
            v: i for i, v in enumerate(df[c].drop_nulls().unique().sort().to_list())
        }
    return mapping


def categorical_df_to_integer(df: pl.DataFrame, mapping: dict):
    for c in df.columns:
        df = df.with_columns([pl.col(c).replace(mapping[c]).cast(pl.Int64)])
    array = df.with_columns(pl.all().to_physical()).to_numpy()
    return array


# We are currently restricting distance metrics to these because we only want to
# run this on categorical data.
DISTANCE_METRICS = ["jaccard", "hamming", "dice", "matching"]
OT_SOLVERS = ["emd", "sinkhorn"]


def score_ot(df1, df2, distance_metric="jaccard", ot_solver="sinkhorn"):
    """
    The score_ot function computes the Wasserstein loss (transport cost) between
    two categorical Polars dataframes using an optimal transport algorithm. This
    function supports currently supports subsets of different distance metrics
    implemented by POT (https://github.com/PythonOT/POT) and two of POT's optimal
    transport solvers: emd (Earth Mover's Distance) and sinkhorn (entropic regularization).

    The function assumes that the two dataframes have categorical columns.

    Parameters:
    - df1 (pl.DataFrame): The first dataframe to compare. It must contain only categorical
      columns with Utf8 dtype.
    - df2 (pl.DataFrame): The second dataframe to compare. It must also contain only categorical
      columns with Utf8 dtype and have the same shape and column names as df1.
    - distance_metric (str, optional): The distance metric to use for computing
      the cost matrix between categorical values. Default is "jaccard". Must be
      one of "jaccard", "hamming", "dice", "matching".
    - ot_solver (str, optional): The optimal transport solver to use. Default is "sinkhorn". Supported
      values are "emd" (Earth Mover's Distance) and "sinkhorn" (entropic regularization).

    Returns:
        float: The Wasserstein loss between df1 and df2, which represents the transport cost based on the chosen distance metric and OT solver.
    Examples:
    >>> from lpm_fidelity.optimal_transport_distance import score_ot
        import polars as pl
        df1 = pl.DataFrame({
            "col1": ["A", "B", "C"],
            "col2": ["X", "Y", "Z"]
        })
        df2 = pl.DataFrame({
            "col1": ["B", "A", "C"],
            "col2": ["Y", "X", "Z"]
        })
        wasserstein_loss = score_ot(df1, df2, distance_metric="jaccard", ot_solver="sinkhorn")
        print(f"Wasserstein loss: {wasserstein_loss}")
    """
    assert df1.columns == df2.columns
    assert set(df1.dtypes + df2.dtypes) == set([pl.Utf8])
    assert df1.shape == df2.shape
    assert distance_metric in DISTANCE_METRICS
    assert ot_solver in OT_SOLVERS
    # Convert to integers.
    mapping = get_mapping(pl.concat([df1, df2]))
    data1 = categorical_df_to_integer(df1, mapping)
    data2 = categorical_df_to_integer(df2, mapping)
    cost_matrix = ot.dist(data1, data2, metric=distance_metric)
    # Sample weights need to be supplied. We assume uniformity.
    n = len(df1)
    sample_weights_1, sample_weights_2 = np.ones((n,)) / n, np.ones((n,)) / n
    if ot_solver == "emd":
        # Solve for optimal transport matrix via earth mover distance problem.
        optimal_transport_matrix = ot.emd(
            sample_weights_1, sample_weights_2, cost_matrix
        )
    elif ot_solver == "sinkhorn":
        # Solve the entropic regularization optimal transport problem and return the OT matrix
        regularization_term = 1e-1  # This is the entropic regularization term.
        optimal_transport_matrix = ot.sinkhorn(
            sample_weights_1, sample_weights_2, cost_matrix, regularization_term
        )
    else:
        ValueError(f"{ot_solver} currently not supported by LPM_fidelity")
    # Return the  Wasserstein Loss.
    return np.sum(optimal_transport_matrix * cost_matrix)
