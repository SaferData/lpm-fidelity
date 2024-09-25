import itertools
import polars as pl
import pytest
from lpm_fidelity.optimal_transport_distance import (
    score_ot,
    DISTANCE_METRICS,
    OT_SOLVERS,
)

# Approximating zero with the following tolerance.
TOLERANCE = 1e-4

df1 = pl.DataFrame(
    {
        "column-1": ["foo", "bar"],
        "column-2": ["bar", "quagga"],
        "column-3": ["baz", "foo"],
    }
)
df2 = pl.DataFrame(
    {
        "column-1": ["foo", "foo"],
        "column-2": ["bar", "quagga"],
        "column-3": ["baz", "baz"],
    }
)


@pytest.mark.parametrize(
    "distance_metric,ot_solver", list(itertools.product(DISTANCE_METRICS, OT_SOLVERS))
)
def test_score_ot_positive(distance_metric, ot_solver):
    assert (
        score_ot(df1, df2, distance_metric=distance_metric, ot_solver=ot_solver) > 0.0
    )


@pytest.mark.parametrize(
    "distance_metric,ot_solver", list(itertools.product(DISTANCE_METRICS, OT_SOLVERS))
)
def test_score_ot_zero(distance_metric, ot_solver):
    assert score_ot(
        df1, df1, distance_metric=distance_metric, ot_solver=ot_solver
    ) == pytest.approx(0, abs=TOLERANCE)


# XXX
@pytest.mark.xfail(
    reason="The cost matrix seems incorrect in the presence of missing data"
)
@pytest.mark.parametrize(
    "distance_metric,ot_solver", list(itertools.product(DISTANCE_METRICS, OT_SOLVERS))
)
def test_score_ot_zero_with_missing(distance_metric, ot_solver):
    df_missing = pl.DataFrame(
        {
            "column-1": ["foo", "bar"],
            "column-2": ["bar", "quagga"],
            "column-3": ["baz", None],
        }
    )
    assert score_ot(
        df_missing, df_missing, distance_metric=distance_metric, ot_solver=ot_solver
    ) == pytest.approx(0, abs=TOLERANCE)
