import numpy as np
import pandas as pd
from vtools import ts_coarsen
import pytest

def test_basic_grid_quantize_and_preserve():
    # We need the input to extend past 00:06 so resample("1min") produces 00:06:00.
    # Construct a preserved run (0.0) that exits at 00:06.
    t = pd.to_datetime([
        "2024-01-01 00:00:10",  # non-preserved
        "2024-01-01 00:01:10",  # enter preserved (0.0)
        "2024-01-01 00:05:50",  # still preserved right before exit
        "2024-01-01 00:06:10",  # exit preserved (nonzero)
    ])
    df = pd.DataFrame({"x": [5.0, 0.0, 0.0, 5.0]}, index=t)

    out = ts_coarsen(
        df,
        grid="1min",
        qwidth=0.1,
        preserve_vals=(0.0,),
        heartbeat_freq=None,
        hyst=0.5,
    )

    # grid-aligned
    assert (out.index.second == 0).all()

    # the pre-exit minute must exist (last preserved tick before leaving)
    assert pd.Timestamp("2024-01-01 00:05:00") in out.index

    # and we should see something at/after the exit
    assert (out.index >= pd.Timestamp("2024-01-01 00:06:00")).any()



def test_no_preserve_means_simple_thin():
    t = pd.date_range("2020-01-01", periods=5, freq="1min")
    df = pd.DataFrame({"x": [1, 1, 1, 1, 1]}, index=t)

    out = ts_coarsen(df, grid="1min", preserve_vals=(), heartbeat_freq=None)
    assert len(out) == 1


def test_heartbeat_optional():
    t = pd.date_range("2020-01-01", periods=10, freq="10min")
    df = pd.DataFrame({"x": np.ones(len(t))}, index=t)

    out = ts_coarsen(df, grid=None, heartbeat_freq="30min")
    assert len(out) > 1

    out2 = ts_coarsen(df, grid=None, heartbeat_freq=None)
    assert len(out2) == 1



def test_use_original_vals_false_disallowed_with_grid():
    df = pd.DataFrame(
        {"x": [0.0, 0.02, 0.04]},
        index=pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 00:00:30", "2024-01-01 00:01:00"]
        ),
    )

    with pytest.raises(ValueError, match="use_original_vals=False is not supported"):
        ts_coarsen(
            df,
            grid="1min",
            qwidth=0.05,
            use_original_vals=False,
            heartbeat_freq=None,
            hyst=0.5,
        )