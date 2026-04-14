import pytest
import pandas as pd
from vtools.data.vtime import divide_interval, compare_interval


def test_int_division():
    assert divide_interval(24, 1) == 24


def test_string_offsets():
    assert divide_interval("1D", "1h") == 24
    assert divide_interval("15min", "5min") == 3


def test_timedelta():
    assert divide_interval(pd.Timedelta("1D"), pd.Timedelta("1h")) == 24


def test_mixed_types():
    assert divide_interval("1D", pd.Timedelta("1h")) == 24
    assert divide_interval(pd.Timedelta("1D"), "1h") == 24


def test_non_integer_division_fails():
    with pytest.raises(ValueError):
        divide_interval("1D", "7h")


def test_month_rejected():
    with pytest.raises(TypeError):
        divide_interval("1M", "1D")


def test_year_rejected():
    with pytest.raises(TypeError):
        divide_interval("1Y", "1D")


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        divide_interval("1D", "0H")

def test_ratio_ok_when_not_int_required():
    assert divide_interval("1D", "7h", require_int=False) == pytest.approx(24/7)

def test_mixed_scalar_interval_rejected():
    with pytest.raises(TypeError):
        divide_interval(24, "1h")
    with pytest.raises(TypeError):
        divide_interval("1h", 24)
        
        
def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        divide_interval("1D", "0h")


def test_int_division():
    assert divide_interval(24, 1) == 24


def test_string_offsets():
    assert divide_interval("1D", "1h") == 24
    assert divide_interval("15min", "5min") == 3


def test_timedelta():
    assert divide_interval(pd.Timedelta("1D"), pd.Timedelta("1h")) == 24


def test_mixed_types():
    assert divide_interval("1D", pd.Timedelta("1h")) == 24
    assert divide_interval(pd.Timedelta("1D"), "1h") == 24


def test_non_integer_division_fails():
    with pytest.raises(ValueError):
        divide_interval("1D", "7h")


def test_month_rejected():
    with pytest.raises(TypeError):
        divide_interval("1M", "1D")


def test_year_rejected():
    with pytest.raises(TypeError):
        divide_interval("1Y", "1D")


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        divide_interval("1D", "0h")


def test_ratio_ok_when_not_int_required():
    assert divide_interval("1D", "7h", require_int=False) == pytest.approx(24 / 7)


def test_mixed_scalar_interval_rejected():
    with pytest.raises(TypeError):
        divide_interval(24, "1h")
    with pytest.raises(TypeError):
        divide_interval("1h", 24)


# ---- comparison tests ----

def test_compare_interval_equal_strings():
    assert compare_interval("1D", "24h") == 0


def test_compare_interval_day_greater_than_hour():
    assert compare_interval("1D", "1h") == 1
    assert compare_interval("1h", "1D") == -1


def test_compare_interval_offset_and_timedelta():
    assert compare_interval(pd.offsets.Day(1), pd.Timedelta("24h")) == 0
    assert compare_interval(pd.offsets.Hour(1), pd.Timedelta("30min")) == 1


def test_compare_interval_minute_vs_second():
    assert compare_interval("15min", "300s") == 1
    assert compare_interval("300s", "15min") == -1


def test_compare_interval_reject_month():
    with pytest.raises(TypeError):
        compare_interval("1M", "1D")


def test_compare_interval_reject_year():
    with pytest.raises(TypeError):
        compare_interval("1Y", "1D")


def test_compare_interval_reject_scalar_interval_mixed():
    with pytest.raises(TypeError):
        compare_interval(24, "1h")
    with pytest.raises(TypeError):
        compare_interval("1h", 24)
        
def test_compare_interval_handles_pandas_day_offset():
    assert compare_interval(pd.offsets.Day(), pd.offsets.Hour(12)) == 1


def test_compare_interval_handles_fixed_offsets_consistently():
    freqs = [pd.offsets.Day(), pd.offsets.Hour(1), pd.offsets.Minute(15)]
    assert min(freqs, key=lambda x: pd.Timedelta(x.nanos, unit="ns")) == pd.offsets.Minute(15)
    