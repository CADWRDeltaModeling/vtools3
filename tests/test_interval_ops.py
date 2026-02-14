import pytest
import pandas as pd
from vtools.data.vtime import divide_interval


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

        