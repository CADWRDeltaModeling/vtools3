import numpy as np
import pandas as pd
import pytest
from vtools.functions.ha_utils import nodal_factors


# ============================================================================
# Basic functionality tests
# ============================================================================

def test_single_constituent():
    """Test with single constituent"""
    t_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    assert F.shape == (10, 1)
    assert U.shape == (10, 1)
    assert V.shape == (10, 1)
    assert np.all(F > 0), "Amplitude factors should be positive"
    assert np.all(np.isfinite(F)), "F should contain no NaN or inf"
    assert np.all(np.isfinite(U)), "U should contain no NaN or inf"
    assert np.all(np.isfinite(V)), "V should contain no NaN or inf"


def test_multiple_constituents():
    """Test with multiple constituents"""
    t_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    assert F.shape == (10, 2)
    assert U.shape == (10, 2)
    assert V.shape == (10, 2)


@pytest.mark.parametrize("constituents", [
    ["M2"],
    ["K1"],
    ["S2"],
    ["M2", "K1"],
    ["M2", "K1", "S2", "N2"],
])
def test_various_constituents(constituents):
    """Test with various combinations of constituents"""
    t_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    assert F.shape == (10, len(constituents))
    assert U.shape == (10, len(constituents))
    assert V.shape == (10, len(constituents))


def test_invalid_constituent():
    """Test that invalid constituent raises error"""
    t_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    lat_deg = 38.0
    
    with pytest.raises(KeyError):
        nodal_factors(t_dates, tref_date, ["INVALID"], lat_deg)


def test_single_timestamp():
    """Test with single timestamp"""
    t_dates = pd.Timestamp("2020-01-01")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    # When single timestamp is passed, it should be treated as array-like
    assert F.ndim == 2
    assert U.ndim == 2
    assert V.ndim == 2



# ============================================================================
# Phase option tests
# ============================================================================

@pytest.mark.parametrize("phase", ["Greenwich", "linear_time", "raw"])
def test_all_phase_options(phase):
    """Test all valid phase options"""
    t_dates = pd.date_range("2020-01-01", periods=20, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg, phase=phase)
    
    assert F.shape == (20, 2)
    assert U.shape == (20, 2)
    assert V.shape == (20, 2)


def test_phase_options_produce_different_results():
    """Test that different phase options produce different V (astronomical arguments)"""
    t_dates = pd.date_range("2020-01-01", periods=20, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F_green, U_green, V_green = nodal_factors(t_dates, tref_date, constituents, lat_deg, phase="Greenwich")
    F_linear, U_linear, V_linear = nodal_factors(t_dates, tref_date, constituents, lat_deg, phase="linear_time")
    F_raw, U_raw, V_raw = nodal_factors(t_dates, tref_date, constituents, lat_deg, phase="raw")
    
    # V should be different for different phase options
    assert not np.allclose(V_green, V_linear)
    assert not np.allclose(V_green, V_raw)
    assert not np.allclose(V_linear, V_raw)


def test_invalid_phase_option():
    """Test that invalid phase option raises error"""
    t_dates = pd.date_range("2020-01-01", periods=20, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    with pytest.raises(ValueError):
        nodal_factors(t_dates, tref_date, constituents, lat_deg, phase="invalid")


# ============================================================================
# Nodal linear time option tests
# ============================================================================

def test_nodal_linear_time_false():
    """Test with nodal_linear_time=False (default)"""
    t_dates = pd.date_range("2020-01-01", periods=50, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg, nodal_linear_time=False)
    
    # F and U should have same length as t_dates
    assert F.shape == (50, 2)
    assert U.shape == (50, 2)
    # V always has same length as t_dates
    assert V.shape == (50, 2)


def test_nodal_linear_time_true():
    """Test with nodal_linear_time=True"""
    t_dates = pd.date_range("2020-01-01", periods=50, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg, nodal_linear_time=True)
    
    # F and U should have single row when nodal_linear_time=True
    assert F.shape == (1, 2)
    assert U.shape == (1, 2)
    # V still has same length as t_dates
    assert V.shape == (50, 2)


def test_nodal_linear_time_comparison():
    """Compare results between nodal_linear_time=True and False"""
    t_dates = pd.date_range("2020-01-01", periods=50, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F_full, U_full, V_full = nodal_factors(t_dates, tref_date, constituents, lat_deg, nodal_linear_time=False)
    F_linear, U_linear, V_linear = nodal_factors(t_dates, tref_date, constituents, lat_deg, nodal_linear_time=True)
    
    # V should be identical
    np.testing.assert_allclose(V_full, V_linear, rtol=1e-10)
    
    # F_linear should be single values, F_full should be time series
    assert F_linear.shape == (1, 2)
    assert F_full.shape == (50, 2)


# ============================================================================
# Time variability tests
# ============================================================================

def test_nodal_factors_vary_with_time():
    """Test that nodal factors vary over time"""
    # Use a longer time period to see variation
    t_dates = pd.date_range("2020-01-01", periods=365, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    # Check that F varies with time (at least for some constituents)
    # M2 is a semidiurnal tide with nodal period of ~18.6 years, so variation over 1 year is small
    # but should still be detectable
    F_std = np.std(F, axis=0)
    assert np.all(F_std > 0), "F should vary with time"
    
    # V should vary significantly with time
    V_std = np.std(V, axis=0)
    assert np.all(V_std > 0), "V should vary with time"


def test_v_increases_monotonically():
    """Test that astronomical argument V increases with time"""
    t_dates = pd.date_range("2020-01-01", periods=100, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    # V should generally increase with time (may wrap but generally increasing)
    # Check first and last values
    assert V[-1, 0] > V[0, 0] or (V[-1, 0] < 0.5 and V[0, 0] > 0.5)  # wrapping allowed


# ============================================================================
# Edge cases and boundary conditions
# ============================================================================

def test_very_short_time_series():
    """Test with minimal time series length"""
    t_dates = pd.date_range("2020-01-01", periods=1, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    assert F.shape == (1, 1)
    assert U.shape == (1, 1)
    assert V.shape == (1, 1)


def test_different_tref_date():
    """Test that different reference dates produce different results"""
    t_dates = pd.date_range("2022-06-01", periods=30, freq="D")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    tref1 = pd.Timestamp("2022-01-01")
    tref2 = pd.Timestamp("2022-07-01")
    nodal_linear_time = True
    
    F1, U1, V1 = nodal_factors(t_dates, tref1, constituents, lat_deg, nodal_linear_time=nodal_linear_time)
    F2, U2, V2 = nodal_factors(t_dates, tref2, constituents, lat_deg, nodal_linear_time=nodal_linear_time)
    
    # U should differ for different reference dates
    assert not np.allclose(U1, U2), "Phase corrections should differ for different tref_date"
    
    # F should differ for different reference dates
    assert not np.allclose(F1, F2), "Amplitude corrections should differ for different tref_date"


@pytest.mark.parametrize("lat_deg", [-90.0, -45.0, 0.0, 45.0, 90.0])
def test_extreme_latitudes(lat_deg):
    """Test with extreme latitude values"""
    t_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    assert F.shape == (10, 1)
    assert np.all(np.isfinite(F)), "Results should be finite at extreme latitudes"
    assert np.all(np.isfinite(U)), "Results should be finite at extreme latitudes"
    assert np.all(np.isfinite(V)), "Results should be finite at extreme latitudes"


def test_datetimeindex_input():
    """Test with DatetimeIndex input"""
    t_dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    assert F.shape == (3, 1)


def test_series_input():
    """Test with Series input (with datetime index)"""
    t_dates = pd.Series(
        [1, 2, 3],
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    )
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    lat_deg = 38.0
    
    F, U, V = nodal_factors(t_dates.index, tref_date, constituents, lat_deg)
    
    assert F.shape == (3, 1)


def test_invalid_tref_type():
    """Test that invalid tref_date type raises error"""
    t_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    constituents = ["M2"]
    lat_deg = 38.0
    
    with pytest.raises(ValueError):
        nodal_factors(t_dates, "2020-01-01", constituents, lat_deg)


def test_invalid_tdates_type():
    """Test that invalid t_dates type raises error"""
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2"]
    lat_deg = 38.0
    
    with pytest.raises(ValueError):
        nodal_factors([1, 2, 3], tref_date, constituents, lat_deg)


# ============================================================================
# Consistency tests
# ============================================================================

def test_same_inputs_same_outputs():
    """Test that identical inputs produce identical outputs"""
    t_dates = pd.date_range("2020-01-01", periods=20, freq="D")
    tref_date = pd.Timestamp("2020-01-01")
    constituents = ["M2", "K1"]
    lat_deg = 38.0
    
    F1, U1, V1 = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    F2, U2, V2 = nodal_factors(t_dates, tref_date, constituents, lat_deg)
    
    np.testing.assert_array_equal(F1, F2)
    np.testing.assert_array_equal(U1, U2)
    np.testing.assert_array_equal(V1, V2)



   
