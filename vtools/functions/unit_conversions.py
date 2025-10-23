"""
Unit conversion helpers.

This module provides:
- linear/affine converters for common engineering units:
  metres↔feet, cms↔cfs, °F↔°C (all *functional*, no in-place mutation).
- Domain-specific conversions between electrical conductivity (EC, μS/cm)
  and practical salinity (PSU) at 25 °C, with optional Hill low-salinity
  correction and an accuracy-improving root-finding “refinement” step.
- a general-purpose unit conversion function `convert_units()` that uses Pint
  by default (with an optional cf_units backend via an environment variable),
  and that has fast paths for the above common conversions.
  
Notes
-----
- PSU is treated here as a practical “unit” for salinity in workflows,
  even though in a strict metrological sense it is unitless.
- The EC↔PSU conversions assume **25 °C** and no explicit temperature
  dependence beyond the optional Hill correction.

References
----------
Schemel, L.E. (2001) Empirical relationships between salinity and
specific conductance in San Francisco Bay, California.

Hill, K. (low-salinity correction widely used in estuarine practice).
"""

from __future__ import annotations
import os
import re
import functools
import numpy as np
import pandas as pd
from scipy.optimize import brentq



# -----------------------------------------------------------------------------
# Constants for conversions (kept from legacy implementation)
# -----------------------------------------------------------------------------
J1 = -16.072
J2 = 4.1495
J3 = -0.5345
J4 = 0.0261

K1 = 0.0120
K2 = -0.2174
K3 = 25.3283
K4 = 13.7714
K5 = -6.4788
K6 = 2.5842
k = 0.0162

s_sea = 35.0        # representative ocean salinity, PSU
ec_sea = 53087.0    # EC (μS/cm) associated with s_sea at 25 °C

# exact base
FT2M = 0.3048  # exact by definition

# derive reciprocals/products to avoid mismatch and rounding drift
M2FT    = 1.0 / FT2M
CFS2CMS = FT2M ** 3           # (ft^3/s) → (m^3/s)
CMS2CFS = 1.0 / CFS2CMS


# vtools/functions/unit_conversions.py


# ---- Aliases & normalization -------------------------------------------------
_ALIASES = {
    # flows (canonical CF style)
    "cfs": "ft^3 s-1",
    "ft3/s": "ft^3 s-1",
    "ft^3/s": "ft^3 s-1",
    "cms": "m^3 s-1",
    "m3/s": "m^3 s-1",
    "m^3/s": "m^3 s-1",

    # temperature (return case-sensitive names Pint expects)
    "deg f": "degF",
    "degree_fahrenheit": "degF",
    "deg c": "degC",
    "degree_celsius": "degC",

    # conductivity spellings
    "us/cm": "uS cm-1",
    "μs/cm": "uS cm-1",
    "microsiemens/cm": "uS cm-1",
    "micromhos/cm": "uS cm-1",
    "us/cm@25c": "uS cm-1",
    "micromhos/cm@25c": "uS cm-1",
}

def _norm(u: str) -> str:
    """Normalize common shorthands to canonical spellings without
    destroying case needed by Pint (e.g., degC/degF)."""
    if u is None:
        return ""
    s = u.strip()
    s = re.sub(r"\s+", " ", s)
    k = s.lower()
    return _ALIASES.get(k, s)

def _rewrap_like(values, arr):
    if isinstance(values, pd.DataFrame):
        return pd.DataFrame(arr, index=values.index, columns=values.columns)
    if isinstance(values, pd.Series):
        return pd.Series(arr, index=values.index, name=values.name)
    return arr

# ---- Optional backend flip via env var (no public API) -----------------------
def _want_cf_units() -> bool:
    return os.environ.get("VTOOLS_UNITS_BACKEND", "").lower() == "cf_units"

@functools.lru_cache(maxsize=128)
def _get_converter(iu: str, ou: str):
    """Return a callable(arr)->arr using Pint by default; cf_units if env-forced."""
    if _want_cf_units():
        try:
            from cf_units import Unit
            u_in, u_out = Unit(iu), Unit(ou)
            def conv(arr): return u_in.convert(np.asarray(arr), u_out)
            return conv
        except Exception:
            # fall through to Pint if cf_units not available
            pass

    import pint
    ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

    q_in, q_out = 1.0 * ureg(iu), 1.0 * ureg(ou)
    # --- Skip fast scaling for temperature units (affine with offsets) ---
    OFFSET_UNITS = {"degC", "degF", "degree_Celsius", "degree_Fahrenheit", "degR"}
    if iu in OFFSET_UNITS or ou in OFFSET_UNITS:
        def conv(arr):
            a = np.asarray(arr)
            return (a * ureg(iu)).to(ureg(ou)).m
        return conv

    # --- Otherwise use fast pure-scale path ---
    try:
        factor = q_in.to(q_out).magnitude
        def conv(arr):
            return np.asarray(arr) * factor
        return conv
    except Exception:
        def conv(arr):
            a = np.asarray(arr)
            return (a * ureg(iu)).to(ureg(ou)).m
        return conv


# ---- Public API --------------------------------------------------------------
def convert_units(values, in_unit: str, out_unit: str):
    """
    Convert array-like / pandas objects between units.
    Fast custom paths for EC↔PSU@25C, temperature, cfs↔cms, ft↔m; else Pint-backed.

    Parameters
    ----------
    values : array-like | pd.Series | pd.DataFrame
    in_unit, out_unit : str
        Unit strings. Shorthands like 'cfs','cms','ft3/s','μS/cm','deg F' accepted.

    Returns
    -------
    Same type as `values`, converted.
    """
    if in_unit is None or out_unit is None:
        raise ValueError("Both in_unit and out_unit must be specified")

    iu, ou = _norm(in_unit), _norm(out_unit)
    if iu == ou:
        return values

    # --- Custom domain-first paths -------------------------------------------
    # Temperature
    if iu == "degf" and ou == "degc":
        arr = (np.asarray(values) - 32.0) * (5.0/9.0)
        return _rewrap_like(values, arr)
    if iu == "degc" and ou == "degf":
        arr = np.asarray(values) * 1.8 + 32.0
        return _rewrap_like(values, arr)


    # Length / Flow shorthands (scale only)
    if iu == "ft" and ou == "m":
        return _rewrap_like(values, np.asarray(values) * FT2M)
    if iu == "m" and ou == "ft":
        return _rewrap_like(values, np.asarray(values) * M2FT)
    if iu == "ft^3 s-1" and ou == "m^3 s-1":
        return _rewrap_like(values, np.asarray(values) * CFS2CMS)
    if iu == "m^3 s-1" and ou == "ft^3 s-1":
        return _rewrap_like(values, np.asarray(values) * CMS2CFS)

    # EC ↔ PSU at 25C (never hand 'psu' to a generic backend)
    if iu in ("ec", "us/cm", "uS cm-1", "micromhos/cm") and ou == "psu":
        return ec_psu_25c(values, hill_correction=True)   # uses your existing impl
    if iu == "psu" and ou in ("ec", "us/cm", "uS cm-1", "micromhos/cm"):
        out = psu_ec_25c(values, refine=True, hill_correction=True)
        return out

    # --- Backend fallback (Pint by default; cf_units if env forces it) -------
    conv = _get_converter(iu, ou)
    base = values.values if isinstance(values, (pd.Series, pd.DataFrame)) else values
    out = conv(base)
    return _rewrap_like(values, out)




# -----------------------------------------------------------------------------
# Linear / affine engineering conversions (functional)
# -----------------------------------------------------------------------------
def m_to_ft(x):
    """
    Convert metres to feet.

    Parameters
    ----------
    x : array-like or scalar
        Value(s) in metres.

    Returns
    -------
    ndarray or scalar
        Value(s) in feet.

    Notes
    -----
    1 m = 3.28084 ft.
    """
    arr = np.asarray(x)
    out = arr * M2FT
    return out.item() if np.isscalar(x) else out


def ft_to_m(x):
    """
    Convert feet to metres.

    Parameters
    ----------
    x : array-like or scalar
        Value(s) in feet.

    Returns
    -------
    ndarray or scalar
        Value(s) in metres.

    Notes
    -----
    1 ft = 0.3048 m.
    """
    arr = np.asarray(x)
    out = arr * FT2M
    return out.item() if np.isscalar(x) else out


def cms_to_cfs(x):
    """
    Convert cubic metres per second (cms) to cubic feet per second (cfs).

    Parameters
    ----------
    x : array-like or scalar
        Flow in m³/s.

    Returns
    -------
    ndarray or scalar
        Flow in ft³/s.

    Notes
    -----
    1 m³/s = 35.31466621 ft³/s.
    """
    arr = np.asarray(x)
    out = arr * CMS2CFS
    return out.item() if np.isscalar(x) else out


def cfs_to_cms(x):
    """
    Convert cubic feet per second (cfs) to cubic metres per second (cms).

    Parameters
    ----------
    x : array-like or scalar
        Flow in ft³/s.

    Returns
    -------
    ndarray or scalar
        Flow in m³/s.

    Notes
    -----
    1 ft³/s = 0.028316847 m³/s.
    """
    arr = np.asarray(x)
    out = arr * CFS2CMS
    return out.item() if np.isscalar(x) else out


def fahrenheit_to_celsius(x):
    """
    Convert degrees Fahrenheit to degrees Celsius.

    Parameters
    ----------
    x : array-like or scalar
        Temperature in °F.

    Returns
    -------
    ndarray or scalar
        Temperature in °C.

    Notes
    -----
    °C = (°F − 32) × 5/9.
    """
    arr = np.asarray(x)
    out = (arr - 32.0) * (5.0 / 9.0)
    return out.item() if np.isscalar(x) else out


def celsius_to_fahrenheit(x):
    """
    Convert degrees Celsius to degrees Fahrenheit.

    Parameters
    ----------
    x : array-like or scalar
        Temperature in °C.

    Returns
    -------
    ndarray or scalar
        Temperature in °F.

    Notes
    -----
    °F = (°C × 1.8) + 32.
    """
    arr = np.asarray(x)
    out = arr * 1.8 + 32.0
    return out.item() if np.isscalar(x) else out


# -----------------------------------------------------------------------------
# EC (μS/cm) ↔ PSU at 25 °C
# -----------------------------------------------------------------------------
def psu_ec_resid(x, psu, hill_correction):
    """
    Residual function for root finding in PSU→EC conversion.

    Parameters
    ----------
    x : float
        Candidate EC (μS/cm) value.
    psu : float
        Target practical salinity (PSU).
    hill_correction : bool
        If True, apply Hill low-salinity correction in EC→PSU mapping.

    Returns
    -------
    float
        `ec_psu_25c(x) - psu` evaluated at `x`.
    """
    return ec_psu_25c(x, hill_correction) - psu


def ec_psu_25c(ec, hill_correction=True):
    """
    Convert electrical conductivity (EC, μS/cm) to practical salinity (PSU) at 25 °C.

    This implements the empirical relationship used for estuarine work, with an
    optional Hill correction that improves behavior at low salinities.

    Parameters
    ----------
    ec : array-like or scalar
        Electrical conductivity in μS/cm.
    hill_correction : bool, default True
        Apply Hill low-salinity correction.

    Returns
    -------
    ndarray or scalar
        Practical salinity (PSU). For negative EC inputs:
        - scalar input → returns NaN
        - array input → returns NaN at those positions

    Notes
    -----
    * Assumes temperature is 25 °C.
    * Negative EC values are internally floored to a small positive ratio
      for computation (`R=1e-4`); those outputs are then set to NaN on
      array paths (or NaN returned for scalar paths).
    """
    arr = np.asarray(ec)
    R = arr / ec_sea
    neg_mask = R < 0.0

    # Handle negative inputs consistently with legacy behavior
    if np.isscalar(ec):
        if neg_mask:
            return np.nan
    else:
        R = R.copy()
        R[neg_mask] = 1.0e-4  # small positive floor for computation

    sqrtR = np.sqrt(R)
    Rsq = R * R
    s = K1 + K2 * sqrtR + K3 * R + K4 * R * sqrtR + K5 * Rsq + K6 * Rsq * sqrtR

    if hill_correction:
        y = 100.0 * R
        x = 400.0 * R
        a_0 = 0.008
        f_ = (25.0 - 15.0) / (1.0 + k * (25.0 - 15.0))  # f(T=25)
        b_0_f = 0.0005 * f_
        s = s - a_0 / (1.0 + 1.5 * x + x * x) - b_0_f / (1.0 + np.sqrt(y) + y + y * np.sqrt(y))

    if np.isscalar(ec):
        return float(s) if not np.isnan(s) else s
    out = np.asarray(s, dtype=float)
    if np.any(neg_mask):
        out[neg_mask] = np.nan
    return out


def psu_ec_25c_scalar(psu, refine=True, hill_correction=True):
    """
    Convert practical salinity (PSU) to EC (μS/cm) at 25 °C for a **scalar** value.

    Parameters
    ----------
    psu : float
        Practical salinity. Must be non-negative and ≤ ~35 for oceanic cases
        (a hard check is enforced near sea salinity when `refine` is True).
    refine : bool, default True
        If True, use a scalar root finder (Brent) to invert the EC→PSU mapping
        accurately. If False, use a closed-form Schemel-style polynomial approximation.
    hill_correction : bool, default True
        Only meaningful with `refine=True`; raises if `refine=False` and
        `hill_correction=True`.

    Returns
    -------
    float
        Electrical conductivity (μS/cm).

    Raises
    ------
    ValueError
        If `psu` < 0, if `psu` exceeds the sea-salinity cap in `refine` mode,
        or if an invalid combination of `refine`/`hill_correction` is requested.

    Notes
    -----
    * The refinement typically converges in ~4–6 iterations.
    * The non-refined polynomial is faster but can drift on round trips (EC→PSU→EC).
    """
    if psu < 0.0:
        raise ValueError(f"Negative psu not allowed: {psu}")
    if np.isnan(psu):
        return np.nan

    if hill_correction and not refine:
        raise ValueError("Unrefined (refine=False) psu-to-ec correction cannot have hill_correction")

    if refine:
        if psu > 34.99969:
            raise ValueError(f"psu is over sea salinity: {psu}")
        ec = brentq(psu_ec_resid, 1.0, ec_sea, args=(psu, hill_correction))
    else:
        sqrtpsu = np.sqrt(psu)
        ec = (psu / s_sea) * ec_sea + psu * (psu - s_sea) * (J1 + J2 * sqrtpsu + J3 * psu + J4 * sqrtpsu * psu)
    return ec


psu_ec_25c_vec = np.vectorize(psu_ec_25c_scalar, otypes="d", excluded=["refine", "hill_correction"])


def psu_ec_25c(psu, refine=True, hill_correction=True):
    """
    Convert practical salinity (PSU) to EC (μS/cm) at 25 °C (vectorized).

    Parameters
    ----------
    psu : array-like or scalar
        Practical salinity value(s).
    refine : bool, default True
        Use root finding via :func:`psu_ec_25c_scalar` for accuracy.
    hill_correction : bool, default True
        See :func:`psu_ec_25c_scalar`.

    Returns
    -------
    ndarray or scalar
        EC in μS/cm. Scalar input returns a scalar; array-like input returns
        a NumPy array of the same shape.
    """
    if np.isscalar(psu):
        return psu_ec_25c_scalar(psu, refine, hill_correction)
    arr = np.asarray(psu)
    return psu_ec_25c_vec(arr, refine, hill_correction)
