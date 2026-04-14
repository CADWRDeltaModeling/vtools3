import numpy as np
from utide.harmonics import FUV
from utide._ut_constants import constit_index_dict
from utide._time_conversion import _normalize_time

import pandas as pd

__all__=["nodal_factors"]


def nodal_factors(
    t_dates,
    tref_date,
    constituents,
    lat_deg,
    nodal_linear_time=False,
    phase="Greenwich",
):
    """
    Compute nodal factors F, nodal phase U (cycles),
    and astronomical argument V (cycles).

    Parameters
    ----------
    t_dates : array-like pandas.Timestamp or array-like of datetime
        Time in datetime.
    tref_date : datetime or pandas.Timestamp
        Reference date, used for nodal amp and phase correction calculations.
        Resulted nodal factors will be computed at this reference date if
        nodal_linear_time=True, otherwise nodal factors will be computed at t_dates.
    constituents : sequence[str]
        Constituent abbreviations, e.g. ["M2", "K1", "MK3", "MO3"].
    lat_deg : float
        Latitude in degrees.
    nodal_linear_time : bool
        Linearized nodal variation, check with tref_date for details.
    phase : {"Greenwich", "linear_time", "raw"}
        Tide Greenwich equilibrium phase correction convention.
        Greenwich: Greenwich equilibrium argument computed on all t_dates
        linear_time: single Greenwich equilibrium argument computed at tref_date (UT) and applied to all t_dates
        raw: no Greenwich equilibrium argument applied.

    Returns
    -------
    F : ndarray (nt, nc)
        Nodal amplitude factors.
    U : ndarray (nt, nc)
        Nodal phase corrections (cycles).
    V : ndarray (nt, nc)
        Astronomical arguments (cycles).
    """
    ## check input t_dates and tref_date are pandas.Timestamp or datetime
    if not (isinstance(t_dates, pd.Series) or isinstance(t_dates, pd.DatetimeIndex) or isinstance(t_dates, pd.Timestamp)):
        raise ValueError("t_dates must be a pandas Series, DatetimeIndex, or Timestamp, or a datetime object")
    if not (isinstance(tref_date, pd.Timestamp) ):
        raise ValueError("tref_date must be a pandas Timestamp or datetime object")
    # Convert t_dates and tref_date to days since epoch
    t_days = _normalize_time(t_dates)
    tref_days = _normalize_time(tref_date)

    nodal = True

    return _fuv(
        t_days,
        tref_days,
        constituents,
        lat_deg,
        nodal=nodal,
        nodal_linear_time=nodal_linear_time,
        phase=phase
    )


def _fuv(
    t_days,
    tref_days,
    constituents,
    lat_deg,
    *,
    nodal=True,
    nodal_linear_time=False,
    phase="Greenwich",
):
    """
    Compute nodal factors F, nodal phase U (cycles),
    and astronomical argument V (cycles).

    Parameters
    ----------
    t_days : array-like
        Time in days since epoch.
    tref_days : float
        Reference time in days since epoch.
    constituents : sequence[str]
        Constituent abbreviations, e.g. ["M2", "K1", "MK3", "MO3"].
    lat_deg : float
        Latitude in degrees.
    nodal : bool
        Apply nodal/satellite corrections.
    nodal_linear_time : bool
        Linearized nodal variation.
    phase : {"Greenwich", "linear_time", "raw"}
        Phase convention.

    Returns
    -------
    F : ndarray (nt, nc)
        Nodal amplitude factors.
    U : ndarray (nt, nc)
        Nodal phase corrections (cycles).
    V : ndarray (nt, nc)
        Astronomical arguments (cycles).
    """

    if phase not in {"Greenwich", "linear_time", "raw"}:
        raise ValueError(f"Invalid phase={phase!r}")

    try:
        lind = np.array([constit_index_dict[c] for c in constituents], dtype=int)
    except KeyError as e:
        raise KeyError(f"Unknown tide constituent {e.args[0]!r}") from None

    ngflgs = np.zeros(4, dtype=bool)

    if not nodal:
        ngflgs[1] = True
    elif nodal_linear_time:
        ngflgs[0] = True

    if phase == "raw":
        ngflgs[3] = True
    elif phase == "linear_time":
        ngflgs[2] = True

    t_days = np.asarray(t_days, dtype=float)

    F, U, V = FUV(
        t_days,
        float(tref_days),
        lind,
        float(lat_deg),
        ngflgs,
    )

    return F, U, V
