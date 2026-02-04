"""Module contains filter used in tidal time series analysis."""

## Python libary import.
from numpy import abs
import pandas as pd
import numpy as np
from vtools.data.vtime import seconds, minutes, hours
from scipy.signal import lfilter, firwin, filtfilt
from scipy.signal import butter
from scipy.ndimage import gaussian_filter1d

__all__ = [
    "cosine_lanczos",
    "butterworth",
    "godin",
    "cosine_lanczos5",
    "lowpass_cosine_lanczos_filter_coef",
    "ts_gaussian_filter",
    "lowpass_lanczos_filter_coef",
    "lanczos",
]


_cached_filt_info = {}


def _resolve_filter_len(filter_len, freq):
    if filter_len is None:
        return None
    if isinstance(filter_len, (int, np.integer)):
        return int(filter_len)

    try:
        # Convert freq to a timedelta
        if hasattr(freq, 'delta') and freq.delta is not None:
            freq_delta = freq.delta
        elif hasattr(freq, 'nanos'):
            freq_delta = pd.Timedelta(freq.nanos, unit='ns')
        else:
            freq_offset = pd.tseries.frequencies.to_offset(freq)
            if hasattr(freq_offset, "delta") and freq_offset.delta is not None:
                freq_delta = freq_offset.delta
            elif hasattr(freq_offset, 'nanos'):
                freq_delta = pd.Timedelta(freq_offset.nanos, unit='ns')
            else:
                raise TypeError("Time series frequency is not fixed")

        # Convert filter_len to a timedelta
        if hasattr(filter_len, 'delta') and filter_len.delta is not None:
            fl_delta = filter_len.delta
        elif hasattr(filter_len, 'nanos'):
            fl_delta = pd.Timedelta(filter_len.nanos, unit='ns')
        else:
            fl_offset = pd.tseries.frequencies.to_offset(filter_len)
            if hasattr(fl_offset, "delta") and fl_offset.delta is not None:
                fl_delta = fl_offset.delta
            elif hasattr(fl_offset, 'nanos'):
                fl_delta = pd.Timedelta(fl_offset.nanos, unit='ns')
            else:
                fl_delta = pd.to_timedelta(filter_len)

        if fl_delta % freq_delta != pd.Timedelta(0):
            raise TypeError("filter_len was not divisible by the time step")
        return int(fl_delta / freq_delta)
    except TypeError:
        raise
    except Exception as e:
        raise TypeError(
            f"filter_len was not an int or divisible by the time step: {e}"
        )


###########################################################################
## Public interface.
###########################################################################


def process_cutoff(cutoff_frequency, cutoff_period, freq):
    if cutoff_frequency is None:
        if cutoff_period is None:
            raise ValueError("One of cutoff_frequency or cutoff_period must be given")
        cp = pd.tseries.frequencies.to_offset(cutoff_period)
        return 2.0 * freq / cp
    else:
        if cutoff_frequency < 0 or cutoff_frequency > 1.0:
            raise ValueError("cutoff frequency must be 0 < cf < 1)")
        return cutoff_frequency


def cosine_lanczos5(
    ts,
    cutoff_period=None,
    cutoff_frequency=None,
    filter_len=None,
    padtype=None,
    padlen=None,
    fill_edge_nan=True,
):
    """squared low-pass cosine lanczos  filter on a regular time series.


    Parameters
    ----------

    ts : :class:`DataFrame <pandas:pandas.DataFrame>`

    filter_len  : int, time_interval
        Size of lanczos window, default is to number of samples within filter_period*1.25.

    cutoff_frequency: float,optional
        Cutoff frequency expressed as a ratio of a Nyquist frequency,
        should within the range (0,1). For example, if the sampling frequency
        is 1 hour, the Nyquist frequency is 1 sample/2 hours. If we want a
        36 hour cutoff period, the frequency is 1/36 or 0.0278 cycles per hour.
        Hence the cutoff frequency argument used here would be
        0.0278/0.5 = 0.056.

    cutoff_period : string  or  _time_interval
         Period of cutting off frequency. If input as a string, it must
         be  convertible to a _time_interval (Pandas freq).
         cutoff_frequency and cutoff_period can't be specified at the same time.

     padtype : str or None, optional
         Must be 'odd', 'even', 'constant', or None. This determines the type
         of extension to use for the padded signal to which the filter is applied.
         If padtype is None, no padding is used. The default is None.

     padlen : int or None, optional
          The number of elements by which to extend x at both ends of axis
          before applying the filter. This value must be less than x.shape[axis]-1.
          padlen=0 implies no padding. If padtye is not None and padlen is not
          given, padlen is be set to 6*m.

     fill_edge_nan: bool,optional
          If pading is not used and fill_edge_nan is true, resulting data on
          the both ends are filled with nan to account for edge effect. This is
          2*m on the either end of the result. Default is true.

    Returns
    -------
    result : :class:`~vtools.data.timeseries.TimeSeries`
        A new regular time series with the same interval of ts. If no pading
        is used the beigning and ending 4*m resulting data will be set to nan
        to remove edge effect.

    Raises
    ------
    ValueError
        If input timeseries is not regular,
        or, cutoff_period and cutoff_frequency are given at the same time,
        or, neither cutoff_period nor curoff_frequence is given,
        or, padtype is not "odd","even","constant",or None,
        or, padlen is larger than data size

    """

    freq = ts.index.freq
    if freq is None:
        raise ValueError("Time series has no frequency attribute")

    m = filter_len

    cf = process_cutoff(cutoff_frequency, cutoff_period, freq)

    if m is None:
        m = int(1.25 * 2.0 / cf)
    else:
        m = _resolve_filter_len(m, freq)

    ##find out nan location and fill with 0.0. This way we can use the
    ## signal processing filtrations out-of-the box without nans causing trouble,
    ## but we have to post process the areas touched by nan
    idx = np.where(np.isnan(ts.values))[0]
    data = np.array(ts.values).copy()

    ## figure out indexes that will be nan after the filtration,which
    ## will "grow" the nan region around the original nan by 2*m
    ## slots in each direction
    if False:
        # len(idx)>0:
        data[idx] = 0.0
        shifts = np.arange(-2 * m, 2 * m + 1)
        result_nan_idx = np.clip(np.add.outer(shifts, idx), 0, len(ts) - 1).ravel()

    if m < 1:
        raise ValueError("bad input cutoff period or frequency")

    if padtype is not None:
        if not padtype in ["odd", "even", "constant"]:
            raise ValueError("unkown padtype :" + padtype)

    if (padlen is None) and (padtype is not None):
        padlen = 6 * m

    if padlen is not None:  # is None sensible?
        if padlen > len(data):
            raise ValueError("Padding length is more  than data size")

    ## get filter coefficients. sizeo of coefis is 2*m+1 in fact
    coefs = lowpass_cosine_lanczos_filter_coef(cf, int(m))

    d2 = filtfilt(coefs, [1.0], data, axis=0, padtype=padtype, padlen=padlen)

    # if(len(idx)>0):
    #    d2[result_nan_idx]=np.nan

    ## replace edge points with nan if pading is not used

    if (padtype is None) and (fill_edge_nan == True):
        d2[0 : 2 * m, np.newaxis] = np.nan
        d2[len(d2) - 2 * m : len(d2), np.newaxis] = np.nan

    out = ts.copy(deep=True)
    out[:] = d2

    return out


def lowpass_cosine_lanczos_filter_coef(cf, m, normalize=True):
    """return the convolution coefficients for low pass lanczos filter.

    Parameters
    ----------

    cf: float
      Cutoff frequency expressed as a ratio of a Nyquist frequency.

    m: int
      Size of filtering window size.

    Returns
    -------
    results: list
           Coefficients of filtering window.

    """
    if (cf, m) in _cached_filt_info:
        return _cached_filt_info[(cf, m)]
    coscoef = [
        cf * np.sin(np.pi * k * cf) / (np.pi * k * cf)
        for k in np.arange(1, m + 1, 1, dtype="d")
    ]
    sigma = [
        np.sin(np.pi * k / m) / (np.pi * k / m)
        for k in np.arange(1, m + 1, 1, dtype="float")
    ]
    prod = [c * s for c, s in zip(coscoef, sigma)]
    temp = prod[-1::-1] + [cf] + prod
    res = np.array(temp)
    if normalize:
        res = res / res.sum()
    _cached_filt_info[(cf, m)] = res
    return res


def cosine_lanczos(
    ts,
    cutoff_period=None,
    cutoff_frequency=None,
    filter_len=None,
    padtype=None,
    padlen=None,
    fill_edge_nan=True,
):
    return _lanczos_impl(
        ts,
        cutoff_period,
        cutoff_frequency,
        filter_len,
        padtype,
        padlen,
        fill_edge_nan,
        cosine_taper=True,
    )


def lanczos(
    ts,
    cutoff_period=None,
    cutoff_frequency=None,
    filter_len=None,
    padtype=None,
    padlen=None,
    fill_edge_nan=True,
):
    return _lanczos_impl(
        ts,
        cutoff_period,
        cutoff_frequency,
        filter_len,
        padtype,
        padlen,
        fill_edge_nan,
        cosine_taper=False,
    )


def _lanczos_impl(
    ts,
    cutoff_period=None,
    cutoff_frequency=None,
    filter_len=None,
    padtype=None,
    padlen=None,
    fill_edge_nan=True,
    cosine_taper=False,
):
    """squared low-pass cosine lanczos  filter on a regular time series.


    Parameters
    ----------

    ts : :class:`DataFrame <pandas:pandas.DataFrame>`

    filter_len  : int, time_interval
        Size of lanczos window, default is to number of samples within filter_period*1.25.

    cutoff_frequency: float,optional
        Cutoff frequency expressed as a ratio of a Nyquist frequency,
        should within the range (0,1). For example, if the sampling frequency
        is 1 hour, the Nyquist frequency is 1 sample/2 hours. If we want a
        36 hour cutoff period, the frequency is 1/36 or 0.0278 cycles per hour.
        Hence the cutoff frequency argument used here would be
        0.0278/0.5 = 0.056.

    cutoff_period : string  or  _time_interval
         Period of cutting off frequency. If input as a string, it must
         be  convertible to a _time_interval (Pandas freq).
         cutoff_frequency and cutoff_period can't be specified at the same time.

     padtype : str or None, optional
         Must be 'odd', 'even', 'constant', or None. This determines the type
         of extension to use for the padded signal to which the filter is applied.
         If padtype is None, no padding is used. The default is None.

     padlen : int or None, optional
          The number of elements by which to extend x at both ends of axis
          before applying the filter. This value must be less than x.shape[axis]-1.
          padlen=0 implies no padding. If padtye is not None and padlen is not
          given, padlen is be set to 6*m.

     fill_edge_nan: bool,optional
          If pading is not used and fill_edge_nan is true, resulting data on
          the both ends are filled with nan to account for edge effect. This is
          2*m on the either end of the result. Default is true.

    Returns
    -------
    result : :class:`~vtools.data.timeseries.TimeSeries`
        A new regular time series with the same interval of ts. If no pading
        is used the beigning and ending 4*m resulting data will be set to nan
        to remove edge effect.

    Raises
    ------
    ValueError
        If input timeseries is not regular,
        or, cutoff_period and cutoff_frequency are given at the same time,
        or, neither cutoff_period nor curoff_frequence is given,
        or, padtype is not "odd","even","constant",or None,
        or, padlen is larger than data size

    """

    freq = ts.index.freq
    if freq is None:
        raise ValueError("Time series has no frequency attribute")

    m = filter_len

    cf = process_cutoff(cutoff_frequency, cutoff_period, freq)

    if m is None:
        m = int(1.25 * 2.0 / cf)
    else:
        m = _resolve_filter_len(m, freq)

    ##find out nan location and fill with 0.0. This way we can use the
    ## signal processing filtrations out-of-the box without nans causing trouble,
    ## but we have to post process the areas touched by nan
    idx = np.where(np.isnan(ts.values))[0]
    data = np.array(ts.values).copy()

    ## figure out indexes that will be nan after the filtration,which
    ## will "grow" the nan region around the original nan by 2*m
    ## slots in each direction
    if False:
        # len(idx)>0:
        data[idx] = 0.0
        shifts = np.arange(-2 * m, 2 * m + 1)
        result_nan_idx = np.clip(np.add.outer(shifts, idx), 0, len(ts) - 1).ravel()

    if m < 1:
        raise ValueError("bad input cutoff period or frequency")

    if padtype is not None:
        if not padtype in ["odd", "even", "constant"]:
            raise ValueError("unkown padtype :" + padtype)

    if (padlen is None) and (padtype is not None):
        padlen = 6 * m

    if padlen is not None:  # is None sensible?
        if padlen > len(data):
            raise ValueError("Padding length is more  than data size")

    ## get filter coefficients. sizeo of coefis is 2*m+1 in fact
    coefs = lowpass_lanczos_filter_coef(cf, int(m), cosine_taper=cosine_taper)

    d2 = filtfilt(coefs, [1.0], data, axis=0, padtype=padtype, padlen=padlen)

    # if(len(idx)>0):
    #    d2[result_nan_idx]=np.nan

    ## replace edge points with nan if pading is not used

    if (padtype is None) and (fill_edge_nan == True):
        d2[0 : 2 * m, np.newaxis] = np.nan
        d2[len(d2) - 2 * m : len(d2), np.newaxis] = np.nan

    out = ts.copy(deep=True)
    out[:] = d2

    return out


def lowpass_lanczos_filter_coef(cf, m, normalize=True, cosine_taper=False):
    """Return the convolution coefficients for a low-pass Lanczos filter.

    Parameters
    ----------
    cf : float
        Cutoff frequency expressed as a ratio of the Nyquist frequency.
    m : int
        Size of the filtering window.
    normalize : bool, optional
        Whether to normalize the filter coefficients so they sum to 1.
    cosine_taper : bool, optional
        If True, applies a cosine-squared taper to the Lanczos window.

    Returns
    -------
    res : np.ndarray
        Coefficients of the filtering window.
    """
    if (cf, m, cosine_taper) in _cached_filt_info:
        return _cached_filt_info[(cf, m, cosine_taper)]

    k = np.arange(1, m + 1, dtype="d")
    ideal_sinc = cf * np.sinc(k * cf)
    lanczos_window = np.sinc(k / m)

    if cosine_taper:
        cos_taper = np.cos(np.pi * k / (2 * m)) ** 2
        prod = ideal_sinc * lanczos_window * cos_taper
    else:
        prod = ideal_sinc * lanczos_window

    temp = np.concatenate((prod[::-1], [cf], prod))  # Centered at k = 0
    res = np.array(temp)

    if normalize:
        res /= res.sum()

    _cached_filt_info[(cf, m, cosine_taper)] = res
    return res


def butterworth(ts, cutoff_period=None, cutoff_frequency=None, order=4):
    """low-pass butterworth-squared filter on a regular time series.


    Parameters
    ----------


    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
        Must be one or two dimensional, and regular.

    order: int ,optional
        The default is 4.

    cutoff_frequency: float,optional
        Cutoff frequency expressed as a ratio with Nyquist frequency,
        should within the range (0,1). For a discretely sampled system,
        the Nyquist frequency is the fastest frequency that can be resolved by that
        sampling, which is half the sampling frequency. For example, if the sampling frequency
        is 1 sample/1 hour, the Nyquist frequency is 1 sample/2 hours. If we want a
        36 hour cutoff period, the frequency is 1/36 or 0.0278 cycles per hour.
        Hence the cutoff frequency argument used here would be
        0.0278/0.5 = 0.056.

    cutoff_period : string  or  _time_interval
         Period corresponding to cutoff frequency. If input as a string, it must
         be  convertible to a regular interval using the same rules as a pandas frequency..
         cutoff_frequency and cutoff_period can't be specified at the same time.

    Returns
    -------
    result :
        A new regular time series with the same interval as ts.

    Raises
    ------
    ValueError
        If input order is not even, or input timeseries is not regular,
        or neither cutoff_period and cutoff_frequency is given while input
        time series interval is not 15min or 1 hour, or  cutoff_period and cutoff_frequency
        are given at the same time.

    """

    if order % 2:
        raise ValueError("only even order is accepted")

    # if not ts.is_regular():
    #    raise ValueError("Only regular time series can be filtered.")

    freq = ts.index.freq

    #    if (not (interval in _butterworth_interval)) and (cutoff_period is None) and (cutoff_frequency is None):
    #        raise ValueError("time interval is not supported by butterworth if no cuttoff period/frequency given.")

    if (not (cutoff_frequency is None)) and (not (cutoff_period is None)):
        raise ValueError(
            "cutoff_frequency and cutoff_period can't be specified simultaneously"
        )

    if (cutoff_frequency is None) and (cutoff_period is None):
        raise ValueError("Either cutoff_frequency or cutoff_period must be given")

    cf = cutoff_frequency

    if cf is None:
        if not (cutoff_period is None):
            cutoff_period = pd.tseries.frequencies.to_offset(cutoff_period)
            cf = 2.0 * freq / cutoff_period
        else:
            cf = butterworth_cutoff_frequencies[interval]

    ## get butter filter coefficients.
    [b, a] = butter(order / 2, cf)
    d2 = filtfilt(b, a, ts.values, axis=0, padlen=90)
    out = ts.copy(deep=True)
    out[:] = d2

    #    prop={}
    #    for key,val in ts.props.items():
    #        prop[key]=val
    #    prop[TIMESTAMP]=INST
    #    prop[AGGREGATION]=INDIVIDUAL
    #    time_interval
    return out


def generate_godin_fir(freq):
    """
    generate godin filter impulse response for given freq
    freq is a pandas freq
    """
    freqstr = str(freq)
    if freqstr in _cached_filt_info:
        return _cached_filt_info[freqstr]
    dt_sec = int(freq / seconds(1))
    nsample24 = int(86400 // dt_sec)  # 24 hours by dt (24 for hour, 96 for 15min)
    wts24 = np.zeros(nsample24, dtype="d")
    wts24[:] = 1.0 / nsample24
    nsample25 = (1490 * 60) // dt_sec  # 24 hr 50min in seconds by dt
    if nsample25 % 2 == 0:
        # ensure odd
        nsample25 += 1
    wts25 = np.zeros(nsample25, dtype="d")
    wts25[:] = 1.0 / nsample25
    wts24 = np.zeros(nsample24, dtype="d")
    wts24[:] = 1.0 / nsample24
    v = np.convolve(wts25, np.convolve(wts24, wts24))
    _cached_filt_info[freqstr] = v
    return v


def godin(ts):
    r"""Low-pass Godin filter a regular time series.
    Applies the :math:`\mathcal{A_{24}^{2}A_{25}}` Godin filter [1]_
    The filter is generalized to be the equivalent of one
    boxcar of the length of the lunar diurnal (~25 hours)
    constituent and two of the solar diurnal (~24 hours), though the
    implementation combines these steps.


    Parameters
    ----------

    ts : :class:`DataFrame <pandas:pandas.DataFrame>`

    Returns
    -------
    result : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new regular time series with the same interval of ts.

    Raises
    ------
    NotImplementedError
        If input time series is not univariate

    References
    ----------
    .. [1] Godin (1972) Analysis of Tides

    """
    freq = ts.index.freq
    if freq is None:
        raise ValueException("Series must be regular (have freq set)")
    godin_ir = generate_godin_fir(freq)
    nfilt = len(godin_ir)
    nhalffilt = nfilt // 2
    # if not (len(ts.columns) == 1):
    #    raise NotImplementedError("Godin Filter not functional for multivariate series yet")
    if hasattr(ts, "columns"):
        dfg = ts.apply(np.convolve, axis=0, v=godin_ir, mode="same")
        dfg.columns = ts.columns
        dfg.iloc[0:nhalffilt, :] = np.nan
        dfg.iloc[-nhalffilt:, :] = np.nan
    else:  # Assume series
        outdata = np.convolve(ts.to_numpy(), v=godin_ir, mode="same")
        dfg = pd.DataFrame(data=outdata, index=ts.index)
    return dfg


def convert_span_to_nstep(freq, span):
    if type(span) == int:
        return span
    span = pd.tseries.frequencies.to_offset(span)
    freq = pd.tseries.frequencies.to_offset(freq)
    return int(span / freq)


def _gf1d(ts, sigma, order, mode, cval, truncate):
    tscopy = ts.copy()
    tscopy.loc[:] = gaussian_filter1d(
        ts.squeeze().to_numpy(),
        sigma=sigma,
        order=order,
        mode=mode,
        cval=cval,
        truncate=truncate,
    )
    return tscopy


def ts_gaussian_filter(ts, sigma, order=0, mode="reflect", cval=0.0, truncate=4.0):
    """Column-wise Gaussian smoothing of regular time series.
    Missing/irregular values are not handled, which means this function is not much different from
    a rolling window gaussian average in pandas which may be preferable in the case of
    missing data (ts.rolling(window=5,win_type='gaussian').mean.
    This function has been kept around awaiting irreg as an aspiration but yet to be implemented.

    Parameters
    ----------

    ts : :class:`DataFrame <pandas:pandas.DataFrame>`
        The series to be smoothed

    sigma : int or freq
        The sigma scale of the smoothing (analogous to std. deviation), given as a number of steps
        or freq


    Returns
    -------
    result : :class:`DataFrame <pandas:pandas.DataFrame>`
        A new regular time series with the same interval of ts.

    """
    freq = ts.index.freq
    if type(sigma) != int:
        sigma = convert_span_to_nstep(freq, sigma)

    if isinstance(ts, pd.Series):
        tsout = _gf1d(
            ts, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate
        )
    else:
        tsout = ts.apply(
            _gf1d, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate
        )
    return tsout
