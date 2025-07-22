import vtools
from vtools.data.timeseries import *
from scipy.stats import trim_mean
import numpy as np


def mse(predictions, targets):
    """Mean squared error

    Parameters
    ----------
    predictions, targets : array_like
        Time series or arrays to analyze

    Returns
    -------
    mse : vtools.data.timeseries.TimeSeries
        Mean squared error between predictions and targets
    """
    return ((predictions - targets) ** 2.0).mean()


def rmse(predictions, targets):
    """Root mean squared error

    Parameters
    ----------
    predictions, targets : array_like
        Time series or arrays to analyze

    Returns
    -------
    mse : float
        Mean squared error
    """
    return ((predictions - targets) ** 2).mean() ** 0.5


def median_error(predictions, targets):
    """Calculate the median error, discounting nan values

    Parameters
    ----------
    predictions, targets : array_like

        Time series or arrays to be analyzed

    Returns
    -------
    med : float
        Median error
    """
    return (predictions - targets).median()


def mean_error(predictions, targets, proportiontocut):
    """Calculate the untrimmed mean error, discounting nan values

    Parameters
    ----------
    predictions, targets : array_like

        Time series or arrays to be analyzed

    Returns
    -------
    med : float
        Median error
    """
    trim = (predictions - targets).mean()
    return trim


def skill_score(predictions, targets, ref=None):
    """Calculate a Nash-Sutcliffe-like skill score based on mean squared error

    As per the discussion in Murphy (1988) other reference forecasts (climatology,
    harmonic tide, etc.) are possible.

    Parameters
    ----------
    predictions, targets : array_like
        Time series or arrays to be analyzed

    Returns
    -------
    rmse : float
        Root mean squared error
    """
    if not ref:
        ref = targets.mean()

    return 1.0 - (mse(predictions, targets) / mse(ref, targets))


def willmott_score(predictions, targets, ref=None):
    """Calculate a Nash-Sutcliffe-like skill score based on mean squared error

    As per the discussion in Murphy (1988) other reference forecasts (climatology,
    harmonic tide, etc.) are possible.

    Parameters
    ----------
    predictions, targets : array_like
        Time series or arrays to be analyzed

    Returns
    -------
    rmse : float
        Root mean squared error
    """
    if not ref:
        ref = targets.mean()
    n = len(predictions)
    a = mse(predictions, targets)
    b = mse(ref, targets)
    c = mse(ref, predictions)
    e = np.abs(mean_error(predictions, ref, None))
    f = np.abs(mean_error(targets, ref, None))

    score = 1.0 - a / (b + c + 2 * e * f * n)
    return score


def tmean_error(predictions, targets, limits=None, inclusive=[True, True]):
    """Calculate the (possibly trimmed) mean error, discounting nan values

    Parameters
    ----------
    predictions, targets : array_like
        Time series or arrays to be analyzed

    limits : tuple(float)
        Low and high limits for trimming

    inclusive : [boolean, boolean]
        True if clipping is inclusive on the low/high end

    Returns
    -------
    mean : float
        Trimmed mean error
    """
    import scipy

    y = np.ma.masked_invalid(predictions)
    z = np.ma.masked_invalid(targets)
    return scipy.stats.mstats.tmean(y - z, limits)


def corr_coefficient(predictions, targets, method="pearson"):
    """Calculates the correlation coefficient (the 'r' in '-squared' between two series.

    For time series where the targets are serially correlated and may span only a fraction
    of the natural variability, this statistic may not be appropriate and Murphy (1988) explains
    why caution should be exercised in using this statistic.

    Parameters
    ----------
    predictions, targets : array_like
        Time series to analyze

    method : pearson’, ‘kendall’, ‘spearman’
        Method compatilble with pandasa

    Returns
    -------
    r : float
        Correlation coefficient
    """

    return predictions.corr(targets, method)


def _main():
    from statsmodels.tsa.arima_process import arma_generate_sample
    import matplotlib.pyplot as plt

    # from vtools.data.sample_series import arma
    start = pd.Timestamp(2009, 3, 12)
    intvl = minutes(15)
    lag = minutes(37)
    index = pd.date_range(start=start, freq=intvl, periods=4000)
    x = np.linspace(0, 500.0, 4000)
    arparams = np.array([0.975])
    maparams = np.array([0.5, 0.25, 0.25])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    noise = arma_generate_sample(ar, ma, nsample=4000)
    base = np.cos(x * 2.0 * np.pi / 12.4) + 0.4 * np.cos(x * 2.0 * np.pi / 24.0 - 1.2)
    scale = 1.30
    y0 = base + noise / 20.0
    y1 = scale * base + x / 400.0

    ts0 = pd.Series(y0, index=index)
    ts1 = pd.Series(y1, index=index)
    ts1 = ts1.shift(2)

    ts0.plot()
    ts1.plot()
    plt.show()

    # lag_sec = calculate_lag(ts0,ts1,minutes(60),res=seconds(1))
    # print("lag = {}".format((lag_sec/60.)))
    # print("skill = {}".format(skill_score(ts0, ts1)))

    dr = pd.date_range(start=pd.Timestamp(2009, 2, 10), freq="15min", periods=4)
    x = pd.Series([2.0, 4.0, 6.0, 8.0], index=dr)
    y = x * 1.5

    print(mse(x, y))
    print(rmse(x, y))
    print(skill_score(x, y))
    print(median_error(x, y))

    z0 = np.array([1.0, np.nan, 7.0, 5.0, 7.0, 10.0, 12.0, 14.0])
    z1 = np.array([2.0, 3.0, 5.0, 6.0, 8.0, 11.0, 13.0, 36.0])
    dr2 = pd.date_range(pd.Timestamp(2009, 2, 10), freq="15min", periods=8)

    zts0 = pd.Series(z0, index=dr2)
    zts1 = pd.Series(z1, index=dr2)
    print("r {}".format(corr_coefficient(zts0, zts1)))


all = [mse, rmse, median_error, tmean_error, corr_coefficient, skill_score]

if __name__ == "__main__":
    _main()
