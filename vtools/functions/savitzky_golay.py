import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P
from numpy.polynomial.polynomial import polyvander
from numba import njit
import matplotlib.pyplot as plt

all = ["savgol_filter_weighted"]


def savgol_filter_weighted(
    data, window_length, degree, error=None, cov_matrix=None, deriv=None, use_numba=True
):
    """
    Apply a Savitzky–Golay filter with weights to a univariate DataFrame or Series.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        DataFrame or Series containing your data.
    window_length : int
        Length of the filter window (must be odd).
    degree : int
        Degree of the polynomial fit.
    error : pandas.Series, optional
        Series containing the error (used to compute weights).
    cov_matrix : 2D numpy array, optional
        Covariance matrix for the errors.
    deriv : int, optional
        Derivative order to compute.
    use_numba : bool, optional
        If True, uses the Numba-accelerated kernel.

    Returns
    -------
    pandas.Series
        Series of the filtered values.

    Notes
    -----
    The practical size of `window_length` depends on the data and the computational resources.
    Larger window lengths provide smoother results but require more computation and may not capture
    local variations well. It is recommended to experiment with different window lengths to find
    the optimal value for your specific application.

    Some of the workflow derived from this work:
    https://github.com/surhudm/savitzky_golay_with_errors

    """

    # Interpolate to fill NaNs using linear interpolation
    interpolated_data = data.interpolate(method="linear")

    # Mark interpolated values in the error series
    if error is not None:
        error = error.copy()
        error[data.isna()] = np.nan

    # Extract data from DataFrame or Series
    y = interpolated_data.to_numpy()
    error = error.to_numpy() if error is not None else None

    # Choose the underlying implementation
    if use_numba:
        if error is None:
            raise ValueError("Numba version requires an 'error' Series for weights.")
        y_filtered = savgol_filter_numba(y, window_length, degree, error)
    else:
        y_filtered = savgol_filter_werror(
            y, window_length, degree, error=error, cov=cov_matrix, deriv=deriv
        )

    return pd.Series(y_filtered, index=data.index)


# --- Corrected Implementation ---
def solve_polyfit(xarr, yarr, degree, weight, deriv=None):
    # Use numpy.polyfit with weights via the polynomial module.
    # Skip NaN values and handle weights properly.
    mask = ~np.isnan(yarr) & ~np.isnan(weight)
    xarr = xarr[mask]
    yarr = yarr[mask]
    weight = weight[mask]
    try:
        z = P.polyfit(xarr, yarr, deg=degree, w=weight)
    except np.linalg.LinAlgError:
        # Fallback to np.linalg.lstsq with regularization
        vander = polyvander(xarr, degree)
        lhs = vander.T * weight
        rhs = yarr * weight
        reg = 1e-4 * np.eye(lhs.shape[0])
        z, _, _, _ = np.linalg.lstsq(lhs @ lhs.T + reg, lhs @ rhs, rcond=None)
    if deriv is not None:
        z = P.polyder(z, m=deriv)
    return z


def solve_leastsq(yarr, ycov, vander, vanderT, deriv=None):
    # Solve (V^T C^{-1} V) z = V^T C^{-1} y using np.linalg.solve for stability.
    ycovinv = np.linalg.inv(ycov)
    A = np.dot(np.dot(vanderT, ycovinv), vander)
    b = np.dot(np.dot(vanderT, ycovinv), yarr)
    z = np.linalg.solve(A, b)
    if deriv is not None:
        z = P.polyder(z, m=deriv)
    return z


def savgol_filter_werror(y, window_length, degree, error=None, cov=None, deriv=None):
    ynew = np.empty_like(y)
    ynew.fill(np.nan)  # Initialize with NaNs

    if window_length % 2 == 0:
        raise ValueError("Window length must be odd")

    margin = window_length // 2
    xarr = np.arange(-margin, margin + 1)

    if cov is not None:
        vander = polyvander(xarr, deg=degree)
        vanderT = np.transpose(vander)
    else:
        weight = np.zeros_like(error)
        weight[~np.isnan(error)] = 1.0 / error[~np.isnan(error)]
        weight[np.isnan(error)] = 1e-10  # Set very low weight for interpolated values

    # Main loop for the central part of the array
    for i in range(margin, y.size - margin):
        if np.isnan(y[i - margin : i + margin + 1]).sum() > margin:
            continue  # Skip if there are too many NaNs in the window
        try:
            if cov is None:
                z = solve_polyfit(
                    xarr,
                    y[i - margin : i + margin + 1],
                    degree,
                    weight[i - margin : i + margin + 1],
                    deriv=deriv,
                )
            else:
                z = solve_leastsq(
                    y[i - margin : i + margin + 1],
                    cov[i - margin : i + margin + 1, i - margin : i + margin + 1],
                    vander,
                    vanderT,
                    deriv=deriv,
                )
            ynew[i] = P.polyval(0.0, z)
        except Exception as e:
            print(f"Error at index {i}: {e}")

    # Left boundary: use the first window_length points
    for i in range(margin):
        if not np.isnan(y[:window_length]).any():
            try:
                if cov is None:
                    z = solve_polyfit(
                        xarr,
                        y[:window_length],
                        degree,
                        weight[:window_length],
                        deriv=deriv,
                    )
                else:
                    z = solve_leastsq(
                        y[:window_length],
                        cov[:window_length, :window_length],
                        vander,
                        vanderT,
                        deriv=deriv,
                    )
                ynew[i] = P.polyval(xarr[i], z)
            except Exception as e:
                print(f"Error at left boundary index {i}: {e}")

    # Right boundary: use the last window_length points
    for i in range(margin):
        if not np.isnan(y[-window_length:]).any():
            try:
                if cov is None:
                    z = solve_polyfit(
                        xarr,
                        y[-window_length:],
                        degree,
                        weight[-window_length:],
                        deriv=deriv,
                    )
                else:
                    z = solve_leastsq(
                        y[-window_length:],
                        cov[-window_length:, -window_length:],
                        vander,
                        vanderT,
                        deriv=deriv,
                    )
                ynew[y.size - margin + i] = P.polyval(xarr[i + margin + 1], z)
            except Exception as e:
                print(f"Error at right boundary index {i}: {e}")

    return ynew


# --- Numba-Accelerated Implementation ---
@njit
def _build_vander(x, degree):
    n = x.shape[0]
    A = np.empty((n, degree + 1))
    for i in range(n):
        v = 1.0
        for j in range(degree + 1):
            A[i, j] = v
            v *= x[i]
    return A


@njit
def _polyfit_window(x, y, w, degree):
    A = _build_vander(x, degree)
    n = A.shape[0]
    m = degree + 1

    ATA = np.zeros((m, m))
    ATy = np.zeros(m)
    for i in range(n):
        for j in range(m):
            ATy[j] += A[i, j] * w[i] * y[i]
            for k in range(m):
                ATA[j, k] += A[i, j] * w[i] * A[i, k]

    c = np.linalg.solve(ATA, ATy)
    return c


@njit
def _evaluate_poly(c, x):
    res = 0.0
    for i in range(c.shape[0] - 1, -1, -1):
        res = res * x + c[i]
    return res


@njit
def savgol_filter_numba(y, window_length, degree, error):
    n = y.shape[0]
    margin = window_length // 2
    ynew = np.empty(n)

    # Precompute the x positions for the window
    xarr = np.empty(window_length)
    for i in range(window_length):
        xarr[i] = i - margin

    inv_error = np.empty_like(error)
    for i in range(error.shape[0]):
        inv_error[i] = 1.0 / error[i]
        if np.isnan(error[i]):
            inv_error[i] = 1e-10  # Set very low weight for interpolated values

    # Main loop over the central data points
    for i in range(margin, n - margin):
        y_window = y[i - margin : i + margin + 1]
        w_window = inv_error[i - margin : i + margin + 1]
        c = _polyfit_window(xarr, y_window, w_window, degree)
        ynew[i] = _evaluate_poly(c, 0.0)

    # Left boundary: fit first window_length points
    c_left = _polyfit_window(xarr, y[:window_length], inv_error[:window_length], degree)
    for i in range(margin):
        ynew[i] = _evaluate_poly(c_left, xarr[i])

    # Right boundary: fit last window_length points
    c_right = _polyfit_window(
        xarr, y[-window_length:], inv_error[-window_length:], degree
    )
    for i in range(margin):
        ynew[n - margin + i] = _evaluate_poly(c_right, xarr[i + margin + 1])

    return ynew


# --- Example Usage ---
def main():
    # Create some sample data.
    t = pd.date_range(start="2023-01-01", periods=50 * 24, freq="h")
    pre_noise = np.sin(np.linspace(0, 40, len(t))) + np.cos(np.linspace(0, 15, len(t)))
    y = pre_noise + 0.23 * np.random.randn(len(t))
    error = 1 * np.ones_like(y)  # Example measurement uncertainties

    # Introduce gaps
    y[100:103] = np.nan  # 3-hour gap
    y[200:209] = np.nan  # 9-hour gap
    y[300:318] = np.nan  # 18-hour gap
    y[400:427] = np.nan  # 27-hour gap

    # Put data in a DataFrame.
    df = pd.DataFrame({"signal": y, "error": error}, index=t)

    # Apply the filter (using the pure numpy version)
    filtered_series = savgol_filter_weighted(
        df["signal"], window_length=75, degree=3, error=df["error"]
    )
    filtered_numba = (
        savgol_filter_weighted(
            df["signal"], window_length=75, degree=3, error=df["error"], use_numba=True
        )
        + 0.05
    )
    # Plot the original and filtered signals
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["signal"], label="Original Signal", alpha=0.5)
    plt.plot(
        df.index, pre_noise, label="pre-noise", linewidth=2, color="0.7", linestyle=":"
    )
    plt.plot(df.index, filtered_series, label="Filtered Signal", linewidth=2)
    plt.plot(df.index, filtered_numba, label="Filtered Numba", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.title("Savitzky-Golay Filter Example")
    plt.show(block=True)  # Ensure the plot is displayed before the script exits


if __name__ == "__main__":
    main()
