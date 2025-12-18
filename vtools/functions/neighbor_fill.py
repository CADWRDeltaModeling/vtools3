"""Neighbor-based time-series gap filling.

This module provides a single high-level API, :func:`fill_from_neighbor`,
with pluggable backends for common algorithms used to infer a target series
from one or more nearby stations. It is designed for operational use in
Delta/Bay hydrodynamics workflows, but is intentionally general.

Highlights
----------
- Robust time alignment and optional resampling.
- Multiple modeling strategies: OLS/robust, rolling regression, lagged elastic-net, and state-space/Kalman.
- Forward-chaining (temporal) cross-validation utilities.
- Optional regime stratification (e.g., barrier in/out, season).
- Uncertainty estimates where available (analytic or residual-based).
- Clear return structure with diagnostics for auditability.

Example
-------
>>> res = fill_from_neighbor(
...     target=y, neighbor=x, method="state_space", lags=range(0, 4),
...     bounds=(0.0, None), regime=regime_series
... )
>>> filled = res["filled"]
>>> info = res["model_info"]

Notes
-----
- "Neighbor" can be one series or multiple (as a DataFrame); both are supported.
- Missing data in the target are left as-is where the model cannot reasonably infer a value (e.g.no
  overlapping neighbor data). Where predictions exist, they are merged into the target to produce `filled`.
  DFM methods can carry through a gap in the neighbor.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional, Tuple, Dict, Any
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.special import expit
from sklearn.neighbors import KNeighborsRegressor


import warnings
import numpy as np
import pandas as pd

__all__ = ["fill_from_neighbor", "FillResult","dfm_pack_params","load_dfm_params","save_dfm_params"]

# Optional heavy dependencies
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf  # noqa: F401
    from statsmodels.robust.robust_linear_model import RLM
    from statsmodels.robust.norms import HuberT

    HAVE_SM = True
except Exception:  # pragma: no cover
    HAVE_SM = False

try:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    HAVE_SK = True
except Exception:  # pragma: no cover
    HAVE_SK = False


@dataclass
class FillResult:
    """Container for gap-filling outputs.

    Parameters
    ----------
    filled : pd.Series
        Target series with gaps filled where possible.

    yhat : pd.Series
        Model predictions aligned to the union index used for fitting/prediction.

    pi_lower, pi_upper : Optional[pd.Series]
        Prediction interval bounds where available; otherwise ``None``.

    model_info : dict
        Method, parameters, chosen lags, training window, etc.

    metrics : dict
        Holdout scores (MAE/RMSE/R^2) using forward-chaining CV where configured.
    """

    filled: pd.Series
    yhat: pd.Series
    pi_lower: Optional[pd.Series]
    pi_upper: Optional[pd.Series]
    model_info: Dict[str, Any]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        # Convert series to a serialization-friendly form if desired
        return out


# ---------------------------- Utilities ------------------------------------ #

_DEF_FREQ_ERR = "Target and neighbor must arrive on the same regular time grid (same step and phase). "


def _assert_same_regular_grid(idx_y: pd.DatetimeIndex, idx_x: pd.DatetimeIndex) -> None:
    """Raise if the two indices are not on the same regular grid (same step & phase)."""
    for name, idx in (("target", idx_y), ("neighbor", idx_x)):
        if idx.tz is not None and idx_y.tz != idx_x.tz:
            raise ValueError(
                "Mixed timezones between target and neighbor are not allowed."
            )
        if idx.size < 2:
            continue  # trivially OK (we'll fail later if there is no overlap)
        # check monotone and constant step
        if not idx.is_monotonic_increasing:
            raise ValueError(f"{name} index must be sorted/monotonic increasing.")
        d = np.diff(idx.view("i8"))
        if not np.all(d == d[0]):
            raise ValueError(f"{name} index is not equally spaced.")
    # compare steps
    if idx_y.size >= 2 and idx_x.size >= 2:
        step_y = (idx_y[1] - idx_y[0]).to_numpy()
        step_x = (idx_x[1] - idx_x[0]).to_numpy()
        if step_y != step_x:
            raise ValueError(
                _DEF_FREQ_ERR
                + f" (step mismatch: {pd.Timedelta(step_y)} vs {pd.Timedelta(step_x)})"
            )
        # compare phase relative to a fixed epoch
        epoch = pd.Timestamp("1970-01-01", tz=idx_y.tz)
        rem_y = (idx_y[0] - epoch).to_timedelta64() % step_y
        rem_x = (idx_x[0] - epoch).to_timedelta64() % step_x
        if rem_y != rem_x:
            raise ValueError(
                _DEF_FREQ_ERR + " (phase mismatch: grids are offset in time)"
            )


def _as_series_like(x: Union[pd.Series, pd.DataFrame], name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(name or (x.name or "x"))
    elif isinstance(x, pd.DataFrame):
        if x.columns.nlevels > 1:
            x = x.copy()
            x.columns = ["_".join(map(str, c)) for c in x.columns]
        return x
    raise TypeError("neighbor must be a Series or DataFrame")


def _align(
    y: pd.Series,
    X: Union[pd.Series, pd.DataFrame],
    how: str = "inner",
    allow_empty: bool = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Align target and neighbor(s) on a common DatetimeIndex."""
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("target index must be a DatetimeIndex")
    X = _as_series_like(X, name="x")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("neighbor index must be a DatetimeIndex")

    _assert_same_regular_grid(y.index, X.index)
    y_al = y
    X_al = X

    y2, X2 = y_al.align(X_al, join=how)
    if (len(y2) == 0 or len(X2) == 0) and not allow_empty:
        raise ValueError("No overlap between target and neighbor after alignment")
    return y2, X2


def _mask_overlap(y: pd.Series, X: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Keep only timestamps where both y and ALL X columns are non-NaN."""
    m = y.notna()
    for c in X.columns:
        m &= X[c].notna()
    return y[m], X[m]


def _suggest_lags(
    y: pd.Series,
    x: pd.Series,
    max_lag: int,
) -> List[int]:
    """Suggest non-negative lags (in steps) by cross-correlation peak.

    Returns a list of lags sorted by decreasing absolute correlation.
    """
    if max_lag <= 0:
        return [0]
    lags = np.arange(0, max_lag + 1)
    corrs = []
    for k in lags:
        corr = y.corr(x.shift(k))
        corrs.append(corr)
    order = np.argsort(-np.abs(np.asarray(corrs)))
    return lags[order].tolist()


def _add_lagged_X(X: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    Xlags = []
    for c in X.columns:
        for k in lags:
            Xlags.append(X[c].shift(int(k)).rename(f"{c}_lag{k}"))
    return pd.concat(Xlags, axis=1)


def _forward_chain_splits(
    n: int, n_splits: int = 3, min_train: int = 50
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate forward-chaining train/test index splits for time series.

    Parameters
    ----------
    n : int
        Number of samples.
    n_splits : int
        How many folds.
    min_train : int
        Minimum size of the initial training window.
    """
    if n < min_train + n_splits:
        # fall back to single split in the tail
        cut = max(min_train, n // 2)
        return [(np.arange(0, cut), np.arange(cut, n))]
    step = (n - min_train) // n_splits
    splits = []
    for i in range(n_splits):
        end_train = min_train + i * step
        test_end = min(min_train + (i + 1) * step, n)
        train_idx = np.arange(0, end_train)
        test_idx = np.arange(end_train, test_end)
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    with np.errstate(invalid="ignore"):
        resid = y_true - y_pred
        mae = float(np.nanmean(np.abs(resid)))
        rmse = float(np.sqrt(np.nanmean(resid**2)))
        # Simple R^2 (may be negative on holdout)
        ss_res = np.nansum(resid**2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ------------------------- Model Backends ---------------------------------- #


def _fit_substitute(y: pd.Series, X: pd.Series | pd.DataFrame):
    x = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X

    # Use ONLY target timestamps; no need for union alignment here
    x_on_y = x.reindex(y.index)

    # Start from target and fill only its gaps from neighbor
    yhat = y.copy()
    gap_mask = yhat.isna() & x_on_y.notna()
    yhat.loc[gap_mask] = x_on_y.loc[gap_mask]

    info = {
        "method": "substitute",
        "n_filled": int(gap_mask.sum()),
        "coverage_all": float(yhat.notna().mean()),
        "note": "Filled target gaps with neighbor where available; kept original data elsewhere.",
    }
    return yhat, None, None, info


def _fit_ols(
    y: pd.Series, X: pd.DataFrame
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Dict[str, Any]]:
    if not HAVE_SM:
        raise ImportError("statsmodels is required for OLS")
    y_fit, X_fit = _mask_overlap(y, X)
    X1 = sm.add_constant(X_fit, has_constant="add")
    model = sm.OLS(y_fit.values, X1.values, missing="drop")
    res = model.fit()
    # Predict where X is present (not necessarily y)
    X1_all = sm.add_constant(X, has_constant="add")
    pred = res.get_prediction(X1_all.values)
    yhat = pd.Series(pred.predicted_mean, index=X.index, name="yhat")
    conf = pred.conf_int(alpha=0.05)
    pi_lower = pd.Series(conf[:, 0], index=X.index, name="pi_lower")
    pi_upper = pd.Series(conf[:, 1], index=X.index, name="pi_upper")
    info = {
        "method": "ols",
        "params": res.params.tolist(),
        "param_names": ["const"] + list(X_fit.columns),
        "rsquared": float(res.rsquared),
    }
    return yhat, pi_lower, pi_upper, info


def _fit_huber(
    y: pd.Series, X: pd.DataFrame
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Dict[str, Any]]:
    if not HAVE_SM:
        raise ImportError("statsmodels is required for robust regression")
    y_fit, X_fit = _mask_overlap(y, X)
    X1 = sm.add_constant(X_fit, has_constant="add")
    rlm = RLM(y_fit.values, X1.values, M=HuberT())
    res = rlm.fit()
    X1_all = sm.add_constant(X, has_constant="add")
    yhat = pd.Series(res.predict(X1_all.values), index=X.index, name="yhat")
    # Approximate PI using residual std (not exact for Huber)
    resid = y_fit.values - res.predict(X1.values)
    sigma = float(np.nanstd(resid))
    pi_lower = yhat - 1.96 * sigma
    pi_upper = yhat + 1.96 * sigma
    info = {
        "method": "huber",
        "params": res.params.tolist(),
        "param_names": ["const"] + list(X_fit.columns),
        "scale": float(res.scale),
    }
    return yhat, pi_lower, pi_upper, info


def _fit_rolling_regression(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
    center: bool = False,
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Dict[str, Any]]:
    if not HAVE_SM:
        raise ImportError("statsmodels is required for rolling regression")
    # Rolling fits only where full window is available
    # Rolling fits only where full window is available
    yhat = pd.Series(index=X.index, dtype=float, name="yhat")
    beta_list = []
    Xc = sm.add_constant(X, has_constant="add")
    for end in range(window, len(Xc) + 1):
        start = end - window
        sl = slice(start, end)
        yi = y.iloc[sl]
        Xi = Xc.iloc[sl]
        yi2, Xi2 = _mask_overlap(yi, Xi)
        if len(yi2) < max(10, Xc.shape[1] * 3):
            continue
        res = sm.OLS(yi2.values, Xi2.values).fit()
        # predict at the last row of this window
        idx = Xc.index[end - 1]  # <- fixed
        yhat.loc[idx] = float(res.predict(Xc.iloc[[end - 1]].values))
        beta_list.append(res.params)

    # PI via rolling residual std (rough)
    pi_lower = None
    pi_upper = None
    info = {
        "method": "rolling_regression",
        "window": int(window),
        "center": bool(center),
    }
    if beta_list:
        info["beta_summary"] = {
            "mean": np.mean(beta_list, axis=0).tolist(),
            "std": np.std(beta_list, axis=0).tolist(),
        }
    return yhat, pi_lower, pi_upper, info


def _fit_lagged_elasticnet(
    y: pd.Series,
    X: pd.DataFrame,
    lags: Iterable[int],
    alphas: Optional[List[float]] = None,
    l1_ratio: float = 0.2,
    n_splits: int = 3,
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Dict[str, Any]]:
    if not HAVE_SK:
        raise ImportError("scikit-learn is required for elastic-net")
    Z = _add_lagged_X(X, lags)
    y_fit, Z_fit = _mask_overlap(y, Z)

    if alphas is None:
        alphas = np.logspace(-3, 1, 20)
    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "enet",
                ElasticNetCV(
                    l1_ratio=l1_ratio, alphas=alphas, cv=min(n_splits, 5), max_iter=5000
                ),
            ),
        ]
    )
    model.fit(Z_fit.values, y_fit.values)

    # Predict only where all lagged features exist (no NaNs)
    mask_pred = Z.notna().all(axis=1)
    yhat = pd.Series(index=Z.index, dtype=float, name="yhat")
    if mask_pred.any():
        yhat.loc[mask_pred] = model.predict(Z.loc[mask_pred].values)

    # PI via residual std (analytic PI isn't native to EN)
    resid = y_fit.values - model.predict(Z_fit.values)
    sigma = float(np.nanstd(resid))
    pi_lower = pd.Series(index=Z.index, dtype=float, name="pi_lower")
    pi_upper = pd.Series(index=Z.index, dtype=float, name="pi_upper")
    if mask_pred.any():
        pi_lower.loc[mask_pred] = yhat.loc[mask_pred] - 1.96 * sigma
        pi_upper.loc[mask_pred] = yhat.loc[mask_pred] + 1.96 * sigma

    info = {
        "method": "lagged_elasticnet",
        "lags": list(map(int, lags)),
        "alpha": float(model.named_steps["enet"].alpha_),
        "l1_ratio": float(l1_ratio),
        "n_features": int(Z.shape[1]),
    }
    return yhat, pi_lower, pi_upper, info


def fit_loess_time_value(
    y: pd.Series,
    X: pd.DataFrame,
    frac_time: float = 0.05,  # fraction of available points as neighbors
    min_neighbors: int = 25,  # floor on neighbors
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Dict[str, Any]]:
    """
    Two-dimensional LOESS-like smoother: y(t) ~ f(x(t), t), implemented as
    distance-weighted KNN in (time, value) space.

    - Avoids Series&DataFrame boolean broadcasting by reducing X→Series first.
    - Scales time and value so distances are comparable.
    - Predicts wherever neighbor is present.
    """
    # Ensure X is a DataFrame; pick the first neighbor column as driver
    if isinstance(X, pd.Series):
        x = X
    else:
        if X.shape[1] == 0:
            raise ValueError("X has no columns")
        x = X.iloc[:, 0]

    # Align y and x to the same index BEFORE making masks (avoids huge align ops)
    y, x = y.align(x, join="inner")

    good = y.notna() & x.notna()  # ← Series & Series (safe)
    if good.sum() < 50:
        raise ValueError("Insufficient overlap for loess2d fit.")

    # Build (time, value) features and scale them
    t = np.arange(len(y), dtype=float)
    t_mu, t_sd = float(t[good].mean()), float(t[good].std() or 1.0)
    x_mu, x_sd = float(x[good].mean()), float(x[good].std() or 1.0)

    t_s = (t - t_mu) / t_sd
    x_s = (x.to_numpy() - x_mu) / x_sd

    # Train KNN with distance weights on observed points
    n_train = int(good.sum())
    k = max(min_neighbors, int(frac_time * n_train))
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", metric="euclidean")

    TX_train = np.column_stack([t_s[good], x_s[good]])
    y_train = y[good].to_numpy()
    knn.fit(TX_train, y_train)

    # Predict wherever neighbor exists (even if y is NaN in gaps)
    pred_mask = x.notna()
    TX_pred = np.column_stack([t_s[pred_mask], x_s[pred_mask]])

    yhat = pd.Series(index=y.index, dtype=float, name="yhat")
    yhat.loc[pred_mask] = knn.predict(TX_pred)

    # Simple PI via residual std on training points
    resid = y_train - knn.predict(TX_train)
    sigma = float(np.nanstd(resid))
    pi_lower = yhat - 1.96 * sigma
    pi_upper = yhat + 1.96 * sigma

    info = {
        "method": "loess2d",
        "k": int(k),
        "frac_time": float(frac_time),
        "min_neighbors": int(min_neighbors),
        "scaled": True,
    }
    return yhat, pi_lower, pi_upper, info


def _fit_loess(
    y: pd.Series,
    X: pd.DataFrame,
    frac: float = 0.2,
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Dict[str, Any]]:
    """Locally weighted regression (LOESS) smoother for neighbor fill."""
    # For univariate neighbor only
    if X.shape[1] > 1:
        raise ValueError("LOESS currently supports one neighbor column.")
    y_fit, X_fit = _mask_overlap(y, X)
    if len(y_fit) < 10:
        raise ValueError("Insufficient overlap for LOESS fit.")

    xvals = X_fit.iloc[:, 0].to_numpy()
    yvals = y_fit.to_numpy()

    smoothed = lowess(yvals, xvals, frac=frac, return_sorted=False)
    yhat = pd.Series(
        np.interp(X.iloc[:, 0], xvals, smoothed), index=X.index, name="yhat"
    )

    # No rigorous PI, but you can provide a rough one via residual std
    resid = y_fit - np.interp(X_fit.iloc[:, 0], xvals, smoothed)
    sigma = float(np.nanstd(resid))
    pi_lower = yhat - 1.96 * sigma
    pi_upper = yhat + 1.96 * sigma

    info = {"method": "loess", "frac": frac, "sigma": sigma}
    return yhat, pi_lower, pi_upper, info


# -------------------- Configurable DFM (factor/anomaly options) -------------------- #
# --- helpers (add near your _DFM2) ------------------------------------------
def _logit(x):
    x = np.clip(float(x), 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


def _inv_logit(z):
    z = float(np.clip(z, -20, 20))
    return 1.0 / (1.0 + np.exp(-z))


def _phi_from_logit(z):
    p = _inv_logit(z)  # in (0,1)
    return 2 * p - 1.0  # map to (-1,1)


# --- variable-size DFM ------------------------------------------------------


def _opt_debug(mod, res):
    """Print start vs fitted params (both transformed & constrained)."""
    sp = mod.start_params
    ep = res.params
    names = mod.param_names

    def _constrained(vals):
        # map transformed → constrained dict, in the model’s own way
        return mod._constrain(vals)

    cs = _constrained(sp)
    ce = _constrained(ep)

    # print("\n=== Optimization summary ===")
    # print("converged:", getattr(res, "mle_retvals", {}).get("converged"))
    # print("niter:", getattr(res, "mle_retvals", {}).get("nit") or getattr(res, "mle_retvals", {}).get("niter"))
    # print(f"{'name':<14} {'start':>12} {'fitted':>12}   {'constrained(start)':>20} {'constrained(fitted)':>20}")
    for i, n in enumerate(names):
        s = float(sp[i])
        e = float(ep[i])
        # show key constrained entries if present; else blank
        k = n.replace("log_", "").replace("logit_", "")
        cs_v = cs.get(k, float("nan"))
        ce_v = ce.get(k, float("nan"))
        print(f"{n:<14} {s:12.4g} {e:12.4g}   {cs_v:20.6g} {ce_v:20.6g}")
    print("============================\n")


def _fit_resid_interp(
    y: pd.Series,
    X: pd.DataFrame,
    kind: str = "linear",  # "linear" | "pchip"
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Dict[str, Any]]:
    """
    Fill y using neighbor via interpolated residuals.

    Steps:
      1) Fit baseline y ≈ a + b x on overlap (OLS; falls back to ratio if needed).
      2) Residuals r = y - (a + b x) on overlap.
      3) Interpolate r only inside gaps (bounded on both sides) using 'linear' or 'pchip'.
      4) Reconstruct yhat = (a + b x) + r_interp wherever x is available.
    """
    # Reduce to one neighbor
    x = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X

    # Align to common index; keep outer so we can predict everywhere x is present
    y_al, x_al = y.align(x, join="outer")
    m_fit = y_al.notna() & x_al.notna()

    if m_fit.sum() < 3:
        # fall back: simple scaling
        raise ValueError(
            "Insufficient overlap to fit residual-interp baseline (need ≥3 points)."
        )

    # --- 1) Fit baseline y ≈ a + b x on overlap
    try:
        import statsmodels.api as sm  # prefer stable OLS if available

        X1 = sm.add_constant(x_al[m_fit].values, has_constant="add")
        res = sm.OLS(y_al[m_fit].values, X1, missing="drop").fit()
        a_hat = float(res.params[0])
        b_hat = float(res.params[1])
    except Exception:
        # Ratio fallback if statsmodels not present
        b_hat = float(
            np.nanmedian((y_al[m_fit] / x_al[m_fit]).replace([np.inf, -np.inf], np.nan))
        )
        if not np.isfinite(b_hat):
            b_hat = 1.0
        a_hat = float(np.nanmedian(y_al[m_fit] - b_hat * x_al[m_fit]))

    # Base prediction everywhere x exists
    y_base = pd.Series(index=y_al.index, dtype=float, name="yhat_base")
    y_base.loc[x_al.notna()] = a_hat + b_hat * x_al.loc[x_al.notna()]

    # --- 2) Residuals on overlap
    resid = y_al[m_fit] - (a_hat + b_hat * x_al[m_fit])
    resid = resid.sort_index()

    # --- 3) Interpolate residuals inside gaps only
    # We'll interpolate in time; no extrapolation past the first/last observed residual.
    r_all = pd.Series(index=y_al.index, dtype=float, name="r_interp")
    if kind == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator

            # Numeric time in seconds for monotonic grid
            t = resid.index.view("i8") / 1e9
            r = resid.values.astype(float)
            # Need strictly increasing unique t; drop duplicates if any
            t_unique, idx_unique = np.unique(t, return_index=True)
            r_unique = r[idx_unique]
            p = PchipInterpolator(t_unique, r_unique, extrapolate=False)
            t_all = y_al.index.view("i8") / 1e9
            r_interp = p(t_all)
            r_all[:] = r_interp
        except Exception:
            # Graceful fallback to linear
            kind = "linear"
            # continue into linear branch below

    if kind == "linear":
        # Pandas 'time' interpolation uses the DatetimeIndex and stays inside by default with limit_area='inside'
        r_series = pd.Series(index=y_al.index, dtype=float)
        r_series.loc[resid.index] = resid.values
        r_all = r_series.interpolate(method="time", limit_area="inside")

    # We only want residuals where the gap is bounded AND neighbor exists
    pred_mask = x_al.notna()
    # yhat = base + interpolated residual (where available)
    yhat = pd.Series(index=y_al.index, dtype=float, name="yhat")
    yhat.loc[pred_mask] = y_base.loc[pred_mask]
    # add residuals where we have an interpolated value
    ok_r = pred_mask & r_all.notna()
    yhat.loc[ok_r] = y_base.loc[ok_r] + r_all.loc[ok_r]

    # --- 4) Simple PI: constant residual sigma from overlap
    sigma = float(np.nanstd(resid.values)) if resid.size else np.nan
    if np.isfinite(sigma):
        pi_lower = yhat - 1.96 * sigma
        pi_upper = yhat + 1.96 * sigma
    else:
        pi_lower = None
        pi_upper = None

    info = {
        "method": "resid_interp_" + kind,
        "baseline": {"a": a_hat, "b": b_hat},
        "sigma_resid": sigma,
        "n_overlap": int(m_fit.sum()),
    }
    return yhat, pi_lower, pi_upper, info


class DFMFill(MLEModel):
    """
    Bivariate DFM with level+slope common factor and optional anomalies.

    """

    def __init__(
        self,
        endog: pd.DataFrame,
        factor: str = "default",  # {"default","trimbur"}
        anomaly_mode: str = "ar",  # {"ar","rw"}
        anom_var: str = "neighbor",  # {"target","neighbor","both"}
        rx_scale: float = 1.0,
    ):
        endog = endog[["y", "x"]]
        assert factor in {"default", "trimbur"}
        assert anomaly_mode in {"ar", "rw"}
        assert anom_var in {"target", "neighbor", "both"}

        # Decide which anomaly slots exist and index layout
        have_ay = anom_var in {"target", "both"}
        have_ax = anom_var in {"neighbor", "both"}
        idx = {"mu": 0, "beta": 1}
        k = 2
        if have_ay:
            idx["ay"], k = k, k + 1
        if have_ax:
            idx["ax"], k = k, k + 1
        self._idx = idx
        self._have_ay, self._have_ax = have_ay, have_ax

        super().__init__(endog=endog, k_states=k, initialization="approximate_diffuse")

        self.factor = factor
        self.anomaly_mode = anomaly_mode
        self.anom_var = anom_var
        self.rx_scale = float(rx_scale)

        # Parameter list (only what we truly estimate)
        names = []
        if self.factor == "default":
            names += ["log_q_mu", "log_q_beta"]
        else:  # trimbur: q_mu is fixed to 0
            names += ["log_q_beta"]

        if have_ay:
            names += (
                ["log_q_ay", "logit_phi_y"] if anomaly_mode == "ar" else ["log_q_ay"]
            )
        if have_ax:
            names += (
                ["log_q_ax", "logit_phi_x"] if anomaly_mode == "ar" else ["log_q_ax"]
            )

        names += ["log_r_y", "log_r_x", "load"]
        if anom_var == "both":
            # drop 'load' → fix at 1.0
            names = [n for n in names if n != "load"]

        self._param_names = names
        self.k_params = len(names)

        # Allocate system matrices with correct size
        T = np.eye(k, dtype=float)
        T[idx["mu"], idx["beta"]] = 1.0  # local-linear trend
        self["transition"] = T
        self["selection"] = np.eye(k, dtype=float)
        self["state_cov"] = np.eye(k, dtype=float)
        self["design"] = np.zeros((2, k), dtype=float)
        self["obs_cov"] = np.eye(2, dtype=float)

    @property
    def param_names(self):
        return list(self._param_names)

    @property
    def start_params(self) -> np.ndarray:
        st = {}
        # factor starts per spec
        if "log_q_mu" in self._param_names:
            st["log_q_mu"] = np.log(1e-8)  # default: nonzero
        # Trimbur has no q_mu param; it will be set to 0.0 in update()
        if "log_q_beta" in self._param_names:
            st["log_q_beta"] = np.log(1e-8)

        # anomaly roughness
        if "log_q_ay" in self._param_names:
            st["log_q_ay"] = np.log(1e-7)
        if "log_q_ax" in self._param_names:
            st["log_q_ax"] = np.log(1e-7)
        # AR phis (if any)
        if "logit_phi_y" in self._param_names:
            st["logit_phi_y"] = _logit(0.98)
        if "logit_phi_x" in self._param_names:
            st["logit_phi_x"] = _logit(0.98)

        # measurement & loading
        st["log_r_y"] = np.log(1e-4)
        st["log_r_x"] = np.log(1e-4)
        st["load"] = 1.0

        return np.array([st[n] for n in self._param_names], dtype=float)

    def _constrain(self, vec):
        # Ensure real (avoids ComplexWarning)
        v = np.asanyarray(vec)
        if np.iscomplexobj(v):
            v = v.real
        v = np.asarray(v, dtype=float, copy=False)
        raw = dict(zip(self._param_names, v))
        out = {}

        # exp-map variances safely
        for k in (
            "log_q_mu",
            "log_q_beta",
            "log_q_ay",
            "log_q_ax",
            "log_r_y",
            "log_r_x",
        ):
            if k in raw:
                out[k.replace("log_", "")] = float(np.exp(np.clip(raw[k], -40, 40)))

        # phis
        for k in ("logit_phi_y", "logit_phi_x"):
            if k in raw:
                out[k.replace("logit_", "")] = float(
                    np.clip(_phi_from_logit(raw[k]), -0.995, 0.995)
                )

        # loading
        out["load"] = float(raw.get("load", 1.0))
        return out

    def update(self, params, transformed=True, **kwargs):
        p = self._constrain(params)

        idx = self._idx
        k = self.k_states

        # --- Transition T ---
        T = self["transition"].copy()
        # anomalies’ AR(1) or RW phi on diagonal only if present
        if self._have_ay:
            T[idx["ay"], idx["ay"]] = (
                1.0 if self.anomaly_mode == "rw" else p.get("phi_y", 0.0)
            )
        if self._have_ax:
            T[idx["ax"], idx["ax"]] = (
                1.0 if self.anomaly_mode == "rw" else p.get("phi_x", 0.0)
            )

        # --- State covariance Q ---
        Q = np.zeros((k, k), dtype=float)
        if self.factor == "default":
            Q[idx["mu"], idx["mu"]] = p.get("q_mu", 1e-8)  # nonzero
            Q[idx["beta"], idx["beta"]] = p.get("q_beta", 1e-8)
        else:  # Trimbur: level has no shock; slope has small roughness
            Q[idx["mu"], idx["mu"]] = 0.0
            Q[idx["beta"], idx["beta"]] = p.get("q_beta", 1e-8)

        if self._have_ay:
            Q[idx["ay"], idx["ay"]] = p.get("q_ay", 1e-7)
        if self._have_ax:
            Q[idx["ax"], idx["ax"]] = p.get("q_ax", 1e-7)

        # --- Design Z ---
        Z = np.zeros((2, k), dtype=float)
        # target: loads 1*mu + ay(if present)
        Z[0, idx["mu"]] = 1.0
        if self._have_ay:
            Z[0, idx["ay"]] = 1.0
        # neighbor: loads load*mu + ax(if present)
        Z[1, idx["mu"]] = p.get("load", 1.0)
        if self._have_ax:
            Z[1, idx["ax"]] = 1.0

        # --- Obs covariance H (inflate x by rx_scale) ---
        r_y = max(p.get("r_y", 1e-4), 1e-5)
        r_x = max(p.get("r_x", 1e-4), 1e-5) * float(self.rx_scale)
        H = np.diag([r_y, r_x])

        # Commit
        self["transition"] = T
        self["state_cov"] = Q
        self["design"] = Z
        self["obs_cov"] = H


# --- DFM parameter helpers (no side effects) -------------------
# --- YAML-first DFM param helpers (public API) -------------------------------
from typing import Any, Dict
import os


def dfm_pack_params(model_info: dict) -> dict:
    """
    Return a portable blob of fitted DFM params.

    Parameters
    ----------
    model_info : dict
        Model info dictionary, typically from `fill_from_neighbor`.

    Returns
    -------
    dict
        Dictionary containing fitted DFM parameters with the following keys:
        - 'param_names': list of parameter names.
        - 'transformed': list of transformed parameter values.
        - 'constrained': dictionary of constrained parameter values.
        - 'mle': dictionary with optimizer info (optional).
        - 'reused': bool indicating if parameters were reused (optional).

    Raises
    ------
    TypeError
        If `model_info` is not a dictionary.
    ValueError
        If no fitted parameters are found in `model_info`.
    """
    if not isinstance(model_info, dict):
        raise TypeError("model_info must be a dict (from fill_from_neighbor).")

    blob = model_info.get("fitted_params")
    if isinstance(blob, dict) and "param_names" in blob and "transformed" in blob:
        return blob

    # ---- legacy fallback: older code returned only param_names + params ----
    if "param_names" in model_info and "params" in model_info:
        return {
            "param_names": list(map(str, model_info["param_names"])),
            "transformed": list(map(float, model_info["params"])),
            "constrained": {},  # unknown in legacy
            "mle": {"converged": True},  # best-effort
            "reused": False,
        }

    raise ValueError(
        "No fitted params found. You may be importing an older neighbor_fill "
        "that does not populate model_info['fitted_params'] for DFM."
    )


def save_dfm_params(params: Dict[str, Any], path: str) -> None:
    """
    Save a DFM parameter blob to YAML (preferred for this codebase).
    File extension may be .yaml or .yml. Other extensions raise.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".yaml", ".yml"):
        raise ValueError("Please use a .yaml or .yml filename for DFM params.")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to save YAML. Install 'pyyaml'.") from e
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)


def load_dfm_params(path: str) -> Dict[str, Any]:
    """
    Load a DFM parameter blob from YAML and validate minimal schema.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".yaml", ".yml"):
        raise ValueError("Expected a .yaml or .yml file for DFM params.")
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to load YAML. Install 'pyyaml'.") from e
    with open(path, "r", encoding="utf-8") as f:
        blob = yaml.safe_load(f)
    if not isinstance(blob, dict):
        raise ValueError("Loaded DFM params are not a dict.")
    if (
        "param_names" not in blob
        or "transformed" not in blob
        or "constrained" not in blob
    ):
        raise ValueError(
            "DFM params missing 'param_names', 'transformed', or 'constrained'."
        )
    return blob


# ---------------------------------------------------------------------------


def _dfm_params_to_vector(mod, params):
    """
    Build transformed vector in the exact order of mod.param_names from either:
      - {'transformed': [...], 'param_names': [...]}, or
      - a constrained dict {'q_beta':..., 'q_ax':..., 'r_y':..., 'r_x':..., 'phi_x':..., 'load':...}
    """
    if params is None or (isinstance(params, dict) and len(params) == 0):
        raise ValueError("params is empty; pass a saved blob or constrained dict.")
    names = list(mod.param_names)

    # Case A: transformed vector given
    if isinstance(params, dict) and "transformed" in params:
        vec = np.asarray(params["transformed"], dtype=float)
        if vec.shape[0] != len(names):
            raise ValueError(
                "Length of transformed vector does not match model param_names."
            )
        return vec

    # Case B: constrained dict -> transformed in correct order
    c = params

    def _logpos(x):
        x = float(x)
        if x <= 0:
            raise ValueError("All variances must be > 0 in constrained dict.")
        return np.log(x)

    def _phi_to_logit(phi):
        p = (float(phi) + 1.0) / 2.0
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    out = []
    for n in names:
        if n.startswith("log_q_"):
            out.append(_logpos(c[n.replace("log_", "")]))
        elif n.startswith("log_r_"):
            out.append(_logpos(c[n.replace("log_", "")]))
        elif n.startswith("logit_phi_"):
            out.append(_phi_to_logit(c[n.replace("logit_", "")]))
        elif n == "load":
            out.append(float(c.get("load", 1.0)))
        else:
            raise ValueError(f"Unrecognized parameter '{n}' in model param_names.")
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------
def _fit_dfm(
    y,
    X,
    *,
    factor: str = "default",
    anomaly_mode: str = "ar",
    anom_var: str = "neighbor",
    rx_scale: float = 3.0,
    maxiter: int = 80,
    disp: int = 0,
    params: dict | None = None,  # <-- ONLY control now
):
    # 1) align + standardize
    x = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
    y, x = y.align(x, join="outer")
    m = y.notna() & x.notna()
    if m.sum() >= 10:
        y_mu, y_sd = float(y[m].mean()), float(y[m].std(ddof=0) or 1.0)
        x_mu, x_sd = float(x[m].mean()), float(x[m].std(ddof=0) or 1.0)
    else:
        y_mu, y_sd = 0.0, float(y.std(ddof=0) or 1.0)
        x_mu, x_sd = 0.0, float(x.std(ddof=0) or 1.0)
    endog = pd.DataFrame({"y": (y - y_mu) / y_sd, "x": (x - x_mu) / x_sd})

    # 2) build model
    mod = DFMFill(
        endog=endog,
        factor=factor,
        anomaly_mode=anomaly_mode,
        anom_var=anom_var,
        rx_scale=rx_scale,
    )

    # 3) reuse OR fit; always produce fitted_params
    reused = False
    if isinstance(params, dict) and len(params) == 0:
        params = None  # treat {} as None

    if params is not None:
        vec = _dfm_params_to_vector(mod, params)  # transformed in model order
        res = mod.smooth(vec)  # no optimizer
        mod.update(vec, transformed=False)
        constrained = mod._constrain(vec)
        transformed = np.asarray(vec, dtype=float)
        mle = {"converged": True, "nit": 0}
        reused = True
    else:
        res = mod.fit(maxiter=maxiter, disp=disp)  # optimizer runs
        vec = np.asarray(res.params, dtype=float)
        mod.update(vec, transformed=False)
        constrained = mod._constrain(vec)
        transformed = vec
        mr = getattr(res, "mle_retvals", {}) or {}
        mle = {
            "converged": bool(mr.get("converged", True)),
            "nit": int(mr.get("nit") or mr.get("niter") or 0),
        }

    # 4) back-transform yhat + PI
    S = res.smoothed_state
    Z = mod["design"]
    H = mod["obs_cov"]
    P = res.smoothed_state_cov
    Zy = Z[0, :].reshape(1, -1)
    yhat_s = (Zy @ S).ravel()
    yhat = pd.Series(yhat_s * y_sd + y_mu, index=endog.index, name="yhat")

    nobs = P.shape[2]
    var_y = np.empty(nobs, dtype=float)
    for t in range(nobs):
        var_y[t] = float(Zy @ P[:, :, t] @ Zy.T + H[0, 0])
    se = np.sqrt(np.clip(var_y, 0.0, np.inf))
    pi_lower = pd.Series(
        yhat.values - 1.96 * se * y_sd, index=yhat.index, name="pi_lower"
    )
    pi_upper = pd.Series(
        yhat.values + 1.96 * se * y_sd, index=yhat.index, name="pi_upper"
    )

    # 5) prints (unchanged)
    print("Z(y):", np.round(Z[0, :], 4), " Z(x):", np.round(Z[1, :], 4))
    print("diag(T):", np.round(np.diag(mod["transition"]), 4))
    print(
        "diag(Q):",
        np.round(np.diag(mod["state_cov"]), 8),
        " diag(H):",
        np.round(np.diag(H), 8),
    )
    print(
        "modes:",
        f"factor={factor}",
        f"anom_mode={anomaly_mode}",
        f"anom_var={anom_var}",
    )
    print("active params:", mod.param_names)

    # 6) always-populated param blob
    fitted_params = {
        "param_names": list(mod.param_names),
        "transformed": transformed.tolist(),
        "constrained": constrained,
        "mle": mle,
        "reused": reused,
    }

    info = {
        "method": "dfm",
        "factor": factor,
        "anomaly_mode": anomaly_mode,
        "anom_var": anom_var,
        "rx_scale": float(rx_scale),
        "param_names": list(mod.param_names),
        "fitted_params": fitted_params,  # <- keep this!
        "scaling": {"y_mu": y_mu, "y_sd": y_sd, "x_mu": x_mu, "x_sd": x_sd},
        "llf": float(getattr(res, "llf", np.nan)),
        "aic": float(getattr(res, "aic", np.nan)),
        "bic": float(getattr(res, "bic", np.nan)),
    }
    return yhat, pi_lower, pi_upper, info

    res = mod.fit(maxiter=maxiter, disp=disp)
    mod.update(
        res.params, transformed=False
    )  # ensure matrices reflect constrained params
    # _opt_debug(mod, res)

    # Smoothed y = Z_y · E[state|all]
    S = res.smoothed_state
    Z = mod["design"]
    H = mod["obs_cov"]
    P = res.smoothed_state_cov
    Zy = Z[0, :].reshape(1, -1)
    yhat_s = (Zy @ S).ravel()
    yhat = pd.Series(yhat_s * y_sd + y_mu, index=endog.index, name="yhat")

    # 95% PI with state covariance
    nobs = P.shape[2]
    var_y = np.empty(nobs, dtype=float)
    for t in range(nobs):
        var_y[t] = float(Zy @ P[:, :, t] @ Zy.T + H[0, 0])
    se = np.sqrt(np.clip(var_y, 0.0, np.inf))
    pi_lower = pd.Series(
        yhat.values - 1.96 * se * y_sd, index=yhat.index, name="pi_lower"
    )
    pi_upper = pd.Series(
        yhat.values + 1.96 * se * y_sd, index=yhat.index, name="pi_upper"
    )

    # Diagnostics similar to your prints
    print("Z(y):", np.round(Z[0, :], 4), " Z(x):", np.round(Z[1, :], 4))
    print("diag(T):", np.round(np.diag(mod["transition"]), 4))
    print(
        "diag(Q):",
        np.round(np.diag(mod["state_cov"]), 8),
        " diag(H):",
        np.round(np.diag(H), 8),
    )
    print(
        "modes:",
        f"factor={factor}",
        f"anom_mode={anomaly_mode}",
        f"anom_var={anom_var}",
    )
    print("active params:", mod.param_names)

    info = {
        "method": "dfm",
        "factor": factor,
        "anomaly_mode": anomaly_mode,
        "anom_var": anom_var,
        "param_names": mod.param_names,
        "params": res.params.tolist(),
        "llf": float(res.llf),
        "aic": float(getattr(res, "aic", np.nan)),
        "bic": float(getattr(res, "bic", np.nan)),
    }

    info = {
        # ... your existing fields ...
    }
    # If we actually fit, return the fitted params for persistence.
    if fit and params is None and hasattr(res, "params"):
        try:
            info["fitted_params"] = _dfm_pack_params(mod, res)
        except Exception:
            pass

    return yhat, pi_lower, pi_upper, info


# ----------------------------- Orchestrator -------------------------------- #


def fill_from_neighbor(
    target: pd.Series,
    neighbor: Union[pd.Series, pd.DataFrame],
    method: str = "substitute",
    regime: Optional[pd.Series] = None,
    bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    *,
    params: Optional[dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Fill gaps in ``target`` using information from ``neighbor``.

    This is a high-level wrapper with multiple method backends (OLS/robust,
    rolling regression, lagged regression, LOESS-in-time, Trimbur-style DFM
    variants, residual-interpolation baselines, or simple substitution).
    Inputs must already lie on the **same regular time grid** (same step and phase);
    this function does **not** resample.

    Parameters
    ----------
    target : pandas.Series
        Target time series with a ``DatetimeIndex`` on a regular grid. Values may be NaN.

    neighbor : pandas.Series or pandas.DataFrame
        One or more neighbor series with a ``DatetimeIndex`` on the **same grid**
        as ``target`` (same step and phase). Values may be NaN.

    method : {'substitute', 'ols', 'huber', 'rolling', 'lagged_reg',
            'loess', 'dfm_trimbur_rw', 'dfm_trimbur_ar',
            'resid_interp_linear', 'resid_interp_pchip'}
        Algorithm to use:

        - ``'substitute'``: pass-through neighbor after mean/scale alignment.
        - ``'ols'``: ordinary least squares on overlap (optionally with lags).
        - ``'huber'``: robust regression with Huber loss (optionally with lags).
        - ``'rolling'``: rolling-window OLS in *sample* units (not time offsets).
        - ``'lagged_reg'``: multivariate regression on specified neighbor lags.
        - ``'loess'``: LOESS (time → value) smoothing using neighbor as scaffold.
        - ``'dfm_trimbur_rw'``: dynamic factor model (Trimbur factor) with
        random-walk anomaly for the target.
        - ``'dfm_trimbur_ar'``: dynamic factor model (Trimbur factor) with AR anomaly
        on the neighbor.
        - ``'resid_interp_linear'`` / ``'resid_interp_pchip'``: baseline y≈a+bx fit
        on overlap, then interpolate residuals (linear or PCHIP) across gaps.

    regime : pandas.Series, optional
        Optional categorical series indexed like ``target`` to stratify fits
        (e.g., barrier in/out). If provided, models are fit per category and
        stitched back together.

    bounds : (float or None, float or None)
        Lower/upper bounds to clip the final filled values (applied at the end).

    params : dict, optional
        Pre-fitted/packed parameter blob for methods that support parameter reuse
        (e.g., the DFM backends). If provided, fitting is skipped and the supplied
        parameters are used directly.

    **kwargs
        Method-specific optional arguments. Unsupported keys are ignored unless
        otherwise noted. Typical extras by method:

        Common
            lags : int or Sequence[int], optional
                Non-negative lags (in samples) for neighbor features. If an int m is
                provided, implementations may expand to range(0, m+1). Default behavior
                varies by method (often no lags or a small heuristic set).
            seed : int, optional
                Random seed for any stochastic initializations (where applicable).

        'ols'
            lags : int or Sequence[int], optional
            add_const : bool, default True
                Include an intercept term.
            fit_intercept : bool, alias of ``add_const``.

        'huber'
            lags : int or Sequence[int], optional
            huber_t : float, default 1.35
                Huber threshold (in residual σ units).
            maxiter : int, default 200
            tol : float, default 1e-6

        'rolling'
            window : int, required
                Rolling window length **in samples** (integer). Time-offset strings
                (e.g., '14D') are **not** supported here.
            min_periods : int, optional
                Minimum non-NaN samples required inside each window (default = window).
            center : bool, default False
                Whether to center the rolling window.
            lags : int or Sequence[int], optional
                If provided, each regression uses lagged neighbor columns inside
                the window.

        'lagged_reg'
            lags : int or Sequence[int], recommended
            alpha : float, optional
                Ridge/L2 penalty (if the backend supports it).
            l1_ratio : float, optional
                Elastic-net mixing (if the backend supports it).
            standardize : bool, default True
                Standardize columns before regression.

        'loess'
            frac : float, default 0.25
                LOESS span as a fraction of the data length (used in time→value smoothing).
            it : int, default 0
                Number of robustifying reweighting iterations.
            degree : int, default 1
                Local polynomial degree.

        'dfm_trimbur_rw' / 'dfm_trimbur_ar'
            rx_scale : float, default 1.0
                Relative scale factor for neighbor measurement noise.
            maxiter : int, default 80
                Maximum optimizer iterations during parameter fitting.
            disp : int, default 0
                Optimizer verbosity (0 = silent).
            anom_var : {'target','neighbor'}, optional
                Which series carries the anomaly/noise term (fixed by the variant,
                but may be overridden).
            ar_order : int, optional
                AR order for the anomaly in the ``'_ar'`` variant (default may be 1).
            param_names : list[str], optional
                For advanced users: explicit parameter naming (used when packing).
            # Note: DFM backends accept ``params=...`` at the top level for reuse.

        'resid_interp_linear' / 'resid_interp_pchip'
            min_overlap : int, default 3
                Minimum overlapping samples required to fit the baseline y≈a+bx.
            clip_residuals_sigma : float, optional
                Winsorize residuals before interpolation (σ units).
            enforce_monotone : bool, default False
                For PCHIP path only: enforce monotonic segments where applicable.

    Returns
    -------
    dict
        Dictionary with the following keys:

        yhat : pandas.Series
            Filled series on the same index as ``target``.
        pi_lower : pandas.Series or None
            Lower uncertainty band (if the method provides one), otherwise None.
        pi_upper : pandas.Series or None
            Upper uncertainty band (if the method provides one), otherwise None.
        model_info : dict
            Method-specific diagnostics and metadata. Typical fields include:
            ``method``, ``param_names``, ``fitted_params`` (packed blob for reuse),
            ``scaling`` (means/stds used), goodness-of-fit (e.g., ``llf``, ``aic``,
            ``bic``), and per-regime info when ``regime`` is provided.

    Raises
    ------
    ValueError
        If indices are not equally spaced, or grids mismatch in step or phase,
        or if required method-specific kwargs are missing (e.g., ``window`` for
        ``method='rolling'``).
    KeyError
        If an unknown method name is provided.
    """

    # add these names in the set:
    recognized = {
        "ols",
        "huber",
        "rolling",
        "lagged_reg",
        "loess",
        "dfm",
        "dfm_trimbur_ar",
        "dfm_trimbur_rw",
        "resid_interp_linear",
        "resid_interp_pchip",
        "substitute",
    }

    if method not in recognized:
        raise ValueError("Unknown method: %s" % method)

    if (params is not None) and method not in {"dfm_trimbur_rw", "dfm_trimbur_ar"}:
        raise ValueError("'fit'/'params' are only supported for DFM methods.")

    y0 = target.copy()
    X0 = _as_series_like(neighbor, name="x")

    # Pull optional tuning params from **kwargs (keeps backward compat in notebooks)
    lags = kwargs.pop("lags", None)
    window = kwargs.pop("window", None)

    # Align to common grid; for rolling with str window we regularize first
    y_al, X_al = _align(y0, X0, how="outer", allow_empty=False)

    # If regime given, process per category and stitch
    if regime is not None:
        reg = regime.reindex(y_al.index).ffill().bfill()
        cats = pd.Categorical(reg).categories
        yhats = []
        pil_list = []
        piu_list = []
        info_all = {"method": method, "by_regime": {}}
        for cat in cats:
            mask = reg == cat
            y_c = y_al.where(mask)
            X_c = X_al.where(mask)
            res_c = fill_from_neighbor(
                y_c, X_c, method=method, lags=lags, window=window, bounds=bounds
            )
            yhats.append(pd.Series(res_c["yhat"], name=str(cat)))
            pil_list.append(pd.Series(res_c.get("pi_lower"), name=str(cat)))
            piu_list.append(pd.Series(res_c.get("pi_upper"), name=str(cat)))
            info_all["by_regime"][str(cat)] = res_c["model_info"]
        # Combine by choosing regime-specific predictions
        yhat = pd.concat(yhats, axis=1).bfill(axis=1).iloc[:, 0]
        pi_lower = pd.concat(pil_list, axis=1).bfill(axis=1).iloc[:, 0]
        pi_upper = pd.concat(piu_list, axis=1).bfill(axis=1).iloc[:, 0]
        info = info_all
    else:
        if method == "substitute":
            yhat, pi_lower, pi_upper, info = _fit_substitute(y_al, X_al)
        elif method == "ols":
            yhat, pi_lower, pi_upper, info = _fit_ols(y_al, X_al)
        elif method == "huber":
            yhat, pi_lower, pi_upper, info = _fit_huber(y_al, X_al)
        elif method == "rolling":
            if window is None:
                raise ValueError(
                    "window must be provided for rolling regression (in samples)"
                )
            elif isinstance(window, str):
                raise TypeError(
                    "String window (e.g., '30D') no longer supported. Pass an integer number of samples."
                )
            else:
                window_n = int(window)
            yhat, pi_lower, pi_upper, info = _fit_rolling_regression(
                y_al, X_al, window=window_n
            )
        elif method == "lagged_reg":
            if lags is None:
                first_col = X_al.columns[0]
                max_lag = 6
                try:
                    lag_list = _suggest_lags(
                        *_mask_overlap(y_al, X_al[[first_col]]), max_lag=max_lag
                    )[:3]
                except Exception:
                    lag_list = [0]
            elif isinstance(lags, int):
                lag_list = list(range(0, int(lags) + 1))
            else:
                lag_list = list(map(int, lags))

            yhat, pi_lower, pi_upper, info = _fit_lagged_elasticnet(
                y_al, X_al, lags=lag_list
            )
        elif method == "loess":
            yhat, pi_lower, pi_upper, info = fit_loess_time_value(y_al, X_al)
        elif method == "dfm_trimbur_rw":
            yhat, pi_lower, pi_upper, info = _fit_dfm(
                y_al,
                X_al,
                factor="trimbur",
                anomaly_mode="rw",
                anom_var="target",
                rx_scale=1.0,
                maxiter=80,
                disp=0,
                params=params,
            )
        elif method == "dfm_trimbur_ar":
            yhat, pi_lower, pi_upper, info = _fit_dfm(
                y_al,
                X_al,
                factor="trimbur",
                anomaly_mode="ar",
                anom_var="neighbor",
                rx_scale=1.0,
                maxiter=80,
                disp=0,
                params=params,
            )
        elif method == "resid_interp_linear":
            yhat, pi_lower, pi_upper, info = _fit_resid_interp(
                y_al, X_al, kind="linear"
            )
        elif method == "resid_interp_pchip":
            yhat, pi_lower, pi_upper, info = _fit_resid_interp(y_al, X_al, kind="pchip")

        else:
            raise ValueError("Unknown method: %s" % method)

    # Merge predictions into target to create filled series
    filled = y_al.copy()
    missing_mask = filled.isna()
    filled[missing_mask] = yhat.reindex(filled.index)[missing_mask]

    # Clip bounds if provided
    lo, hi = bounds
    if lo is not None:
        filled = filled.clip(lower=lo)
    if hi is not None:
        filled = filled.clip(upper=hi)

    # Forward-chaining CV (optional, rough)
    metrics: Dict[str, float] = {}

    result = FillResult(
        filled=filled,
        yhat=yhat.reindex(filled.index),
        pi_lower=(None if pi_lower is None else pi_lower.reindex(filled.index)),
        pi_upper=(None if pi_upper is None else pi_upper.reindex(filled.index)),
        model_info=info,
        metrics=metrics,
    )
    return result.to_dict()


# ----------------------- Optional CSV writer with YAML ---------------------- #


def write_filled_csv_with_yaml_header(
    filled: pd.Series,
    path: str,
    model_info: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    float_format: str = "{:.6g}",
):
    """Write a CSV file with a YAML-like header as #-comments.

    Parameters
    ----------
    filled : pd.Series
        Series to write; index must be a DatetimeIndex.
    path : str
        Destination filepath.
    model_info : dict
        Metadata from fill_from_neighbor; will be serialized.
    metrics : dict, optional
        Metrics to include.
    extra_meta : dict, optional
        Any additional metadata.
    float_format : str
        Format string for values.
    """
    if not isinstance(filled.index, pd.DatetimeIndex):
        raise TypeError("filled.index must be a DatetimeIndex")
    meta = {"model_info": model_info}
    if metrics:
        meta["metrics"] = metrics
    if extra_meta:
        meta.update(extra_meta)
    # Simple YAML-ish dump
    import json

    header_lines = ["# --- neighbor_fill metadata ---"]
    header_lines.append("# " + json.dumps(meta, default=str))
    header = "\n".join(header_lines) + "\n"
    # Build CSV body
    df = filled.to_frame(name="value")
    df.index.name = "datetime"
    csv_body = df.to_csv(date_format="%Y-%m-%dT%H:%M:%S", float_format=None)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(csv_body)
