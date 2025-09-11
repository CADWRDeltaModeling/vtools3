"""
Column naming alignment utilities for time series composition functions.

This module provides decorators that standardize how functions like
``ts_merge``, ``ts_splice``, and ``transition_ts`` handle their ``names``
argument and enforce column consistency across multiple time series inputs.

Main features
-------------
- **Column consistency enforcement**:
  Ensures that when ``names=None`` (default), all input DataFrames share
  identical columns. This prevents accidental creation of staggered or
  mismatched columns.

- **Centralized naming behavior**:
  Applies uniform handling of ``names`` values:

  * ``None`` — require identical columns across all inputs and keep them.
  * ``str`` — require univariate inputs (single column each); output is
    a single-column DataFrame (or Series if all inputs were Series) with
    this name.
  * ``Iterable[str]`` — treated as a column selector: these columns are
    selected (and ordered) from the final output and must exist in every
    input.

- **Support for both list-style and pairwise APIs**:
  Works for functions that accept a sequence of time series (like
  ``ts_merge``/``ts_splice``) or two explicit series arguments
  (like ``transition_ts``).

Usage pattern
-------------
Decorate your functions as follows::

    @columns_aligned(mode="same_set")
    @names_aligned(seq_arg=0, pre_rename=True)
    def ts_splice(series, names=None, ...):
        ...

    @columns_aligned(mode="same_set")
    @names_aligned_pair(ts0_kw="ts0", ts1_kw="ts1")
    def transition_ts(ts0, ts1, names=None, ...):
        ...

This ensures consistent semantics for all multi-series combination tools.
"""

# colname_align.py
import pandas as pd
from functools import wraps
from inspect import signature

# module level error so test can access it
ERR_MULTI_NAMES_SERIES = "Cannot assign multiple names to a Series; pass a single name."


def align_names(result, names):
    if not names:
        return result

    # Series case
    if isinstance(result, pd.Series):
        if isinstance(names, str):
            result = result.copy()
            result.name = names
            return result
        elif hasattr(names, "__iter__"):
            lst = list(names)
            if len(lst) != 1:
                raise ValueError(ERR_MULTI_NAMES_SERIES)
            result = result.copy()
            result.name = lst[0]
            return result
        return result  # any other type: no-op

    # DataFrame case
    if isinstance(names, str):
        return result.rename(columns={result.columns[0]: names})
    elif hasattr(names, "__iter__"):
        return result[list(names)]
    return result


def _coerce_inputs_strict(seq, names):
    """
    Strict input alignment policy:
    - names is None  -> all inputs must have identical column lists (no unions/intersections).
    - names is str   -> leave inputs as-is; final renaming happens via align_names(...).
    - names is list  -> for each DF, select exactly those columns; for a Series, only len==1 allowed.
    """
    out = []

    if names is None:
        # Promote Series->DataFrame for apples-to-apples checks
        tmp = [s.to_frame(name=s.name) if isinstance(s, pd.Series) else s for s in seq]

        # 1) Same number of columns across all inputs
        ncols0 = tmp[0].shape[1] if isinstance(tmp[0], pd.DataFrame) else 1
        for t in tmp[1:]:
            ncols = t.shape[1] if isinstance(t, pd.DataFrame) else 1
            if ncols != ncols0:
                raise ValueError(
                    "All inputs must have the same number of columns when `names` is None."
                )

        # 2) Exact column-name equality (order matters) if DataFrames
        if isinstance(tmp[0], pd.DataFrame):
            cols0 = list(tmp[0].columns)
            for t in tmp[1:]:
                if not isinstance(t, pd.DataFrame) or list(t.columns) != cols0:
                    raise ValueError(
                        "All input columns must be identical when `names` is None"
                    )

        return seq  # keep original types; they already match strictly

    # colname_align.py  (_coerce_inputs_strict)
    elif isinstance(names, str):
        # If ALL inputs are univariate, pre-rename their single column to `names`
        def is_uni(x):
            return (isinstance(x, pd.Series)) or (
                isinstance(x, pd.DataFrame) and x.shape[1] == 1
            )

        if all(is_uni(s) for s in seq):
            out = []
            for s in seq:
                if isinstance(s, pd.Series):
                    out.append(s.rename(names))
                else:  # 1-col DataFrame
                    only = s.columns[0]
                    out.append(s.rename(columns={only: names}))
            return out
        # Otherwise leave inputs as-is; final rename happens on the output
        return seq

    else:
        # Iterable of names: enforce and select exactly these columns
        req = list(names)
        if not req:
            raise ValueError("`names` selection is empty.")
        for s in seq:
            if isinstance(s, pd.DataFrame):
                missing = set(req) - set(s.columns)
                if missing:
                    raise ValueError(
                        f"DataFrame missing requested columns: {sorted(missing)}"
                    )
                out.append(s[req])
            else:  # Series
                if len(req) != 1:
                    raise ValueError(ERR_MULTI_NAMES_SERIES)  # <-- updated
                out.append(s.rename(req[0]))
        return out


def align_inputs_strict(seq_arg=0, names_kw="names"):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            sig = signature(fn)
            param_name = list(sig.parameters)[seq_arg]  # e.g., "series"
            names = kwargs.get(names_kw, None)

            # read sequence regardless of positional/keyword
            seq = kwargs[param_name] if param_name in kwargs else args[seq_arg]

            # >>> Early passthrough on bad/empty input so the wrapped fn raises its own error <<<
            if not isinstance(seq, (list, tuple)) or len(seq) == 0:
                # ts_merge/ts_splice keep their original messages:
                # - ts_merge: "`series` must be a non-empty tuple or list"  (test expects this)
                # - ts_splice: "`series` must be a non-empty tuple or list of pandas.Series or pandas.DataFrame."
                return fn(*args, **kwargs)

            # strict coercion only for non-empty sequences
            seq2 = _coerce_inputs_strict(seq, names)

            # write back
            if param_name in kwargs:
                kwargs = dict(kwargs)
                kwargs[param_name] = seq2
            else:
                args = list(args)
                args[seq_arg] = seq2
                args = tuple(args)

            out = fn(*args, **kwargs)
            return align_names(out, names)

        return wrapper

    return deco


def align_inputs_pair_strict(ts0_kw="ts0", ts1_kw="ts1", names_kw="names"):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            names = kwargs.get(names_kw, None)

            # accept positional or keyword
            from inspect import signature

            sig = signature(fn)
            param_names = list(sig.parameters)
            # defaults: try kwargs first, else positional fallback
            ts0 = kwargs.get(
                ts0_kw,
                args[param_names.index(ts0_kw)] if ts0_kw in param_names else args[0],
            )
            ts1 = kwargs.get(
                ts1_kw,
                args[param_names.index(ts1_kw)] if ts1_kw in param_names else args[1],
            )

            ts0_new, ts1_new = _coerce_inputs_strict([ts0, ts1], names)

            # write back
            if ts0_kw in kwargs:
                kwargs = dict(kwargs)
                kwargs[ts0_kw] = ts0_new
            else:
                args = list(args)
                idx0 = param_names.index(ts0_kw) if ts0_kw in param_names else 0
                args[idx0] = ts0_new
                args = tuple(args)

            if ts1_kw in kwargs:
                kwargs = dict(kwargs)
                kwargs[ts1_kw] = ts1_new
            else:
                args = list(args)
                idx1 = param_names.index(ts1_kw) if ts1_kw in param_names else 1
                args[idx1] = ts1_new
                args = tuple(args)

            out = fn(*args, **kwargs)
            return align_names(out, names)

        return wrapper

    return deco
