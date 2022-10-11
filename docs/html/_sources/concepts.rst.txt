
Concepts
--------


time-series
^^^^^^^^^^^
A time series is an xarray or pandas



..__time_interval

Intervals
^^^^^^^^^

Intervals in vtools are equivalent to `freq` arguments in Pandas. In some cases strings or timeDelta objects can be used, but
the one that behaves best in math operations and the main implementation class in Pandas
is the `Offset <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ class.

One way to construct intervals quickly are the :func:`vtools.seconds` :func:`vtools.minutes` etc convenience functions.
