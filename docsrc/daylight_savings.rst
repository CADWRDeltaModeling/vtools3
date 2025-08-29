.. _dst_st:

Daylight Savings Time Conversion
================================

The ``dst_st`` function provides robust conversion of a pandas Series or DataFrame
with a naive DatetimeIndex that observes daylight savings time (DST) to a fixed
standard time zone (e.g., PST) using POSIX conventions.

This is useful for hydrology, environmental, and other time series applications
where a consistent standard time is required for analysis or reporting.

**Function reference:**

.. autofunction:: vtools.data.dst.dst_st

See also the :ref:`API documentation <dst_st>` for details and usage examples.