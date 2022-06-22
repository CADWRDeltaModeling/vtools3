.. _Interpolation
Interpolation
-------------

There are several vtools functions for interpolation, which complement Pandas. The main
contributions are:

 * The function :func:`vtools.functions.interpolate.interpolate_to_index` 
   interpolates from one time series to another index based on time. This is particularly
   useful when the destination times are irregular or clipped. 
   
 * The function :func:`vtools.functions.interpolate.mspline` provides a local spline that is exact, monotonicity/shape preserving and
   accurate. 
   
 * The function :func:`vtools.functions.interpolate.rhistinterp` is a spline with tension that is useful
   for interpolating PeriodIndex data to a regular, DatetimeIndex while maintaining integral
   quantities (volumes) within each period (gaussian smoothing can be used but it only maintains global mass).
    
 * The helper function gap_size which can provide the size of gaps (in time or number of steps). 


