VTools Documentation 
====================
VTools represents a toolset for data acquisition and downloading that the Delta Modeling Section uses frequently as base functionality for other tools or for analysis. The package contents includes standalone scripts and pandas/numpy-based utilities. Recently we retooled vtools to catch up with Python 3, switched over to Pandas (and soon xarray) data structures, which means that all the data structures for time series are gone and the package is leaner than before. For data provision, we have moved HEC-DSS support completely out of the package, and provided helpers for downloading and reading sources such as NOAA, USGS (NWIS) and (coming soon) NetCDF formats. The functional part includes interpolation, filtering, merging etc that augment Pandas or make its functionality accessible with less work. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <installation>
   Introduction and concepts <concepts>
   Interpolation <interpolation>
   Averaging and Filtering <notebooks/filters.ipynb>
   Downloading scripts (standalone) <download>
   API Documentation <modules>
   Contributing <contributing>
   Authors <authors>
   History <history>

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
