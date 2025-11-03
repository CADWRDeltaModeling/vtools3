"""Top-level package for vtools. Most functions can be directly imported from vtools namespace"""

__author__ = """Eli Ateljevich, Nicky Sandhu, Kijin Nam"""
__email__ = "Eli.Ateljevich@water.ca.gov; Kijin.Nam@water.ca.gov"

import os

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python <3.8, use importlib_metadata backport
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("vtools")
except PackageNotFoundError:
    # Fallback for running from a VCS checkout without installation
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:
        __version__ = "unknown"

from vtools.data.gap import *
from vtools.data.vtime import *
from vtools.data.timeseries import *
from vtools.data.dst import *

from vtools.functions.transition import *
from vtools.functions.climatology import *
from vtools.functions.interannual import *
from vtools.functions.filter import *
from vtools.functions.error_detect import *
from vtools.functions.merge import *
from vtools.functions.interpolate import *
from vtools.functions.lag_cross_correlation import *
from vtools.functions.envelope import *
from vtools.functions.neighbor_fill import *
