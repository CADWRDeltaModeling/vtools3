"""Top-level package for vtools. Most functions can be directly imported from vtools namespace"""

__author__ = """Eli Ateljevich, Nicky Sandhu, Kijin Nam"""
__email__ = "Eli.Ateljevich@water.ca.gov; Kijin.Nam@water.ca.gov"

import os
import sys
from importlib.metadata import version, PackageNotFoundError

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"  # fallback for weird dev cases

from vtools.data.gap import *
from vtools.data.vtime import *
from vtools.data.timeseries import *

from vtools.functions.transition import *
from vtools.functions.climatology import *
from vtools.functions.interannual import *
from vtools.functions.filter import *
from vtools.functions.error_detect import *
from vtools.functions.merge import *
from vtools.functions.interpolate import *
from vtools.functions.lag_cross_correlation import *
from vtools.functions.envelope import *
