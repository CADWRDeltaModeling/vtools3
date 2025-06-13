"""Top-level package for vtools. Most functions can be directly imported from vtools namespace"""

__author__ = """Eli Ateljevich, Nicky Sandhu, Kijin Nam"""
__email__ = 'eli@water.ca.gov'
__version__ = '3.0.4'

from . import _version
__version__ = _version.get_versions()['version']

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

