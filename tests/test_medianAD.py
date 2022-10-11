# -*- coding: utf-8 -*-
"""

"""


import pytest


from vtools.functions.medianAD import *

import matplotlib.pyplot as plt


def test_MedianAD():
    
    medianAD = get_medianAD(level=8,scale=None,filter_len=7,quantiles=(0.02,0.98))

   
if __name__ == "__main__":
    test_MedianAD()