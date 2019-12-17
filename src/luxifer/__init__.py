###########################################################################
# luxifer - Land Use ClaSsIFiER
# Authors:
# Hamidreza Omidvar, h.omidvar@reading.ac.uk
# Ting Sun, ting.sun@reading.ac.uk
# History:
# ?? 2019: first alpha release
# 16 Dec 2019: pypi packaging
###########################################################################


# core functions
from ._luxifer_module import (
    get_land_cover
)


# utilities
from . import util


# version info
from ._version import show_version, __version__


# module docs
__doc__ = """
luxifer - Land Use ClaSsIFiER
=====================================

luxifer is a machine-learning based land use/land cover (LULC) classifier using Sentinel imagery.

"""
