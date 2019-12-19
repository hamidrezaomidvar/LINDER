###########################################################################
# linder - Land Use ClaSsIFiER
# Authors:
# Hamidreza Omidvar, h.omidvar@reading.ac.uk
# Ting Sun, ting.sun@reading.ac.uk
# History:
# ?? 2019: first alpha release
# 16 Dec 2019: pypi packaging
###########################################################################


# core functions
from ._linder_module import (
    get_land_cover
)


# utilities
# from . import util


# version info
from ._version import show_version, __version__


# module docs
__doc__ = """
linder - Land cover INDexER
=====================================

linder is a machine-learning based land cover indexer using Sentinel imagery.

"""
