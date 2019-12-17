# version info for supy

# from supy_driver import __version__ as sd_ver
from ._env import path_module

import pandas as pd

ser_ver = pd.read_json(path_module / "lucifer_version.json", typ="series")


__version__ = f"{ser_ver.ver_milestone}.{ser_ver.ver_major}.{ser_ver.ver_minor}{ser_ver.ver_remark}"



def show_version():
    """print `lucifer` version information.
    """
    print(f"lucifer: {__version__}")
