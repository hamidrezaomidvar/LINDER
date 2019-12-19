# version info for linder

from ._env import path_module

import pandas as pd

ser_ver = pd.read_json(path_module / "linder_version.json", typ="series")


__version__ = f"{ser_ver.ver_milestone}.{ser_ver.ver_major}.{ser_ver.ver_minor}{ser_ver.ver_remark}"



def show_version():
    """print `linder` version information.
    """
    print(f"linder: {__version__}")
