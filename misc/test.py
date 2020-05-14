import sys
import pandas as pd
from pathlib import Path
import linder as ld

# list_path_fraction = ld.get_land_cover(
#     51.515070,
#     -0.008555,
#     51.489564,
#     0.034932,
#     "2017-01-01",
#     "2017-05-01",
#     path_GUF="/Users/sunt05/Dropbox/6.Repos/land_cover/Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326",
#     path_data_building="ss",
#     path_save="par",
# #     download_img=False,
# )

from pathlib import Path
from time import time

from multiprocessing import Pool
from itertools import repeat

from linder.task_util import process_overlap, find_overlap

path_out = Path("/Users/sunt05/Downloads/20191221test-lc/xx/")
path_raster = Path(
    "/Users/sunt05/Downloads/20191221test-lc/par/predicted_tiff/prediction-eopatch_0-pic_0.tiff"
)
name_job = "0-0"
name_v1 = f"predict_GUF_roads_mod_{name_job}"
name_v2 = f"grid_{name_job}"
name_out = f"test_out_{name_job}"

path_save = Path("/Users/sunt05/Downloads/LC-London")
path_GUF = Path(
    "/Users/sunt05/Dropbox/6.Repos/land_cover/Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326"
)

# ld.merge_vector_data(path_out, path_raster, name_v1, name_v2)
path_v1 = Path("/Users/sunt05/Downloads/xxx1/predict_GUF_roads_mod0-0/")
path_v2 = Path("/Users/sunt05/Downloads/xxx1/grid0-0/")

# ld.grass_overlay(path_v1,path_v2,path_raster)

t_start = time()

# London
lat, lon = 51.515070,-0.008555
start, end = "20160101", "20171231"

# new site
# lat, lon = 28.1086, 112.7864
# start, end = "20160101", "20171231"

# old site
# lat, lon = 28.13, 112.75
# start, end = "20100101", "20141231"

delta_scale = 0.02

list_path_fraction = ld.get_land_cover(
    lat + delta_scale,
    lon - delta_scale,
    lat - delta_scale,
    lon + delta_scale,
    start,
    end,
    path_GUF=path_GUF,
    # path_GUF=None,
    path_save=path_save,
)

# patch_n = 0
# for pic_n in range(9):
#     overlap_found = find_overlap(path_save, patch_n, pic_n, path_GUF)
#     print(f"{pic_n},overlap_found:{overlap_found}")
#     # process_overlap(
#     #     path_save=path_save, path_GUF=path_GUF, patch_n=0, pic_n=pic_n,
#     # )

t_end = time()

print(f"\n time spent: {t_end - t_start:.2f}")
path_save
