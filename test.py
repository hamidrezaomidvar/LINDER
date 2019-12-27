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

path_out = Path("/Users/sunt05/Downloads/20191221test-lc/xx/")
path_raster = Path(
    "/Users/sunt05/Downloads/20191221test-lc/par/predicted_tiff/prediction-eopatch_0-pic_0.tiff"
)
name_job = "0-0"
name_v1 = f"predict_GUF_roads_mod{name_job}"
name_v2 = f"grid{name_job}"
name_out = f"test-out{name_job}"

# ld.merge_vector_data(path_out, path_raster, name_v1, name_v2)
path_v1 = Path("/Users/sunt05/Downloads/xxx1/predict_GUF_roads_mod0-0/")
path_v2 = Path("/Users/sunt05/Downloads/xxx1/grid0-0/")

# ld.grass_overlay(path_v1,path_v2,path_raster)

start = time()
# p = Pool()
#
# list_prm_default = [path_v1, path_v2, path_raster]
# list_gdf = p.starmap(
#     ld.grass_overlay,
#     zip(
#         sorted(list(path_v1.parent.glob("predict_GUF_mod*"))),
#         sorted(list(path_v1.parent.glob("roads_*"))),
#         sorted(list(path_raster.parent.glob("prediction*"))),
#     ),
# )


list_path_fraction = ld.get_land_cover(
    51.515070,
    -0.008555,
    51.489564,
    0.034932,
    "2017-01-01",
    "2017-10-01",
    path_GUF="/Users/sunt05/Dropbox/6.Repos/land_cover/Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326",
    path_data_building="ss",
    path_save=Path("~/Downloads/xxx1").expanduser(),
    #     download_img=False,
)

end = time()

print(f'\n time spent: {end - start:.2f}')
