import os
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .fraction_util import calculate_fraction
from .other_util import download_data, save_images
from .predict_util import predict_raster_patch
from .task_util import download_OSM_box, predict_shape


def get_land_cover(
        lat_left_top_t,
        lon_left_top_t,
        lat_right_bot_t,
        lon_right_bot_t,
        s_date,
        e_date,
        path_GUF,
        path_save,
        nx=1,
        ny=1,
        xn=20,
        yn=20,
        path_data_building=None,
        download_img=True,
        debug=False,
):
    # size = 10  # display parameter. Do not change
    scale = abs((lat_left_top_t - lat_right_bot_t) / (lon_left_top_t - lon_right_bot_t))

    # use `no` to force skip building merging
    Building_data = "no"

    # use `OSM` to force download OSM data in `other_tasks`
    Road_data = "OSM"

    # cast to Path
    if path_GUF:
        path_GUF = Path(path_GUF)
        print(f'path_GUF: {path_GUF}')
        # check if GUF data existing
        list_tif_GUF=list(path_GUF.glob('*tif'))
        if len(list_tif_GUF)==0:
            raise RuntimeError(f'No tiff images found in the specified GUF path:\n {path_GUF.resolve().as_posix()}')
    path_save = Path(path_save)
    # list_of_GUF = list(path_GUF.glob("*tif"))

    # path_save = cname
    all_lats = np.linspace(lat_right_bot_t, lat_left_top_t, num=ny + 1)
    all_lons = np.linspace(lon_left_top_t, lon_right_bot_t, num=nx + 1)
    patch_n = 0

    # TODO: the below can be done in parallel
    list_path_fraction = []
    for i in range(0, len(all_lons) - 1):
        lon_left_top, lon_right_bot = [all_lons[i], all_lons[i + 1]]
        for j in range(0, len(all_lats) - 1):
            lat_right_bot, lat_left_top = [all_lats[j], all_lats[j + 1]]

            coords_top = [lat_left_top, lon_left_top]
            coords_bot = [lat_right_bot, lon_right_bot]

            # download sentinel images
            if download_img:
                path_EOPatch = download_data(
                    path_save, coords_top, coords_bot, patch_n, s_date, e_date, debug
                )
            else:
                path_EOPatch = path_save / f"eopatch_{patch_n}"
                if not path_EOPatch.exists():
                    raise RuntimeError(
                        "\n".join(
                            [
                                f"{path_EOPatch} does NOT exist!",
                                " set `download_img=True` to download Sentinel images.",
                            ]
                        )
                    )

            # save images as local files
            list_path_image = save_images(path_EOPatch, patch_n, scale)

            # index land cover by prediction
            list_path_raster = predict_raster_patch(path_EOPatch, patch_n, scale, debug)

            if Road_data != "no":
                download_OSM_box(lat_left_top, lat_right_bot, lon_right_bot, lon_left_top, debug)

            # # other tasks
            # list_path_shp = [
            #     predict_shape(path_raster_predict, lat_left_top, lat_right_bot, lon_left_top, lon_right_bot, path_GUF,
            #                   Road_data, Building_data, path_data_building)
            #     for path_raster_predict in list_path_raster
            # ]
            #
            # # print("\nworking on land cover fraction calculation:")
            # for path_shp, path_raster in zip(list_path_shp, list_path_raster):
            #     # print(f"shape file: {path_shp}")
            #     # print(f"raster file: {path_raster}")
            #     path_fraction = calculate_fraction(path_shp, path_raster, xn, yn)
            #     list_path_fraction.append(path_fraction)
            # n_worker = int(os.cpu_count() / 2 - 1)
            p = Pool()
            # merge vector layers
            list_prm_default = [
                lat_left_top,
                lat_right_bot,
                lon_left_top,
                lon_right_bot,
                path_GUF,
                Road_data,
                Building_data,
                path_data_building,
                debug,
            ]
            list_path_shp = p.starmap(
                predict_shape,
                zip(list_path_raster, *[repeat(prm) for prm in list_prm_default]),
            )

            # calculate land cover fractions
            list_prm_default = [xn, yn, debug]
            list_path_fraction += p.starmap(
                calculate_fraction,
                zip(
                    list_path_shp,
                    list_path_raster,
                    *[repeat(prm) for prm in list_prm_default],
                ),
            )

            patch_n = patch_n + 1

    return list_path_fraction
