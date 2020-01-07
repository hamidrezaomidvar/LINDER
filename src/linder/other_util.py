import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from eolearn.core import (
    EOExecutor,
    EOPatch,
    FeatureType,
    LinearWorkflow,
    OverwritePermission,
    SaveToDisk,
)
from eolearn.io import S2L1CWCSInput
from eolearn.mask import (
    AddCloudMaskTask,
    AddValidDataMaskTask,
    get_s2_pixel_cloud_detector,
)
from sentinelhub import CRS, BBox, CustomUrlParam, config

from .sent_util import (
    CountValid,
    EuclideanNorm,
    NormalizedDifferenceIndex,
    SentinelHubValidData,
)


def check_sentinel_cfg():
    dict_sc = config.SHConfig().get_config_dict()
    str_id = dict_sc["instance_id"]
    if str_id:
        # print(f"instance_id `{str_id}` is found.")
        pass
    else:
        list_str_info = [
            "sentinelhub has NOT been set up with a valid `instance_id`:",
            "please set it up following",
            "https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html#Requirements.",
        ]
        raise RuntimeError("\n".join(list_str_info))


def download_data(
        path_save, coords_top, coords_bot, patch_n, s_date, e_date, debug=False
):
    # before moving onto actual tasks, check setup
    check_sentinel_cfg()

    [lat_left_top, lon_left_top] = coords_top
    [lat_right_bot, lon_right_bot] = coords_bot
    # TASK FOR BAND DATA
    # add a request for B(B02), G(B03), R(B04), NIR (B08), SWIR1(B11), SWIR2(B12)
    # from default layer 'ALL_BANDS' at 10m resolution
    # Here we also do a simple filter of cloudy scenes. A detailed cloud cover
    # detection is performed in the next step
    custom_script = "return [B02, B03, B04, B08, B11, B12];"
    add_data = S2L1CWCSInput(
        layer="BANDS-S2-L1C",
        feature=(FeatureType.DATA, "BANDS"),  # save under name 'BANDS'
        # custom url for 6 specific bands
        custom_url_params={CustomUrlParam.EVALSCRIPT: custom_script},
        resx="10m",  # resolution x
        resy="10m",  # resolution y
        maxcc=0.1,  # maximum allowed cloud cover of original ESA tiles
    )

    # TASK FOR CLOUD INFO
    # cloud detection is performed at 80m resolution
    # and the resulting cloud probability map and mask
    # are scaled to EOPatch's resolution
    cloud_classifier = get_s2_pixel_cloud_detector(
        average_over=2, dilation_size=1, all_bands=False
    )
    add_clm = AddCloudMaskTask(
        cloud_classifier,
        "BANDS-S2CLOUDLESS",
        cm_size_y="80m",
        cm_size_x="80m",
        cmask_feature="CLM",  # cloud mask name
        cprobs_feature="CLP",  # cloud prob. map name
    )

    # TASKS FOR CALCULATING NEW FEATURES
    # NDVI: (B08 - B04)/(B08 + B04)
    # NDWI: (B03 - B08)/(B03 + B08)
    # NORM: sqrt(B02^2 + B03^2 + B04^2 + B08^2 + B11^2 + B12^2)
    ndvi = NormalizedDifferenceIndex("NDVI", "BANDS/3", "BANDS/2")
    ndwi = NormalizedDifferenceIndex("NDWI", "BANDS/1", "BANDS/3")
    norm = EuclideanNorm("NORM", "BANDS")

    # TASK FOR VALID MASK
    # validate pixels using SentinelHub's cloud detection mask and region of acquisition
    add_sh_valmask = AddValidDataMaskTask(
        SentinelHubValidData(), "IS_VALID"  # name of output mask
    )

    # TASK FOR COUNTING VALID PIXELS
    # count number of valid observations per pixel using valid data mask
    count_val_sh = CountValid(
        "IS_VALID", "VALID_COUNT"  # name of existing mask  # name of output scalar
    )

    # TASK FOR SAVING TO OUTPUT (if needed)
    path_save = Path(path_save)
    path_save.mkdir(exist_ok=True)
    # if not os.path.isdir(path_save):
    #     os.makedirs(path_save)
    save = SaveToDisk(
        path_save, overwrite_permission=OverwritePermission.OVERWRITE_PATCH
    )

    # Define the workflow
    workflow = LinearWorkflow(
        add_data, add_clm, ndvi, ndwi, norm, add_sh_valmask, count_val_sh, save
    )
    # Execute the workflow

    # time interval for the SH request
    # TODO: need to check if specified time interval is valid
    time_interval = [s_date, e_date]

    # define additional parameters of the workflow
    execution_args = []

    path_EOPatch = path_save / f"eopatch_{patch_n}"

    execution_args.append(
        {
            add_data: {
                "bbox": BBox(
                    ((lon_left_top, lat_left_top), (lon_right_bot, lat_right_bot)),
                    crs=CRS.WGS84,
                ),
                "time_interval": time_interval,
            },
            save: {"eopatch_folder": path_EOPatch.stem},
        }
    )

    executor = EOExecutor(workflow, execution_args, save_logs=True)
    if debug:
        print("Downloading Satellite data ...")

    executor.run(workers=2, multiprocess=False)
    if executor.get_failed_executions():
        raise RuntimeError("EOExecutor failed in finishing tasks!")

    if debug:
        executor.make_report()
    if debug:
        print("Satellite data is downloaded")
    return path_EOPatch


def save_images(path_EOPatch: Path, patch_n: int, scale):
    # Draw the RGB image
    size = 20
    # (Path(path_out) / f"eopatch_{patch_n}").mkdir(exist_ok=True)
    eopatch = EOPatch.load(path_EOPatch, lazy_loading=True)
    path_dir_image = path_EOPatch.parent / "images" / f"patch_{patch_n}"
    path_dir_image.mkdir(parents=True, exist_ok=True)
    # if not os.path.isdir(path_dir_image):
    #     os.makedirs(path_dir_image)

    print(f"saving the images into {path_dir_image} ...")

    list_timestamp = eopatch.timestamp
    list_path_image = []
    for i, timestamp in enumerate(list_timestamp):
        # replace `:` with `_` to avoid path issue on Windows
        str_timestamp = timestamp.isoformat().replace(':','_')

        fig = plt.figure(figsize=(size * 1, size * scale))
        ax = plt.subplot(1, 1, 1)
        plt.imshow(np.clip(eopatch.data["BANDS"][i][..., [2, 1, 0]] * 3.5, 0, 1))
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect("auto")
        fn = f"{str_timestamp}.png"
        path_img = path_dir_image / fn
        print(f"Saving {path_img}")
        plt.savefig(path_img)
        plt.close()
        list_path_image.append(path_img)

    return list_path_image
