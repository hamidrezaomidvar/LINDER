import ast
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
import rasterio.mask
import requests
from rasterio.features import shapes
from rasterio.merge import merge
from shapely.geometry import box

from ._env import path_module
from .clip_util import clip_shp
from .grass_util import grass_overlay


def find_overlap(path_out, patch_n, pic_n, path_GUF):
    path_shp = Path(f"shape_box_{patch_n}-{pic_n}.shp")
    with fiona.open(path_out / path_shp.stem, "r", ) as shapefile:
        features = [feature["geometry"] for feature in shapefile]

    # list_of_GUF = list(path_GUF.glob("*tif"))
    overlap_found = 0
    for GUF_data_dir in path_GUF.glob("*tif"):

        try:
            with rasterio.open(GUF_data_dir) as src:
                out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
                out_meta = src.meta.copy()
            overlap_found += 1

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            with rasterio.open(
                    path_out / f"masked_footprint_{patch_n}-{pic_n}-{overlap_found}.tif",
                    "w",
                    **out_meta,
            ) as dest:
                dest.write(out_image)

        except:
            pass
    return overlap_found


def merge_overlap(overlap_found, path_merge):
    all_files = [
        path_merge.parent / (path_merge.stem + f"-{i}.tif")
        for i in range(1, overlap_found + 1)
    ]
    src_files_to_mosaic = [rasterio.open(fp) for fp in all_files]

    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )

    # path_merge = path_out / f"masked_footprint{patch_n}-{pic_n}.tif"

    with rasterio.open(path_merge, "w", **out_meta) as dest:
        dest.write(mosaic)

    return path_merge


# def other_tasks(
#         path_out,
#         path_GUF,
#         Building_data,
#         Road_data,
#         building_dir,
#         lat_left_top,
#         lon_left_top,
#         lat_right_bot,
#         lon_right_bot,
# ):
#     # cast to Path
#     path_out = Path(path_out)
#
#     # get file list of predicted sentinel images
#     list_path_raster = sorted(list((path_out / "predicted_tiff").glob("*/*/*tiff")))
#
#     # index land use of each tiff image
#     list_path_shp_merge = [
#         predict_shape(path_raster_predict, lat_left_top, lat_right_bot, lon_left_top, lon_right_bot, path_GUF,
#                       Road_data, Building_data, building_dir)
#         for path_raster_predict in list_path_raster
#     ]
#     return list_path_shp_merge


def predict_shape(path_raster_predict, lat_left_top, lat_right_bot, lon_left_top, lon_right_bot, path_GUF,
                  Road_data="OSM", Building_data="no", path_data_building=None, debug=False):
    if debug:
        print("\n")
        print(f"predicting shapefile using {path_raster_predict} ...")

    # derive root path for output
    path_out = path_raster_predict.parent.parent

    # parse `patch_n` and `pic_n` from `path_raster_predict`
    str_patch_n, str_pic_n = path_raster_predict.stem.split("-")[1:]
    patch_n = int(str_patch_n.split("_")[-1])
    pic_n = int(str_pic_n.split("_")[-1])

    raster_predict = rasterio.open(path_raster_predict)
    df = raster_predict.bounds
    sh_box = box(df.left, df.bottom, df.right, df.top)
    shape_box = gpd.GeoDataFrame({"geometry": sh_box, "col": [np.nan]})
    # shape_box.crs = {"init": "epsg:4326"}
    shape_box.crs = {"init": "epsg:4326"}

    # path for saving shape_box
    path_shp = Path(f"shape_box_{patch_n}-{pic_n}.shp")
    path_shp = path_out / path_shp.stem / path_shp
    shape_box.to_file(path_shp.parent)
    box_domain = gpd.read_file(path_shp)

    if debug:
        print("Converting the predicted tiff file to shapefile ...")

    path_shp_predict = tif2shp(
        path_raster_predict,
        "predicted",
        "predicted_shape",
        path_out,
        box_domain,
        patch_n,
        pic_n,
    )
    if Building_data != "no":
        clip_buildings(
            Building_data, box_domain, path_data_building, path_out, patch_n, pic_n
        )
    if Road_data != "no":
        download_OSM_road(lat_left_top, lat_right_bot, lon_right_bot, lon_left_top, path_out, patch_n, pic_n)
    if path_GUF:
        if debug:
            print("Clipping the GUF data ...")
        path_footprint_masked = process_overlap(path_out, path_GUF, patch_n, pic_n)

        if debug:
            print("Converting clipped GUF tiff to shapefile ...")

        path_shp_footprint = tif2shp(
            path_footprint_masked,
            "GUF",
            "urban_foot_shape",
            path_out,
            box_domain,
            patch_n,
            pic_n,
        )

        gdf_GUF = merge_GUF(path_out, path_raster_predict, patch_n, pic_n)

        if Building_data == "OSM":
            merge_OSM(
                gdf_GUF,
                lat_left_top,
                lat_right_bot,
                lon_left_top,
                lon_right_bot,
                path_out,
                patch_n,
            )
    if path_GUF:
        if Building_data != "no" and Road_data == "no":
            if debug:
                print("Merging the predicted-GUF to Building data ...")
            # key names
            name_v1 = f"predict_GUF_mod_{patch_n}-{pic_n}"
            name_v2 = f"{Building_data}_sh_clipped_{patch_n}-{pic_n}"

            var_use = "a_LC"
            list_rule = [
                ("b_build", 1, 4),
            ]
            list_var_drop = ["a_cat", "b_cat", "a_LC", "b_build"]
            str_fn_out = f"predict_GUF_{Building_data}_mod_{patch_n}-{pic_n}"

            path_shp_merge = predict_vector(
                list_rule,
                list_var_drop,
                name_v1,
                name_v2,
                path_out,
                path_raster_predict,
                str_fn_out,
                var_use,
            )

        if Road_data != "no":
            # if Building_data == "no":
            if debug:
                print("Merging the predicted-GUF to Road data ...")
            # key names
            name_v1 = f"predict_GUF_mod_{patch_n}-{pic_n}"
            name_v2 = f"roads_{patch_n}-{pic_n}"

            var_use = "a_LC"
            list_rule = [
                ("b_cat", 1, 5),
                ("LC", 0, 4),
                ("LC", 5, 0),
            ]
            list_var_drop = ["cat", "a_cat", "b_cat", "a_LC", "b_FID"]
            str_fn_out = f"predict_GUF_roads_mod_{patch_n}-{pic_n}"

            path_shp_merge = predict_vector(
                list_rule,
                list_var_drop,
                name_v1,
                name_v2,
                path_out,
                path_raster_predict,
                str_fn_out,
                var_use,
            )
            if Building_data != "no":
                if debug:
                    print("Merging the predicted-GUF-roads to Building data ...")

                # key names
                name_v1 = f"predict_GUF_roads_mod_{patch_n}-{pic_n}"
                name_v2 = f"{Building_data}_sh_clipped_{patch_n}-{pic_n}"

                var_use = "a_LC"
                list_rule = [
                    ("a_LC", 4, 5),
                    ("b_build", 1, 4),
                ]
                list_var_drop = ["cat", "a_cat", "b_cat", "a_LC", "b_build"]
                str_fn_out = f"predict_GUF_roads_{Building_data}_mod_{patch_n}-{pic_n}"

                path_shp_merge = predict_vector(
                    list_rule,
                    list_var_drop,
                    name_v1,
                    name_v2,
                    path_out,
                    path_raster_predict,
                    str_fn_out,
                    var_use,
                )

    else:
        if Building_data != "no" and Road_data == "no":
            if debug:
                print("Merging the predicted to Building data ...")
            # key names
            name_v1 = f"predicted_shape_{patch_n}-{pic_n}"
            name_v2 = f"{Building_data}_sh_clipped_{patch_n}-{pic_n}"

            var_use = "a_predicte"
            list_rule = [
                ("b_build", 1, 3),
            ]
            list_var_drop = ["a_cat", "b_cat", "a_predicte", "b_build"]
            str_fn_out = f"predict_{Building_data}_mod_{patch_n}-{pic_n}"

            path_shp_merge = predict_vector(
                list_rule,
                list_var_drop,
                name_v1,
                name_v2,
                path_out,
                path_raster_predict,
                str_fn_out,
                var_use,
            )

    return path_shp_merge


def process_overlap(path_save, path_GUF, patch_n, pic_n):
    overlap_found = find_overlap(path_save, patch_n, pic_n, path_GUF)
    path_footprint_masked = path_save / f"masked_footprint_{patch_n}-{pic_n}.tif"
    if overlap_found == 0:
        raise RuntimeError("Overlap is not found ...")
        # sys.exit()
    elif overlap_found == 1:
        print("Overlap is found. Clipping the data..")
        path_footprint_overlap = (
                path_save / (path_footprint_masked.stem + f"-{overlap_found}.tif")
        )
        os.rename(path_footprint_overlap, path_footprint_masked)
    else:
        print("more than one overlap is found ...")
        path_footprint_masked = merge_overlap(overlap_found, path_footprint_masked)
    return path_footprint_masked


def predict_vector(
        list_rule,
        list_var_drop,
        name_v1,
        name_v2,
        path_out,
        path_raster,
        str_fn_out,
        var_use,
):
    predict_GUF_rd = merge_vector_data(path_out, path_raster, name_v1, name_v2)
    path_shp_predict = predict_feature(
        predict_GUF_rd, var_use, list_rule, list_var_drop, path_out, str_fn_out,
    )

    return path_shp_predict


def tif2shp(path_raster, name_feature, name_out, path_out, box_domain, patch_n, pic_n):
    predicted = gen_gdf(path_raster)
    try:
        predicted = clip_shp(predicted, box_domain)
    except:
        print("Some exception happened when clipping urban_foot and box_domain")
    path_shp = save_shp(predicted, name_feature, name_out, path_out, patch_n, pic_n)
    return path_shp


def save_shp(gdf_in, name_feature, name_save, path_out, patch_n, pic_n):
    gdf_in = gdf_in.rename(columns={"raster_val": name_feature})
    gdf_in = gdf_in[gdf_in[name_feature] != 128]
    urban_foot_temp = gdf_in.buffer(0)
    urban_foot_temp = gpd.GeoDataFrame(urban_foot_temp)
    urban_foot_temp[name_feature] = gdf_in[name_feature]
    urban_foot_temp = urban_foot_temp.rename(columns={0: "geometry"})
    urban_foot_temp.crs = {"init": "epsg:4326"}
    path_shp_save = path_out / f"{name_save}_{patch_n}-{pic_n}"
    urban_foot_temp.to_file(path_shp_save)
    return path_shp_save


def gen_gdf(path_raster_in):
    with rasterio.Env():
        with rasterio.open(path_raster_in) as src:
            image = src.read(1)  # first band
            results = (
                {"properties": {"raster_val": v}, "geometry": s}
                for i, (s, v) in enumerate(
                shapes(image, mask=None, transform=src.transform)
            )
            )
    gdf_out = gpd.GeoDataFrame.from_features(list(results))
    return gdf_out


def merge_GUF(path_out, path_raster, patch_n, pic_n):
    print("Merging the GUF data with the predicted ...")
    time.sleep(3)
    # key names
    name_v1 = f"predicted_shape_{patch_n}-{pic_n}"
    name_v2 = f"urban_foot_shape_{patch_n}-{pic_n}"
    predict_GUF = merge_vector_data(path_out, path_raster, name_v1, name_v2)
    predict_GUF = predict_GUF[~np.isnan(predict_GUF.a_predicte)]
    predict_GUF["LC"] = predict_GUF.a_predicte
    gdf_GUF = predict_GUF[
        (predict_GUF.b_GUF == 255)
        & (predict_GUF.a_predicte != 1)
        & (predict_GUF.a_predicte != 2)
        ]
    predict_GUF.loc[gdf_GUF.index, "LC"] = 0
    gdf_GUF_x = predict_GUF[
        (~np.isnan(predict_GUF.b_GUF))
        & (predict_GUF.b_GUF != 255)
        & (predict_GUF.a_predicte == 0)
        ]
    predict_GUF.loc[gdf_GUF_x.index, "LC"] = 3
    predict_GUF = predict_GUF.drop(
        ["a_cat", "b_GUF", "b_cat", "a_predicte", "cat"], axis=1
    )
    predict_GUF_temp = predict_GUF.buffer(0)
    predict_GUF_temp = gpd.GeoDataFrame(predict_GUF_temp)
    predict_GUF_temp["LC"] = predict_GUF["LC"]
    predict_GUF_temp = predict_GUF_temp.rename(columns={0: "geometry"})
    predict_GUF_temp.crs = {"init": "epsg:4326"}
    predict_GUF_temp.to_file(path_out / f"predict_GUF_mod_{patch_n}-{pic_n}")
    return gdf_GUF


def merge_OSM(
        gdf_OSM, lat_left_top, lat_right_bot, lon_left_top, lon_right_bot, path_out, patch_n
):
    xtile_min, xtile_max, ytile_min, ytile_max, zoom = cal_prm_tile(
        lon_left_top, lat_right_bot, lon_right_bot, lat_left_top
    )
    path_OSM = download_OSM(
        path_out, patch_n, xtile_min, xtile_max, ytile_min, ytile_max, zoom,
    )
    print("Attaching the OSM data together ...")
    counter = 0
    for i in range(xtile_min, xtile_max + 1):
        for j in range(ytile_min, ytile_max + 1):
            try:
                gdf_tile = gpd.read_file(path_OSM / f"buildings_{i}-{j}.geojson")
                gdf_tile = gdf_tile.buffer(0)

                if counter == 0:
                    gdf_OSM = gdf_tile

                if counter != 0:
                    gdf_OSM = gdf_OSM.union(gdf_tile)
                counter = 1
            except:
                pass
    path_OSM_sh = path_OSM.stem + "_sh"
    if not os.path.isdir(path_OSM_sh):
        os.makedirs(path_OSM_sh)
    gdf_OSM.to_file(path_OSM_sh)


def clip_buildings(Building_data, box_domain, building_dir, path_out, patch_n, pic_n):
    if Building_data == "OSM":
        path_osm = Path(f"OSM_{patch_n}-{pic_n}_sh.shp")
        buildings = gpd.read_file(path_out / path_osm.stem / path_osm)
    elif Building_data == "MICROSOFT":
        print("Reading the Microsoft building data ...")
        buildings = gpd.read_file(building_dir)
    print("Clipping the building data to the selected domain")
    buildings_clipped = clip_shp(buildings, box_domain)
    buildings_clipped = buildings_clipped.buffer(0)
    buildings_clipped = gpd.GeoDataFrame(buildings_clipped)
    buildings_clipped["build"] = 1
    buildings_clipped = buildings_clipped.rename(columns={0: "geometry"})
    buildings_clipped.crs = {"init": "epsg:4326"}
    path_fn = path_out / f"{Building_data}_sh_clipped_{patch_n}-{pic_n}"
    print(f"Writing clipped data into {path_fn.as_posix()}")
    buildings_clipped.to_file(path_fn)


def cal_prm_tile(lon_left_top, lat_right_bot, lon_right_bot, lat_left_top):
    lon_deg_min, lat_deg_min = (lon_left_top, lat_right_bot)
    lon_deg_max, lat_deg_max = (lon_right_bot, lat_left_top)
    lat_rad_min = lat_deg_min * np.pi / 180
    lat_rad_max = lat_deg_max * np.pi / 180

    zoom = 15
    n = 2 ** zoom

    xtile_min = n * ((lon_deg_min + 180) / 360)
    ytile_min = (
            n * (1 - (np.log(np.tan(lat_rad_min) + (1 / np.cos(lat_rad_min))) / np.pi)) / 2
    )

    xtile_max = n * ((lon_deg_max + 180) / 360)
    ytile_max = (
            n * (1 - (np.log(np.tan(lat_rad_max) + (1 / np.cos(lat_rad_max))) / np.pi)) / 2
    )

    if ytile_max < ytile_min:
        temp = ytile_min
        ytile_min = ytile_max
        ytile_max = temp

    ytile_min = int(np.floor(ytile_min))
    ytile_max = int(np.ceil(ytile_max))

    xtile_min = int(np.floor(xtile_min))
    xtile_max = int(np.ceil(xtile_max))
    xtile_min = xtile_min - 3
    return xtile_min, xtile_max, ytile_min, ytile_max, zoom


def predict_feature(gdf_in, var_use, list_rule, list_var_drop, path_out, str_fn_out):
    # select feature to use
    gdf_in["LC"] = gdf_in[var_use]

    # apply rules to set new values
    # rule:
    # `var`: variable for selection
    # `val_var`: value of variable for selection
    # `val_set`: new value to set in `gdf_in`
    for var, val_var, val_set in list_rule:
        temp = gdf_in[gdf_in[var] == val_var]
        gdf_in.loc[temp.index, "LC"] = val_set

    # drop unnecessary variables
    gdf_in = gdf_in.drop(list_var_drop, axis=1)

    # create new geoDF as output
    gdf_out = gdf_in.buffer(0)
    gdf_out = gpd.GeoDataFrame(gdf_out)
    gdf_out["LC"] = gdf_in["LC"]
    gdf_out = gdf_out.rename(columns={0: "geometry"})
    gdf_out.crs = {"init": "epsg:4326"}
    print("dissolving the final result ... .")
    gdf_out = gdf_out.dissolve("LC")
    gdf_out = gdf_out.reset_index()

    # output as an external file
    path_fn_out = Path(path_out) / str_fn_out
    gdf_out.to_file(path_fn_out)

    return path_fn_out


def download_OSM_road(lat_left_top, lat_right_bot, lon_right_bot, lon_left_top, path_out, patch_n, pic_n, debug=False):
    # cast to `Path`
    path_out_x = Path(path_out)

    gdf = download_OSM_box(lat_left_top, lat_right_bot, lon_right_bot, lon_left_top, False)
    road_kind = []
    for rd in gdf.highway:
        if rd[0] == "[":
            rd = ast.literal_eval(rd)
        if type(rd) != list:
            rd = [rd]
        road_kind.append(rd[0])

    gdf["cat"] = road_kind

    with open(path_module / "road_width.json") as setting_file:
        subset = json.load(setting_file)

    buffered = gpd.GeoDataFrame(columns=["geometry", "cat"])
    for kind in subset.keys():
        temp0 = gdf[gdf.cat == kind]
        temp = temp0.buffer(subset[kind])
        temp = gpd.GeoDataFrame(temp, columns=["geometry"])
        temp["cat"] = temp0.cat
        buffered = buffered.append(temp)

    buffered.crs = gdf.crs
    buffered.to_file(path_out_x / f"roads_all_{patch_n}-{pic_n}")

    buffered.cat = "5"
    if debug:
        print("Dissolving roads ...")
    buffered = buffered.dissolve(by="cat")
    buffered = buffered.to_crs(epsg=4326)
    buffered.to_file(path_out_x / f"roads_{patch_n}-{pic_n}")


@lru_cache(maxsize=32)
def download_OSM_box(lat_left_top, lat_right_bot, lon_right_bot, lon_left_top, debug):
    if debug:
        print("Downloading road data from OSM ...")
    G = ox.graph_from_bbox(
        lat_left_top,
        lat_right_bot,
        lon_right_bot,
        lon_left_top,
        network_type="all_private",
    )
    G_projected = ox.project_graph(G)
    gdf = ox.save_load.graph_to_gdfs(G_projected)[1]
    return gdf


def merge_vector_data(path_out: Path, path_raster: Path, name_v1: str, name_v2: str):
    # # TODO: why do we need this sleep?
    # time.sleep(3)

    # force cast to `Path`
    path_out_x = Path(path_out)

    # dir names
    path_dir_v1 = path_out_x / name_v1
    path_dir_v2 = path_out_x / name_v2
    # path_dir_out = path_out_x / name_out

    # overlay results
    gdf_merge = grass_overlay(path_dir_v1, path_dir_v2, path_raster)
    gdf_out = gdf_merge.copy()

    return gdf_out


def download_OSM(path_out, patch_n, xtile_min, xtile_max, ytile_min, ytile_max, zoom):
    print("Downloading the OSM data ... ")
    to_wait = 60
    path_OSM = path_out / f"OSM{patch_n}"
    if not os.path.isdir(path_OSM):
        os.makedirs(path_OSM)

    for i in np.arange(xtile_min, xtile_max + 1):
        print(f"Xtile {i - xtile_min + 1} out of {xtile_max - xtile_min + 1}")
        for j in np.arange(ytile_min, ytile_max + 1):
            path_geojson = path_OSM / f"buildings_{i}-{j}.geojson"
            if not path_geojson.exists():
                url = f"https://a.data.osmbuildings.org/0.2/anonymous/tile/{zoom}/{i}/{j}.json"
                try:

                    r = requests.get(url, timeout=10)
                    while r.status_code == 429:
                        print("Too many requests. Waiting for {to_wait} seconds ...")
                        time.sleep(to_wait)
                        print("Trying again")
                        r = requests.get(url, timeout=10)

                    with open(path_geojson, "wb", ) as f:
                        f.write(r.content)
                except:
                    print("Passing this domain")

    print("All data are collected. DONE")
    return path_OSM
