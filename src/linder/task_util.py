import ast
import json
import os
import sys
import time
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
import rasterio.mask
from eolearn.core import EOPatch
from rasterio.features import shapes
from rasterio.merge import merge
from shapely.geometry import box

from ._env import path_module
from .clip_util import clip_shp
from .grass_util import grass_overlay


def find_overlap(path_out, patch_n, pic_n, list_of_GUF):
    path_shp = Path(f"shape_box{patch_n}-{pic_n}.shp")
    with fiona.open(path_out / path_shp.stem / path_shp, "r",) as shapefile:
        features = [feature["geometry"] for feature in shapefile]

    overlap_found = 0
    for GUF_data_dir in list_of_GUF:

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
                path_out / f"masked_footprint{patch_n}-{pic_n}{overlap_found}.tif",
                "w",
                **out_meta,
            ) as dest:
                dest.write(out_image)

        except:
            pass
    return overlap_found


def merger(overlap_found, path_out, patch_n, pic_n):
    all_files = []
    for i in range(1, overlap_found + 1):
        all_files.append(path_out / f"masked_footprint{patch_n}-{pic_n}{i}.tif")

    src_files_to_mosaic = []
    for fp in all_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )

    out_fp = path_out / f"masked_footprint{patch_n}-{pic_n}.tif"

    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)


# TODO: this function is too heavy; should be modularised into several independent ones.
def other_tasks(
    path_out,
    patch_n,
    use_GUF,
    Building_data,
    Road_data,
    building_dir,
    list_of_GUF,
    lat_left_top,
    lon_left_top,
    lat_right_bot,
    lon_right_bot,
):
    # cast to Path
    path_out = Path(path_out)

    # get number of pictures
    eopatch = EOPatch.load(path_out / f"eopatch_{patch_n}", lazy_loading=True)
    n_pics = eopatch.data["BANDS"].shape[0]

    for pic_n in range(n_pics):
        print("Getting the boundaries of the image. . .")
        path_raster = (
            path_out
            / "predicted_tiff"
            / f"patch{patch_n}"
            / f"picture{pic_n}"
            / "merged_prediction.tiff"
        )
        raster = rasterio.open(path_raster)
        df = raster.bounds
        sh_box = box(df.left, df.bottom, df.right, df.top)
        shape_box = gpd.GeoDataFrame({"geometry": sh_box, "col": [np.nan]})
        shape_box.crs = {"init": "epsg:4326"}
        shape_box.to_file(path_out / f"shape_box{patch_n}-{pic_n}")

        if use_GUF:
            print("Clipping the GUF data . . .")
            overlap_found = find_overlap(path_out, patch_n, pic_n, list_of_GUF)

            if overlap_found == 0:
                print("Overlap is not found . . .")
                sys.exit()
            elif overlap_found == 1:
                print("Overlap is found. Clipping the data..")
                path_fn1 = (
                    path_out / f"masked_footprint{patch_n}-{pic_n}{overlap_found}.tif"
                )
                path_fn2 = path_out / f"masked_footprint{patch_n}-{pic_n}.tif"
                os.rename(path_fn1, path_fn2)
            else:
                print("more than one overlap is found . . .")
                merger(overlap_found, path_out, patch_n, pic_n)

            mask = None
            with rasterio.Env():
                with rasterio.open(
                    path_out / f"masked_footprint{patch_n}-{pic_n}.tif"
                ) as src:
                    image = src.read(1)  # first band
                    results = (
                        {"properties": {"raster_val": v}, "geometry": s}
                        for i, (s, v) in enumerate(
                            shapes(image, mask=mask, transform=src.transform)
                        )
                    )
            geoms_foot = list(results)
            urban_foot = gpd.GeoDataFrame.from_features(geoms_foot)

            path_shp = Path(f"shape_box{patch_n}-{pic_n}.shp")
            box_domain = gpd.read_file(path_out / path_shp.stem / path_shp)
            try:
                urban_foot = clip_shp(urban_foot, box_domain)
            except:
                print("Some exception happened when clipping urban_foot and box_domain")
            urban_foot = urban_foot.rename(columns={"raster_val": "GUF"})
            urban_foot = urban_foot[urban_foot.GUF != 128]

            urban_foot_temp = urban_foot.buffer(0)
            urban_foot_temp = gpd.GeoDataFrame(urban_foot_temp)
            urban_foot_temp["GUF"] = urban_foot["GUF"]
            urban_foot_temp = urban_foot_temp.rename(columns={0: "geometry"})
            urban_foot_temp.crs = {"init": "epsg:4326"}
            urban_foot_temp.to_file(path_out / f"urban_foot_shape{patch_n}-{pic_n}")

        mask = None
        print("Converting the predicted tiff file to shapefile . . .")
        with rasterio.Env():
            with rasterio.open(path_raster) as src:
                image = src.read(1)  # first band
                results = (
                    {"properties": {"raster_val": v}, "geometry": s}
                    for i, (s, v) in enumerate(
                        shapes(image, mask=mask, transform=src.transform)
                    )
                )
        geoms_predict = list(results)
        predicted = gpd.GeoDataFrame.from_features(geoms_predict)

        predicted = predicted.rename(columns={"raster_val": "predicted"})
        predicted_temp = predicted.buffer(0)
        predicted_temp = gpd.GeoDataFrame(predicted_temp)
        predicted_temp["predicted"] = predicted["predicted"]
        predicted_temp = predicted_temp.rename(columns={0: "geometry"})
        predicted_temp.crs = {"init": "epsg:4326"}
        predicted_temp.to_file(path_out / f"predicted_shape{patch_n}-{pic_n}")

        if use_GUF:
            print("Merging the GUF data with the predicted . . .")
            time.sleep(3)
            # key names
            name_v1 = f"predicted_shape{patch_n}-{pic_n}"
            name_v2 = f"urban_foot_shape{patch_n}-{pic_n}"
            name_out = f"predict_GUF{patch_n}-{pic_n}"

            predict_GUF = merge_vector_data(
                path_out, path_raster, name_v1, name_v2, name_out,
            )

            predict_GUF = predict_GUF[~np.isnan(predict_GUF.a_predicte)]
            predict_GUF["LC"] = predict_GUF.a_predicte
            a = predict_GUF[
                (predict_GUF.b_GUF == 255)
                & (predict_GUF.a_predicte != 1)
                & (predict_GUF.a_predicte != 2)
            ]
            predict_GUF.loc[a.index, "LC"] = 0
            b = predict_GUF[
                (~np.isnan(predict_GUF.b_GUF))
                & (predict_GUF.b_GUF != 255)
                & (predict_GUF.a_predicte == 0)
            ]
            predict_GUF.loc[b.index, "LC"] = 3
            predict_GUF = predict_GUF.drop(
                ["a_cat", "b_GUF", "b_cat", "a_predicte", "cat"], axis=1
            )
            predict_GUF_temp = predict_GUF.buffer(0)
            predict_GUF_temp = gpd.GeoDataFrame(predict_GUF_temp)
            predict_GUF_temp["LC"] = predict_GUF["LC"]
            predict_GUF_temp = predict_GUF_temp.rename(columns={0: "geometry"})
            predict_GUF_temp.crs = {"init": "epsg:4326"}
            predict_GUF_temp.to_file(path_out / f"predict_GUF_mod{patch_n}-{pic_n}")

        # if Building_data != "no":
        if Building_data == "OSM":
            xtile_min, xtile_max, ytile_min, ytile_max, zoom = cal_prm_tile(
                lon_left_top, lat_right_bot, lon_right_bot, lat_left_top
            )
            path_OSM = download_OSM(
                path_out, patch_n, xtile_min, xtile_max, ytile_min, ytile_max, zoom,
            )

            print("Attaching the OSM data together . . .")
            counter = 0
            for i in range(xtile_min, xtile_max + 1):
                for j in range(ytile_min, ytile_max + 1):
                    try:
                        b = gpd.read_file(path_OSM / f"buildings{i}-{j}.geojson")
                        b = b.buffer(0)

                        if counter == 0:
                            a = b

                        if counter != 0:
                            a = a.union(b)
                        counter = 1
                    except:
                        pass

            path_OSM_sh = path_OSM.stem + "_sh"
            if not os.path.isdir(path_OSM_sh):
                os.makedirs(path_OSM_sh)

            a.to_file(path_OSM_sh)

        path_shp = Path(f"shape_box{patch_n}-{pic_n}.shp")
        box_domain = gpd.read_file(path_out / path_shp.stem / path_shp)

        if Building_data != "no":
            if Building_data == "OSM":
                path_shp = Path(f"OSM{patch_n}-{pic_n}_sh.shp")
                buildings = gpd.read_file(path_out / path_shp.stem / path_shp)
            elif Building_data == "MICROSOFT":
                print("Reading the Microsoft building data . . .")
                buildings = gpd.read_file(building_dir)

            print("Clipping the building data to the selected domain")
            buildings_clipped = clip_shp(buildings, box_domain)
            buildings_clipped = buildings_clipped.buffer(0)
            buildings_clipped = gpd.GeoDataFrame(buildings_clipped)
            buildings_clipped["build"] = 1
            buildings_clipped = buildings_clipped.rename(columns={0: "geometry"})
            buildings_clipped.crs = {"init": "epsg:4326"}

            path_fn = path_out / f"{Building_data}_sh_clipped{patch_n}-{pic_n}"
            print(f"Writing clipped data into {path_fn.as_posix()}")
            buildings_clipped.to_file(path_fn)

        if Road_data != "no":
            download_OSM_road(
                lat_left_top,
                lat_right_bot,
                lon_right_bot,
                lon_left_top,
                path_out,
                patch_n,
                pic_n
            )

        if use_GUF:
            if Building_data != "no" and Road_data == "no":
                print("Merging the predicted-GUF to Building data . . .")
                # key names
                name_v1 = f"predict_GUF_mod{patch_n}-{pic_n}"
                name_v2 = f"{Building_data}_sh_clipped{patch_n}-{pic_n}"
                name_out = f"predict_GUF_{Building_data}{patch_n}-{pic_n}"

                predict_GUF_bld = merge_vector_data(
                    path_out, path_raster, name_v1, name_v2, name_out,
                )

                var_use = "a_LC"
                list_rule = [
                    ("b_build", 1, 4),
                ]
                list_var_drop = ["a_cat", "b_cat", "a_LC", "b_build"]
                str_fn_out = f"predict_GUF_{Building_data}_mod{patch_n}-{pic_n}"

                predict_feature(
                    predict_GUF_bld,
                    var_use,
                    list_rule,
                    list_var_drop,
                    path_out,
                    str_fn_out,
                )

            elif Road_data != "no" and Building_data == "no":
                print("Merging the predicted-GUF to Road data . . .")
                # key names
                name_v1 = f"predict_GUF_mod{patch_n}-{pic_n}"
                name_v2 = f"roads_{patch_n}-{pic_n}"
                name_out = f"predict_GUF_roads_{patch_n}-{pic_n}"

                predict_GUF_rd = merge_vector_data(
                    path_out, path_raster, name_v1, name_v2, name_out,
                )

                var_use = "a_LC"
                list_rule = [
                    ("b_cat", 1, 5),
                    ("LC", 0, 4),
                    ("LC", 5, 0),
                ]
                list_var_drop = ["cat", "a_cat", "b_cat", "a_LC", "b_FID"]
                str_fn_out = f"predict_GUF_roads_mod{patch_n}-{pic_n}"

                predict_feature(
                    predict_GUF_rd,
                    var_use,
                    list_rule,
                    list_var_drop,
                    path_out,
                    str_fn_out,
                )

            elif Road_data != "no" and Building_data != "no":
                print(
                    "Both Building and Road: Merging the predicted-GUF to Road data . . ."
                )
                # key names
                name_v1 = f"predict_GUF_mod{patch_n}-{pic_n}"
                name_v2 = f"roads_{patch_n}-{pic_n}"
                name_out = f"predict_GUF_roads_{patch_n}-{pic_n}"

                predict_GUF_rd = merge_vector_data(
                    path_out, path_raster, name_v1, name_v2, name_out,
                )

                var_use = "a_LC"
                list_rule = [
                    ("b_cat", 1, 5),
                    ("LC", 0, 4),
                    ("LC", 5, 0),
                ]
                list_var_drop = ["cat", "a_cat", "b_cat", "a_LC", "b_FID"]
                str_fn_out = f"predict_GUF_roads_mod{patch_n}-{pic_n}"

                predict_feature(
                    predict_GUF_rd,
                    var_use,
                    list_rule,
                    list_var_drop,
                    path_out,
                    str_fn_out,
                )

                print("Merging the predicted-GUF-roads to Building data . . .")

                # key names
                name_v1 = f"predict_GUF_roads_mod{patch_n}-{pic_n}"
                name_v2 = f"{Building_data}_sh_clipped{patch_n}-{pic_n}"
                name_out = f"predict_GUF_roads_{Building_data}{patch_n}-{pic_n}"

                predict_GUF_rd_bd = merge_vector_data(
                    path_out, path_raster, name_v1, name_v2, name_out,
                )

                var_use = "a_LC"
                list_rule = [
                    ("a_LC", 4, 5),
                    ("b_build", 1, 4),
                ]
                list_var_drop = ["cat", "a_cat", "b_cat", "a_LC", "b_build"]
                str_fn_out = f"predict_GUF_roads_{Building_data}_mod{patch_n}-{pic_n}"

                predict_feature(
                    predict_GUF_rd_bd,
                    var_use,
                    list_rule,
                    list_var_drop,
                    path_out,
                    str_fn_out,
                )

        else:
            if Building_data != "no" and Road_data == "no":
                print("Merging the predicted to Building data . . .")
                # key names
                name_v1 = f"predicted_shape{patch_n}-{pic_n}"
                name_v2 = f"{Building_data}_sh_clipped{patch_n}-{pic_n}"
                name_out = f"predict_{Building_data}{patch_n}-{pic_n}"
                predict_bld = merge_vector_data(
                    path_out, path_raster, name_v1, name_v2, name_out,
                )

                var_use = "a_predicte"
                list_rule = [
                    ("b_build", 1, 3),
                ]
                list_var_drop = ["a_cat", "b_cat", "a_predicte", "b_build"]
                str_fn_out = f"predict_{Building_data}_mod{patch_n}-{pic_n}"

                predict_feature(
                    predict_bld, var_use, list_rule, list_var_drop, path_out, str_fn_out
                )


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
    print("dissolving the final result . . . .")
    gdf_out = gdf_out.dissolve("LC")
    gdf_out = gdf_out.reset_index()

    # output as an external file
    path_out_x = Path(path_out)
    gdf_out.to_file(path_out_x / str_fn_out)


def download_OSM_road(
    lat_left_top, lat_right_bot, lon_right_bot, lon_left_top, path_out, patch_n,pic_n
):
    # cast to `Path`
    path_out_x = Path(path_out)

    print("Downloading road data from OSM . . .")
    G = ox.graph_from_bbox(
        lat_left_top,
        lat_right_bot,
        lon_right_bot,
        lon_left_top,
        network_type="all_private",
    )
    G_projected = ox.project_graph(G)
    gdf = ox.save_load.graph_to_gdfs(G_projected)[1]
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
    print("Dissolving roads . . .")
    buffered = buffered.dissolve(by="cat")
    buffered = buffered.to_crs(epsg=4326)
    buffered.to_file(path_out_x / f"roads_{patch_n}-{pic_n}")


def merge_vector_data(
    path_out: Path, path_raster: Path, name_v1: str, name_v2: str, name_out: str,
):
    # TODO: why do we need this sleep?
    time.sleep(3)

    # force cast to `Path`
    path_out_x = Path(path_out)

    # filenames
    path_fn_v1 = Path(f"{name_v1}.shp")
    path_fn_v2 = Path(f"{name_v2}.shp")
    path_fn_out = Path(f"{name_out}.shp")
    # dir names
    path_dir_v1 = path_out_x / path_fn_v1.stem
    path_dir_v2 = path_out_x / path_fn_v2.stem
    path_dir_out = path_out_x / path_fn_out.stem

    # overlay results
    grass_overlay(path_dir_v1, path_dir_v2, path_dir_out, path_raster)

    # load geoDF as returned object for later processing
    gdf_merge = gpd.read_file(path_dir_out / path_fn_out)

    return gdf_merge


def download_OSM(path_out, patch_n, xtile_min, xtile_max, ytile_min, ytile_max, zoom):
    print("Downloading the OSM data . . . ")
    to_wait = 60
    path_OSM = path_out / f"OSM{patch_n}"
    if not os.path.isdir(path_OSM):
        os.makedirs(path_OSM)

    for i in np.arange(xtile_min, xtile_max + 1):
        print(f"Xtile {i - xtile_min + 1} out of {xtile_max - xtile_min + 1}")
        for j in np.arange(ytile_min, ytile_max + 1):
            path_geojson = path_OSM / f"buildings{i}-{j}.geojson"
            if not path_geojson.exists():
                url = f"https://a.data.osmbuildings.org/0.2/anonymous/tile/{zoom}/{i}/{j}.json"
                try:

                    r = requests.get(url, timeout=10)
                    while r.status_code == 429:
                        print("Too many requests. Waiting for {to_wait} seconds ...")
                        time.sleep(to_wait)
                        print("Trying again")
                        r = requests.get(url, timeout=10)

                    with open(path_geojson, "wb",) as f:
                        f.write(r.content)
                except:
                    print("Passing this domain")

    print("All data are collected. DONE")
    return path_OSM
