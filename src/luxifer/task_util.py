import rasterio
import rasterio.mask
from shapely.geometry import box
import geopandas as gpd
import fiona
from rasterio.features import shapes
import time
import numpy as np
import os
from clip_util import clip_shp
from grass_util import grass_overlay
import json
import osmnx as ox
from pathlib import Path
import ast
import sys
from rasterio.merge import merge


def find_overlap(path_out, patch_n, list_of_GUF):

    with fiona.open(
        path_out + "/shape_box" + str(patch_n) + "/shape_box" + str(patch_n) + ".shp",
        "r",
    ) as shapefile:
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
                path_out
                + "/masked_footprint"
                + str(patch_n)
                + str(overlap_found)
                + ".tif",
                "w",
                **out_meta
            ) as dest:
                dest.write(out_image)

        except:
            pass
    return overlap_found


def merger(overlap_found, path_out, patch_n):
    all_files = []
    for i in range(1, overlap_found + 1):
        all_files.append(
            path_out + "/masked_footprint" + str(patch_n) + str(i) + ".tif"
        )

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

    out_fp = path_out + "/masked_footprint" + str(patch_n) + ".tif"

    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)


def other_tasks(
    path_out,
    patch_n,
    GUF_data,
    Building_data,
    Road_data,
    building_dir,
    list_of_GUF,
    lat_left_top,
    lon_left_top,
    lat_right_bot,
    lon_right_bot,
):
    print("Getting the boundaries of the image. . .")
    raster = rasterio.open(
        path_out + "/predicted_tiff/patch" + str(patch_n) + "/merged_prediction.tiff"
    )
    df = raster.bounds
    sh_box = box(df.left, df.bottom, df.right, df.top)
    shape_box = gpd.GeoDataFrame({"geometry": sh_box, "col": [np.nan]})
    shape_box.crs = {"init": "epsg:4326"}
    shape_box.to_file(path_out + "/shape_box" + str(patch_n))

    if GUF_data:
        print("Clipping the GUF data . . .")
        overlap_found = find_overlap(path_out, patch_n, list_of_GUF)

        if overlap_found == 0:
            print("Overlap is not found . . .")
            sys.exit()
        elif overlap_found == 1:
            print("Overlap is found. Clipping the data..")
            name1 = (
                path_out
                + "/masked_footprint"
                + str(patch_n)
                + str(overlap_found)
                + ".tif"
            )
            name2 = path_out + "/masked_footprint" + str(patch_n) + ".tif"
            os.rename(name1, name2)
        else:
            print("more than one overlap is found . . .")
            merger(overlap_found, path_out, patch_n)

        mask = None
        with rasterio.Env():
            with rasterio.open(
                path_out + "/masked_footprint" + str(patch_n) + ".tif"
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

        box_domain = gpd.read_file(
            path_out
            + "/shape_box"
            + str(patch_n)
            + "/shape_box"
            + str(patch_n)
            + ".shp"
        )
        try:
            urban_foot = clip_shp(urban_foot, box_domain)
        except:
            print("Some exception happend when clipping urban_foot and box_domain")
        urban_foot = urban_foot.rename(columns={"raster_val": "GUF"})
        urban_foot = urban_foot[urban_foot.GUF != 128]

        urban_foot_temp = urban_foot.buffer(0)
        urban_foot_temp = gpd.GeoDataFrame(urban_foot_temp)
        urban_foot_temp["GUF"] = urban_foot["GUF"]
        urban_foot_temp = urban_foot_temp.rename(columns={0: "geometry"})
        urban_foot_temp.crs = {"init": "epsg:4326"}
        urban_foot_temp.to_file(path_out + "/urban_foot_shape" + str(patch_n))

    mask = None
    print("Converting the predicted tiff file to shapefile . . .")
    with rasterio.Env():
        with rasterio.open(
            path_out
            + "/predicted_tiff/patch"
            + str(patch_n)
            + "/merged_prediction.tiff"
        ) as src:
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
    predicted_temp.to_file(path_out + "/predicted_shape" + str(patch_n))

    if GUF_data:
        print("Merging the GUF data with the predicted . . .")
        time.sleep(3)
        v1_dir = (
            path_out
            + "/predicted_shape"
            + str(patch_n)
            + "/predicted_shape"
            + str(patch_n)
            + ".shp"
        )
        v2_dir = (
            path_out
            + "/urban_foot_shape"
            + str(patch_n)
            + "/urban_foot_shape"
            + str(patch_n)
            + ".shp"
        )
        out_dir = path_out + "/predict_GUF" + str(patch_n)
        grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out)

        predict_GUF = gpd.read_file(
            path_out
            + "/predict_GUF"
            + str(patch_n)
            + "/predict_GUF"
            + str(patch_n)
            + ".shp"
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
        predict_GUF_temp.to_file(path_out + "/predict_GUF_mod" + str(patch_n))

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

    if Building_data != "no":
        if Building_data == "OSM":
            print("Downloading the OSM data . . . ")
            to_wait = 60
            path_OSM = path_out + "/OSM" + str(patch_n)
            if not os.path.isdir(path_OSM):
                os.makedirs(path_OSM)

            for i in np.arange(xtile_min, xtile_max + 1):
                print(
                    "Xtile "
                    + str(i - xtile_min + 1)
                    + " out of "
                    + str(xtile_max - xtile_min + 1)
                )
                for j in np.arange(ytile_min, ytile_max + 1):

                    if not Path(
                        path_OSM + "/buildings" + str(i) + "-" + str(j) + ".geojson"
                    ).exists():
                        url = "https://a.data.osmbuildings.org/0.2/anonymous/tile/{}/{}/{}.json".format(
                            zoom, i, j
                        )
                        try:

                            r = requests.get(url, timeout=10)
                            while r.status_code == 429:
                                print(
                                    "Too many requests. Waiting for "
                                    + str(to_wait)
                                    + " seconds..."
                                )
                                time.sleep(to_wait)
                                print("Trying again")
                                r = requests.get(url, timeout=10)

                            with open(
                                path_OSM
                                + "/buildings"
                                + str(i)
                                + "-"
                                + str(j)
                                + ".geojson",
                                "wb",
                            ) as f:
                                f.write(r.content)
                        except:
                            print("Passing this domain")

            print("All data are collected. DONE")

            print("Attaching the OSM data together . . .")
            counter = 0
            for i in range(xtile_min, xtile_max + 1):
                for j in range(ytile_min, ytile_max + 1):

                    try:
                        b = gpd.read_file(
                            path_OSM + "/buildings" + str(i) + "-" + str(j) + ".geojson"
                        )
                        b = b.buffer(0)

                        if counter == 0:
                            a = b

                        if counter != 0:
                            a = a.union(b)
                        counter = 1
                    except:
                        pass

            path_OSM_sh = path_OSM + "_sh"
            if not os.path.isdir(path_OSM_sh):
                os.makedirs(path_OSM_sh)

            a.to_file(path_OSM_sh)

    box_domain = gpd.read_file(
        path_out + "/shape_box" + str(patch_n) + "/shape_box" + str(patch_n) + ".shp"
    )

    if Building_data != "no":
        if Building_data == "OSM":
            buildings = gpd.read_file(
                path_out + "/OSM" + str(patch_n) + "_sh/OSM" + str(patch_n) + "_sh.shp"
            )
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
        print(
            "Writing clipped data into "
            + path_out
            + "/"
            + Building_data
            + "_sh_clipped"
            + str(patch_n)
        )
        buildings_clipped.to_file(
            path_out + "/" + Building_data + "_sh_clipped" + str(patch_n)
        )

    if Road_data != "no":
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

        with open("./road_width.json") as setting_file:
            subset = json.load(setting_file)

        buffered = gpd.GeoDataFrame(columns=["geometry", "cat"])
        for kind in subset.keys():
            temp0 = gdf[gdf.cat == kind]
            temp = temp0.buffer(subset[kind])
            temp = gpd.GeoDataFrame(temp, columns=["geometry"])
            temp["cat"] = temp0.cat
            buffered = buffered.append(temp)

        buffered.crs = gdf.crs
        buffered.to_file(path_out + "/roads_all_" + str(patch_n))

        buffered.cat = "5"
        print("Dissolving roads . . .")
        buffered = buffered.dissolve(by="cat")
        buffered = buffered.to_crs(epsg=4326)
        buffered.to_file(path_out + "/roads_" + str(patch_n))

    if GUF_data:
        if Building_data != "no" and Road_data == "no":
            print("Merging the predicted-GUF to Building data . . .")
            time.sleep(3)
            v1_dir = (
                path_out
                + "/predict_GUF_mod"
                + str(patch_n)
                + "/predict_GUF_mod"
                + str(patch_n)
                + ".shp"
            )
            v2_dir = (
                path_out
                + "/"
                + Building_data
                + "_sh_clipped"
                + str(patch_n)
                + "/"
                + Building_data
                + "_sh_clipped"
                + str(patch_n)
                + ".shp"
            )
            out_dir = path_out + "/predict_GUF_" + Building_data + str(patch_n)
            grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out)

            predict_GUF_bld = gpd.read_file(
                path_out
                + "/predict_GUF_"
                + Building_data
                + str(patch_n)
                + "/predict_GUF_"
                + Building_data
                + str(patch_n)
                + ".shp"
            )
            predict_GUF_bld["LC"] = predict_GUF_bld.a_LC
            a = predict_GUF_bld[predict_GUF_bld.b_build == 1]
            predict_GUF_bld.loc[a.index, "LC"] = 4
            predict_GUF_bld = predict_GUF_bld.drop(
                ["a_cat", "b_cat", "a_LC", "b_build"], axis=1
            )
            predict_GUF_bld_temp = predict_GUF_bld.buffer(0)
            predict_GUF_bld_temp = gpd.GeoDataFrame(predict_GUF_bld_temp)
            predict_GUF_bld_temp["LC"] = predict_GUF_bld["LC"]
            predict_GUF_bld_temp = predict_GUF_bld_temp.rename(columns={0: "geometry"})
            predict_GUF_bld_temp.crs = {"init": "epsg:4326"}
            print("dissolving the final result . . . .")
            predict_GUF_bld_temp = predict_GUF_bld_temp.dissolve("LC")
            predict_GUF_bld_temp = predict_GUF_bld_temp.reset_index()
            predict_GUF_bld_temp.to_file(
                path_out + "/predict_GUF_" + Building_data + "_mod" + str(patch_n)
            )

        elif Road_data != "no" and Building_data == "no":
            print("Merging the predicted-GUF to Road data . . .")
            time.sleep(3)
            v1_dir = (
                path_out
                + "/predict_GUF_mod"
                + str(patch_n)
                + "/predict_GUF_mod"
                + str(patch_n)
                + ".shp"
            )
            v2_dir = (
                path_out
                + "/"
                + "roads_"
                + str(patch_n)
                + "/"
                + "roads_"
                + str(patch_n)
                + ".shp"
            )
            out_dir = path_out + "/predict_GUF_roads_" + str(patch_n)
            grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out)

            predict_GUF_rd = gpd.read_file(
                path_out
                + "/predict_GUF_roads_"
                + str(patch_n)
                + "/predict_GUF_roads_"
                + str(patch_n)
                + ".shp"
            )
            predict_GUF_rd["LC"] = predict_GUF_rd.a_LC

            a = predict_GUF_rd[predict_GUF_rd.b_cat == 1]
            predict_GUF_rd.loc[a.index, "LC"] = 5
            b = predict_GUF_rd[predict_GUF_rd.LC == 0]
            predict_GUF_rd.loc[b.index, "LC"] = 4
            c = predict_GUF_rd[predict_GUF_rd.LC == 5]
            predict_GUF_rd.loc[c.index, "LC"] = 0

            predict_GUF_rd = predict_GUF_rd.drop(
                ["cat", "a_cat", "b_cat", "a_LC", "b_FID"], axis=1
            )
            predict_GUF_rd_temp = predict_GUF_rd.buffer(0)
            predict_GUF_rd_temp = gpd.GeoDataFrame(predict_GUF_rd_temp)
            predict_GUF_rd_temp["LC"] = predict_GUF_rd["LC"]
            predict_GUF_rd_temp = predict_GUF_rd_temp.rename(columns={0: "geometry"})
            predict_GUF_rd_temp.crs = {"init": "epsg:4326"}
            print("dissolving the final result . . . .")
            predict_GUF_rd_temp = predict_GUF_rd_temp.dissolve("LC")
            predict_GUF_rd_temp = predict_GUF_rd_temp.reset_index()
            predict_GUF_rd_temp.to_file(
                path_out + "/predict_GUF_roads" + "_mod" + str(patch_n)
            )

        elif Road_data != "no" and Building_data != "no":
            print(
                "Both Building and Road: Merging the predicted-GUF to Road data . . ."
            )
            time.sleep(3)
            v1_dir = (
                path_out
                + "/predict_GUF_mod"
                + str(patch_n)
                + "/predict_GUF_mod"
                + str(patch_n)
                + ".shp"
            )
            v2_dir = (
                path_out
                + "/"
                + "roads_"
                + str(patch_n)
                + "/"
                + "roads_"
                + str(patch_n)
                + ".shp"
            )
            out_dir = path_out + "/predict_GUF_roads_" + str(patch_n)
            grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out)

            predict_GUF_rd = gpd.read_file(
                path_out
                + "/predict_GUF_roads_"
                + str(patch_n)
                + "/predict_GUF_roads_"
                + str(patch_n)
                + ".shp"
            )
            predict_GUF_rd["LC"] = predict_GUF_rd.a_LC

            a = predict_GUF_rd[predict_GUF_rd.b_cat == 1]
            predict_GUF_rd.loc[a.index, "LC"] = 5
            b = predict_GUF_rd[predict_GUF_rd.LC == 0]
            predict_GUF_rd.loc[b.index, "LC"] = 4
            c = predict_GUF_rd[predict_GUF_rd.LC == 5]
            predict_GUF_rd.loc[c.index, "LC"] = 0

            predict_GUF_rd = predict_GUF_rd.drop(
                ["cat", "a_cat", "b_cat", "a_LC", "b_FID"], axis=1
            )
            predict_GUF_rd_temp = predict_GUF_rd.buffer(0)
            predict_GUF_rd_temp = gpd.GeoDataFrame(predict_GUF_rd_temp)
            predict_GUF_rd_temp["LC"] = predict_GUF_rd["LC"]
            predict_GUF_rd_temp = predict_GUF_rd_temp.rename(columns={0: "geometry"})
            predict_GUF_rd_temp.crs = {"init": "epsg:4326"}
            print("dissolving the result . . . .")
            predict_GUF_rd_temp = predict_GUF_rd_temp.dissolve("LC")
            predict_GUF_rd_temp = predict_GUF_rd_temp.reset_index()
            predict_GUF_rd_temp.to_file(
                path_out + "/predict_GUF_roads" + "_mod" + str(patch_n)
            )

            print("Merging the predicted-GUF-roads to Building data . . .")
            time.sleep(3)
            v1_dir = (
                path_out
                + "/predict_GUF_roads_mod"
                + str(patch_n)
                + "/predict_GUF_roads_mod"
                + str(patch_n)
                + ".shp"
            )
            v2_dir = (
                path_out
                + "/"
                + Building_data
                + "_sh_clipped"
                + str(patch_n)
                + "/"
                + Building_data
                + "_sh_clipped"
                + str(patch_n)
                + ".shp"
            )
            out_dir = path_out + "/predict_GUF_roads_" + Building_data + str(patch_n)
            grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out)

            predict_GUF_rd_bd = gpd.read_file(
                path_out
                + "/predict_GUF_roads_"
                + Building_data
                + str(patch_n)
                + "/predict_GUF_roads_"
                + Building_data
                + str(patch_n)
                + ".shp"
            )
            predict_GUF_rd_bd["LC"] = predict_GUF_rd_bd.a_LC

            a = predict_GUF_rd_bd[predict_GUF_rd_bd.a_LC == 4]
            predict_GUF_rd_bd.loc[a.index, "LC"] = 5
            b = predict_GUF_rd_bd[predict_GUF_rd_bd.b_build == 1]
            predict_GUF_rd_bd.loc[b.index, "LC"] = 4

            predict_GUF_rd_bd = predict_GUF_rd_bd.drop(
                ["cat", "a_cat", "b_cat", "a_LC", "b_build"], axis=1
            )
            predict_GUF_rd_bd_temp = predict_GUF_rd_bd.buffer(0)
            predict_GUF_rd_bd_temp = gpd.GeoDataFrame(predict_GUF_rd_bd_temp)
            predict_GUF_rd_bd_temp["LC"] = predict_GUF_rd_bd["LC"]
            predict_GUF_rd_bd_temp = predict_GUF_rd_bd_temp.rename(
                columns={0: "geometry"}
            )
            predict_GUF_rd_bd_temp.crs = {"init": "epsg:4326"}
            print("dissolving the final result . . . .")
            predict_GUF_rd_bd_temp = predict_GUF_rd_bd_temp.dissolve("LC")
            predict_GUF_rd_bd_temp = predict_GUF_rd_bd_temp.reset_index()
            predict_GUF_rd_bd_temp.to_file(
                path_out + "/predict_GUF_roads_" + Building_data + "_mod" + str(patch_n)
            )

    else:
        if Building_data != "no" and Road_data == "no":
            print("Merging the predicted to Building data . . .")
            time.sleep(3)
            v1_dir = (
                path_out
                + "/predicted_shape"
                + str(patch_n)
                + "/predicted_shape"
                + str(patch_n)
                + ".shp"
            )
            v2_dir = (
                path_out
                + "/"
                + Building_data
                + "_sh_clipped"
                + str(patch_n)
                + "/"
                + Building_data
                + "_sh_clipped"
                + str(patch_n)
                + ".shp"
            )
            out_dir = path_out + "/predict_" + Building_data + str(patch_n)
            grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out)

            predict_bld = gpd.read_file(
                path_out
                + "/predict_"
                + Building_data
                + str(patch_n)
                + "/predict_"
                + Building_data
                + str(patch_n)
                + ".shp"
            )
            predict_bld["LC"] = predict_bld.a_predicte
            a = predict_bld[predict_bld.b_build == 1]
            predict_bld.loc[a.index, "LC"] = 3
            predict_bld = predict_bld.drop(
                ["a_cat", "b_cat", "a_predicte", "b_build"], axis=1
            )
            predict_bld_temp = predict_bld.buffer(0)
            predict_bld_temp = gpd.GeoDataFrame(predict_bld_temp)
            predict_bld_temp["LC"] = predict_bld["LC"]
            predict_bld_temp = predict_bld_temp.rename(columns={0: "geometry"})
            predict_bld_temp.crs = {"init": "epsg:4326"}
            print("dissolving the final result . . . .")
            predict_bld_temp = predict_bld_temp.dissolve("LC")
            predict_bld_temp = predict_bld_temp.reset_index()
            predict_bld_temp.to_file(
                path_out + "/predict_" + Building_data + "_mod" + str(patch_n)
            )
