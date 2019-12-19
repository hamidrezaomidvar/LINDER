import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
from .grass_util import grass_overlay


def calculate_fraction(path_out, patch_n, xn, yn):
    print("calculating the land cover fraction . . .")
    box = gpd.read_file(
        path_out + "/shape_box" + str(patch_n) + "/shape_box" + str(patch_n) + ".shp"
    )
    xmin, ymin, xmax, ymax = box.total_bounds
    cols = list(np.linspace(xmin, xmax, xn, endpoint=True))
    rows = list(np.linspace(ymin, ymax, yn, endpoint=True))
    rows.reverse()

    polygons = []
    X, Y = np.meshgrid(cols, rows)
    for x in range(xn - 1):
        for y in range(yn - 1):
            polygons.append(
                Polygon(
                    [
                        (X[y, x], Y[y, x]),
                        (X[y + 1, x], Y[y + 1, x]),
                        (X[y + 1, x + 1], Y[y + 1, x + 1]),
                        (X[y, x + 1], Y[y, x + 1]),
                    ]
                )
            )

    grid = gpd.GeoDataFrame({"geometry": polygons})
    grid.crs = box.crs
    grid["grid_num"] = grid.index
    grid["grid_area"] = grid.area
    grid.to_file(path_out + "/grid" + str(patch_n))
    v1_dir = (
        path_out
        + "/predict_GUF_roads_mod"
        + str(patch_n)
        + "/predict_GUF_roads_mod"
        + str(patch_n)
        + ".shp"
    )
    v2_dir = path_out + "/grid" + str(patch_n) + "/grid" + str(patch_n) + ".shp"
    out_dir = path_out + "/grid_intersect" + str(patch_n)
    grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out, how="and")
    grid_intersect = gpd.read_file(out_dir)
    grid_intersect = grid_intersect.drop(["cat", "a_cat", "b_cat"], axis=1)
    grid_intersect["area"] = grid_intersect.area

    temp = grid_intersect.groupby(["b_grid_num", "a_LC"]).sum()
    temp.b_grid_are = grid_intersect.b_grid_are.iloc[0]
    temp["percentage"] = temp.area / temp.b_grid_are
    temp = temp.reset_index()

    list_LC = gpd.read_file(v1_dir).LC.unique()
    centroids = grid.centroid
    fraction = pd.DataFrame(
        columns=np.concatenate((["lat", "lon"], list_LC)), index=centroids.index
    )
    for i in temp.b_grid_num.unique():

        a_list = []
        for x in list_LC:
            try:
                a = temp[(temp.b_grid_num == i) & (temp.a_LC == x)].percentage.values[0]
            except:
                a = 0
            a_list.append(a)

        aa = np.array(a_list)
        for j in aa:
            dy = (1 - np.sum(a_list)) / len(aa[aa != 0])
            if j != 0:
                aa[aa == j] = j + dy
        a_list = aa.T.tolist()
        fraction.loc[i] = [centroids.loc[i].y, centroids.loc[i].x] + a_list

    fraction.to_csv(path_out + "/fraction" + str(patch_n) + ".csv")
