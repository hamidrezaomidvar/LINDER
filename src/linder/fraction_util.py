import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pandas as pd

# from .grass_util import grass_overlay
from .task_util import merge_vector_data
from pathlib import Path


def calculate_fraction(path_shp, path_raster, xn=20, yn=20, debug=False):
    # cast to Path
    path_out = Path(path_shp).resolve().parent

    # retrieve name for this job from `path_shp`
    name_job = path_shp.stem.split('_')[-1]
    if debug:
        print(f"calculating the land cover fraction for patch-image {name_job} ...")
    box = gpd.read_file(path_shp)
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
    grid.to_file(path_out / f"grid_{name_job}")
    # key names
    name_v1 = f"predict_GUF_roads_mod_{name_job}"
    name_v2 = f"grid_{name_job}"
    grid_intersect = merge_vector_data(path_out, path_raster, name_v1, name_v2)
    path_fn_v1 = Path(f"{name_v1}.shp")
    path_dir_v1 = Path(path_out) / path_fn_v1.stem

    grid_intersect = grid_intersect.drop(["cat", "a_cat", "b_cat"], axis=1)
    grid_intersect["area"] = grid_intersect.area

    temp = grid_intersect.groupby(["b_grid_num", "a_LC"]).sum()
    temp.b_grid_are = grid_intersect.b_grid_are.iloc[0]
    temp["percentage"] = temp.area / temp.b_grid_are
    temp = temp.reset_index()

    list_LC = gpd.read_file(path_dir_v1).LC.unique().astype(int)
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

    path_fraction = path_out / f"fraction_{name_job}.csv"

    # rename columns with explicit LC names
    dict_cat4 = {
        # '0': 'water', '1': 'green', '2': 'urban', '3': 'other',
        '0': 'urban', '1': 'green', '2': 'water', '3': 'other',
        # '0': 'road', '1': 'green', '2': 'water', '3': 'other', '4': 'building'
    }
    dict_cat5 = {
        # '0': 'water', '1': 'green', '2': 'building', '3': 'paved', '4': 'other'
        '0': 'road', '1': 'green', '2': 'water', '3': 'other', '4': 'building'
    }
    dict_use = dict_cat4 if fraction.columns[2:].max() == 3 else dict_cat5
    fraction = fraction.rename(dict_use, axis=1)

    fraction.to_csv(path_fraction)

    return path_fraction


def proc_fraction(list_path_fraction: list) -> pd.DataFrame:
    # load all fraction info into one DataFrame
    df_lc_raw = pd.concat(
        [pd.read_csv(p, index_col=[0]) for p in list_path_fraction],
        keys=[i for i, p in enumerate(list_path_fraction)],
    ).unstack(0)

    # calculate median values of each land cover type
    df_lc_median = df_lc_raw.median(level=0, axis=1).set_index(['lat', 'lon'])

    # normalise values of each grid to make sums to ONE
    df_lc = df_lc_median.apply(lambda ser: ser / ser.sum(), axis=1)
    return df_lc
