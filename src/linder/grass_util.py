import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import rasterio

import random
import string
from shutil import rmtree, copytree


def grass_overlay(path_v1, path_v2, path_raster, how="or"):
    gisdb = os.path.join(os.path.expanduser("~"), "Documents")
    location = "nc_spm_08"
    mapset = "PERMANENT"
    gisbase = "/Applications/GRASS-7.6.app/Contents/Resources"
    os.environ["GISBASE"] = gisbase
    grass_pydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(grass_pydir)
    os.environ["LD_LIBRARY_PATH"] = os.path.join(gisbase, "lib")

    import grass.script as gscript
    import grass.script.setup as gsetup

    # start = time.time()

    job_id = "p".join([x.split("_")[-1] for x in path_raster.stem.split("-")[-2:]])
    job_id = "p" + job_id + "_" + randomString()
    v1_name = path_v1.stem.replace("-", "_") + randomString()
    v2_name = path_v2.stem.replace("-", "_") + randomString()

    # set up GISDB and validate the path
    path_gisdb = Path(gisdb)
    path_location_default = (path_gisdb / location).resolve()
    if not (path_location_default / mapset / "DEFAULT_WIND").exists():
        raise RuntimeError("GISDB is NOT properly set for GRASS.")

    # localise GISDB to avoid possible conflict in parallel mode
    path_location_local = path_gisdb / job_id
    copytree(path_location_default, path_location_local)

    gsetup.init(gisbase, gisdb, path_location_local.stem, mapset)

    # print(f'g.proj: {job_id}')
    gscript.run_command(
        "g.proj", flags="c", proj4="+proj=longlat +datum=WGS84 +no_defs"
    )

    # print(f'v.in.ogr, vector 3: {path_v1}')
    gscript.run_command(
        "v.in.ogr",
        min_area=0.0001,
        snap=-1.0,
        input=path_v1,
        output=v1_name,
        overwrite=True,
        flags="o",
    )

    # print(f'v.in.ogr, vector 4: {path_v2}')
    gscript.run_command(
        "v.in.ogr",
        min_area=0.0001,
        snap=-1.0,
        input=path_v2,
        output=v2_name,
        overwrite=True,
        flags="o",
    )

    raster = rasterio.open(path_raster)
    df = raster.bounds

    # print(f'g.region: {job_id}')
    gscript.run_command("g.region", n=df.top, s=df.bottom, e=df.right, w=df.left)

    # print(f'v.overlay: {job_id}')
    gscript.run_command(
        "v.overlay",
        overwrite=True,
        ainput=v1_name,
        atype="area",
        binput=v2_name,
        btype="area",
        operator=how,
        snap=0,
        output=f"{job_id}_output_b",
    )
    out_file = f"{job_id}.geojson"
    force_del(out_file)
    # print(f'v.out.ogr: {job_id}, {out_file}')
    gscript.run_command(
        "v.out.ogr",
        type="auto",
        input=f"{job_id}_output_b",
        output=out_file,
        format="GeoJSON",
        overwrite=True,
    )

    # end = time.time()
    # print(f'time spent: {end - start:.2f} s\n')

    gdf_merge = gpd.read_file(out_file)
    force_del(out_file)

    gdf_merge.crs = {"init": "epsg:4326"}
    gdf_merge.to_file(out_file)
    gdf_merge = gpd.read_file(out_file)
    force_del(out_file)

    # remove local GISDB
    force_del(path_location_local)

    return gdf_merge


def force_del(out_file):
    if os.path.exists(out_file):
        if os.path.isdir(out_file):
            rmtree(out_file)
        else:
            os.remove(out_file)


def force_rm_path(path: Path):
    path = Path(path)
    # recursively remove files
    for child in path.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            force_rm_path(child)
    # remove the root directory once empty
    path.rmdir()


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))
