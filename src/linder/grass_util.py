import os
import sys
import time

import geopandas as gpd
import rasterio

import random
import string

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


def grass_overlay(v1_dir, v2_dir, out_dir, path_raster, how="or"):
    start = time.time()

    job_id = 'p'.join([x.split('_')[-1] for x in path_raster.stem.split('-')[-2:]])
    job_id = 'p' + job_id
    print(f'\nworking on {job_id}')
    print(f'{v1_dir}, {v2_dir}')
    print(f'gsetup.init: {job_id}')
    # TODO: check if gisdb/nc_spm_08 folders exist?
    gsetup.init(gisbase, gisdb, location, mapset)

    print(f'g.proj: {job_id}')
    gscript.run_command(
        "g.proj", flags="c", proj4="+proj=longlat +datum=WGS84 +no_defs"
    )

    print(f'v.in.ogr, vector 3: {job_id}')
    gscript.run_command(
        "v.in.ogr",
        min_area=0.0001,
        snap=-1.0,
        input=v1_dir,
        output=f"{job_id}_vector3",
        overwrite=True,
        flags="o",
    )

    print(f'v.in.ogr, vector 4: {job_id}')
    gscript.run_command(
        "v.in.ogr",
        min_area=0.0001,
        snap=-1.0,
        input=v2_dir,
        output=f"{job_id}_vector4",
        overwrite=True,
        flags="o",
    )

    raster = rasterio.open(path_raster)
    df = raster.bounds

    print(f'g.region: {job_id}')
    gscript.run_command("g.region", n=df.top, s=df.bottom, e=df.right, w=df.left)

    print(f'v.overlay: {job_id}')
    gscript.run_command(
        "v.overlay",
        overwrite=True,
        ainput=f"{job_id}_vector3",
        atype="area",
        binput=f"{job_id}_vector4",
        btype="area",
        operator=how,
        snap=0,
        output=f"{job_id}_output_b",
    )
    out_file = f"{job_id}.geojson"
    print(f'v.out.ogr: {job_id}, {out_file}')
    gscript.run_command(
        "v.out.ogr",
        type="auto",
        input=f"{job_id}_output_b",
        output=out_file,
        format="GeoJSON",
        overwrite=True,
    )

    # gsetup.finish()

    # this `sleep` is to allow enough time for writing out geojson before loading by geopandas
    # time.sleep(4)

    temp = gpd.read_file(out_file)
    os.remove(out_file)

    temp.crs = {"init": "epsg:4326"}
    temp.to_file(out_dir)
    end = time.time()
    print(f'time spent: {end - start:.2f} s\n')


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
