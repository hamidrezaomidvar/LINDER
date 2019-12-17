import os
import sys
import rasterio
import geopandas as gpd

gisdb = os.path.join(os.path.expanduser("~"), "Documents")
location = "nc_spm_08"
mapset = "PERMANENT"
gisbase = "/Applications/GRASS-7.6.app/Contents/Resources"
os.environ["GISBASE"] = gisbase
grass_pydir = os.path.join(gisbase, "etc", "python")
sys.path.append(grass_pydir)
os.environ["LD_LIBRARY_PATH"] = os.path.join(gisbase,"lib")
import grass.script as gscript
import grass.script.setup as gsetup


def grass_overlay(v1_dir, v2_dir, out_dir, patch_n, path_out, how="or"):
    # TODO: check if gisdb/nc_spm_08 folders exist?
    gsetup.init(gisbase, gisdb, location, mapset)

    gscript.run_command(
        "g.proj", flags="c", proj4="+proj=longlat +datum=WGS84 +no_defs"
    )

    gscript.run_command(
        "v.in.ogr",
        min_area=0.0001,
        snap=-1.0,
        input=v1_dir,
        output="vector3",
        overwrite=True,
        flags="o",
    )

    gscript.run_command(
        "v.in.ogr",
        min_area=0.0001,
        snap=-1.0,
        input=v2_dir,
        output="vector4",
        overwrite=True,
        flags="o",
    )

    raster = rasterio.open(
        path_out + "/predicted_tiff/patch" + str(patch_n) + "/merged_prediction.tiff"
    )
    df = raster.bounds

    gscript.run_command("g.region", n=df.top, s=df.bottom, e=df.right, w=df.left)

    gscript.run_command(
        "v.overlay",
        overwrite=True,
        ainput="vector3",
        atype="area",
        binput="vector4",
        btype="area",
        operator=how,
        snap=0,
        output="output_b",
    )

    out_file = "converted_temp.geojson"
    gscript.run_command(
        "v.out.ogr",
        type="auto",
        input="output_b",
        output=out_file,
        format="GeoJSON",
        overwrite=True,
    )

    temp = gpd.read_file(out_file)
    os.remove(out_file)
    temp.crs = {"init": "epsg:4326"}
    temp.to_file(out_dir)
