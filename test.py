# %%
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
os.environ["LD_LIBRARY_PATH"] = "/Applications/GRASS-7.6.app/Contents/Resources/lib"
import grass.script as gscript
import grass.script.setup as gsetup

# %%
gsetup.init(gisbase, gisdb, location, mapset)

gscript.run_command("g.proj", flags="c", proj4="+proj=longlat +datum=WGS84 +no_defs")
