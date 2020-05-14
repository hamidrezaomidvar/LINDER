from pathlib import Path
import linder as ld


lat_left_top_t, lon_left_top_t, = [51.501106, -0.519267]
lat_right_bot_t, lon_right_bot_t = [51.453078, -0.404774]
start, end = "20160101", "20181231"
path_GUF = Path('Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326/')

list_path_fraction = ld.get_land_cover(
    lat_left_top_t,
    lon_left_top_t,
    lat_right_bot_t,
    lon_right_bot_t,
    start,
    end,
    nx=1,
    ny=1,
    path_GUF=path_GUF,
    path_save=Path("London"),
    debug=True,
)
