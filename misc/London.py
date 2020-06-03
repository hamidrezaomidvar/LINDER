from pathlib import Path
import linder as ld


lat_left_top_t, lon_left_top_t, = [51.541148, -0.239574]
lat_right_bot_t, lon_right_bot_t = [51.465183, 0.016867]
start, end = "20160101", "20161231"
path_GUF = Path('Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326/')

list_path_fraction = ld.get_land_cover(
    lat_left_top_t,
    lon_left_top_t,
    lat_right_bot_t,
    lon_right_bot_t,
    start,
    end,
    nx=2,
    ny=2,
    xn=40,
    yn=40,
    path_GUF=path_GUF,
    path_save=Path("London"),
    debug=True,
)
