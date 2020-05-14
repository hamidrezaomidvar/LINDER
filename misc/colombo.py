from pathlib import Path
import linder as ld


lat_left_top_t, lon_left_top_t, = [6.962103, 79.835858]
lat_right_bot_t, lon_right_bot_t = [6.910388, 79.892159]
start, end = "20180101", "20181231"
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
    path_GUF=path_GUF,
    path_save=Path("Colombo"),
    debug=True,
)
