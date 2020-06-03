from pathlib import Path
import linder as ld


lat_left_top_t, lon_left_top_t, = [6.994101, 79.809540]
lat_right_bot_t, lon_right_bot_t = [6.806518, 79.951731]
start, end = "20180101", "20181231"
path_GUF = Path('Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326/')

list_path_fraction = ld.get_land_cover(
    lat_left_top_t,
    lon_left_top_t,
    lat_right_bot_t,
    lon_right_bot_t,
    start,
    end,
    nx=1,
    ny=2,
    xn=40,
    yn=40,
    path_GUF=path_GUF,
    path_save=Path("Colombo"),
    debug=True,
)
