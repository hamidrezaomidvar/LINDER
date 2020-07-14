[![PyPI version](https://badge.fury.io/py/linder.svg)](https://badge.fury.io/py/linder)
[![Downloads](https://pepy.tech/badge/linder)](https://pepy.tech/project/linder)

# LINDER: Land use INDexER

A pipeline for calculating land cover over urban/rural areas using machine learning and other global datasets.

## How to use?

*A quick demo:*

```python
from pathlib import Path
import linder as ld


lat_left_top_t, lon_left_top_t, = [6.994101, 79.809540] #top left coordinates
lat_right_bot_t, lon_right_bot_t = [6.806518, 79.951731] #bottom right coordinates
start, end = "20180101", "20181231" # start and end dates to download images
path_GUF = Path('Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326/') # GUF data path

list_path_fraction = ld.get_land_cover(
    lat_left_top_t,
    lon_left_top_t,
    lat_right_bot_t,
    lon_right_bot_t,
    start,
    end,
    nx=1, # deviding the domain (number of devisions in x) for very large domains
    ny=1, # deviding the domain (number of devisions in y) for very large domains
    xn=40, # number of pixels in x direction to calculate fraction in the last step
    yn=40, # number of pixels in 7 direction to calculate fraction in the last step
    path_GUF=path_GUF,
    path_save=Path("./Colombo"), # directory name to save outputs
    debug=True,
)

# synthesise the above results to one `DataFrame`
df_lc = ld.proc_fraction(list_path_fraction)

```
You can see more example in this [directory](https://github.com/hamidrezaomidvar/LINDER/tree/master/misc) (London and Colombo)

## Instruction (for developers only, for macOS)
Please use the following steps to prepare required libraries:

1- Clone this package

2- use `conda` to create a fresh environment for this pipeline:
```zsh
conda env create -f GDAL.yml
```

Dependency details refer to [`GDAL.yml`](./GDAL.yml).

3- Change the environment to GDAL environment (from step 2):
```zsh
conda activate GDAL
```

4- Go to cloned LINDER folder and type:

```zsh
pip install -e src/
```
This will create a link to the src file. So any change in the src file is effective without the need of reinstalling the package.

5- for macOS, download and install the grass package:
```
http://grassmac.wikidot.com/downloads
```
For macOS, it should be install somewhere similar to `/Applications/GRASS-7.6.app/Contents/Resources`. This path is needed to be match in [here](https://github.com/hamidrezaomidvar/LINDER/blob/7a2d4c6783bc780903f33181a45491d6c9e508ae/src/linder/grass_util.py#L18) (you can find this file (`grass_util.py`) in the src folder)

6- `nc_spm_08` dataset: this dataset includes projection files required by `GRASS`.
download it [here](https://grassbook.org/datasets/datasets-3rd-edition/). Then its path needs to be specified in `grass_util.py` in this [line](https://github.com/hamidrezaomidvar/LINDER/blob/7a2d4c6783bc780903f33181a45491d6c9e508ae/src/linder/grass_util.py#L15). In the default case, it is `~/Documents`. If you put it in this folder, no change is needed in this line. The name of the subfolder is specified [here](https://github.com/hamidrezaomidvar/LINDER/blob/7a2d4c6783bc780903f33181a45491d6c9e508ae/src/linder/grass_util.py#L16) (no need to be changed if it is he same for your case)


7- sentinel-hub: Refer to [this page for setting up a new configuration](https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html#Requirements).
Then update the `sentinelhub` instance ID as follows:
```
sentinelhub.config --instance_id [your-instance-ID]
```

8- `GUF` dataset (optional) [GUF (Global Urban Footprint)](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-9628/16557_read-40454/) is a global urban coverage dataset produced by DLR.
This pipeline use `GUF` to improve accuracy in predicting urban features (You should be able to run the model without this data set, underdevelopment)

9- Use the demo script (top) to run your case by specifying the coordinates.

## Other details


### `OpenStreetMap` data for building and road network
[OSM](https://www.openstreetmap.org/) data is used automatically through the OSM API (for some cities, OSM is not complete and this limitation can be solved by only using road network and consider the rest of urban category as building). The road network needs a [setting](https://github.com/hamidrezaomidvar/LINDER/blob/master/src/linder/road_width.json) file to specify the road width based on its category (currently based on London roads).

### `Microsoft Building Footprint` for USA
Only for the USA, you will be able to use [Mictosoft building footprint dataset](https://github.com/Microsoft/USBuildingFootprints).

### methods
This is a pipeline for calculating the landcover over desired regions. It includes:

- Step 1: Getting the location of the region (`lat` and `lon`)

- Step 2: Getting the satellite image for the desired region

- Step 3: Prediction of the land cover into 3 categories:
  - 1-Urban
  - 2-Green
  - 3-Water and
  - 4-other

- Step 4: Overlaying the [GUF](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-9628/16557_read-40454/) data into the prediction. Therefore the landcover becomes 1-Water 2-Green 3-Urban 4-Other

- Step 5: Overlaying the building data from [OSM](https://osmbuildings.org/) or [Microsoft data](https://github.com/microsoft/USBuildingFootprints) (only for the US). Therefore, the final land cover includes 1-Water 2-Green 3-Buildings 4-Paved 5-Others

Using this pipeline is as simple as choosing the coordinates, and providing the GUF data and/or building data. The user can choose which data is available. For example, in the case of no GUF data, the pipeline uses the prediction, or in the case of no Microsoft data, the pipeline uses the OSM data for buildings. Note that adding GUF data makes the final result of the landcover more accurate.

Some technical details:

- Merging various maps might be very computationally expensive, and the current Python packages like GDAL are not very efficient.
  The pipeline instead uses a python interface to use GRASS functions (such as `v.overlay`) directly to speed up the merging processes.

- The pipeline uses a pre-trained model to predict the land cover.
  Currently, the model is trained over Colombo, but various tests has shown it has a good performance on other places as well.
  A more sophisticated model can be trained by using more datasets.

- Note that while OSM data are automatically fetched from the website for the desired region, the Microsoft data need to be downloaded for the chosen location manually. This can be automated in the future.


## Some examples

**Colombo, Sri Lanka**

![Colombo, Sri Lanka](https://github.com/hamidrezaomidvar/LINDER/raw/master/examples/Colombo.png)

**Pittsburgh, United States**

![Pittsburgh, United States](https://github.com/hamidrezaomidvar/LINDER/raw/master/examples/Pittsburgh.png)
