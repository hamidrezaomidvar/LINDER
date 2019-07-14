# Pipeline for calculating land cover over urban/rural areas:
This is a pipeline for calculating the landcover over desired regions. It includes:

- Step 1: Getting the location of the region (lat and lon)

- Step 2: Getting the satellite image for the desired region

- Step 3: Prediction of the land cover into 3 gategories: 1-Water 2-Green 3-Urban and other

- Step 4: Overlaying the [GUF](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-9628/16557_read-40454/) data into the prediction. Therefore the landcover becomes 1-Water 2-Green 3-Urban 4-Other

- Step 5: Overlaying the building data from [OSM](https://osmbuildings.org/) or [Microsoft data](https://github.com/microsoft/USBuildingFootprints) (only for the US). Therefore, the final land cover includes 1-Water 2-Green 3-Buildings 4-Paved 5-Others

Using this pipeline is as simple as choosing the coordinates, and providing the GUF data and/or building data. The user can choose which data is available. For example, in the case of no GUF data, the pipeline uses the prediction, or in the case of no Microsoft data, the pipeline uses the OSM data for buildings. Note that adding GUF data makes the final result of the landcover more accurate.

Some technical details:

- Merging various maps might be very computationally expensive, and the current Python packages like GDAL are not very efficients. The pipeline instead uses a python interface to use GRASS functions (such as `v.overlay`) directly to speed up the merging processes. 

- The pipeline uses a pre-trained model to predict the land cover. Currently, the model is trained over Colombo, but various tests has shown it has a good perfomance on other places as well. A more sophesticated model can be trained by using more datasets.

- Note that while OSM data are automatically fetched from the website for the desired region, the Microsoft data are needed to be downloaded for the chosen location. This can be improved in the future.


## Some examples

**Colombo, Sri Lanka**

![](./Examples/Colombo.png)


**Matara, Sri Lanka**

![](./Examples/Matara.png)

**Jaffna, Sri Lanka**

![](./Examples/Jaffna.png)
