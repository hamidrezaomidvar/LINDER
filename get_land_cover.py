import json
from tqdm import tqdm
from glob import glob
import pickle
import sys
import os
import datetime
import itertools
from enum import Enum
import time
import requests
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
from tqdm import tqdm_notebook as tqdm
from pyproj import Proj, transform
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import preprocessing
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadFromDisk, SaveToDisk, EOExecutor
from eolearn.io import S2L1CWCSInput, ExportToTiff
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask
from eolearn.geometry import VectorToRaster, PointSamplingTask, ErosionTask
from eolearn.features import LinearInterpolation, SimpleFilterTask
from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam
import rasterio
from rasterio.plot import show
from shapely.geometry import box
import fiona
import rasterio.mask
from rasterio.features import shapes
from pathlib import Path
import osmnx as ox


gisdb = os.path.join(os.path.expanduser("~"), "Documents")
location = "nc_spm_08"
mapset = "PERMANENT"
gisbase='/Applications/GRASS-7.6.app/Contents/Resources'
os.environ['GISBASE'] = gisbase
grass_pydir = os.path.join(gisbase, "etc", "python")
sys.path.append(grass_pydir)
os.environ['LD_LIBRARY_PATH']="/Applications/GRASS-7.6.app/Contents/Resources/lib"
import grass.script as gscript
import grass.script.setup as gsetup


with open('./setting.json') as setting_file:
    settings = json.load(setting_file)

cname = 'London'
nx=1
ny=1
downloading_img=settings[cname]['downloading_img']=='yes'
s_date = settings[cname]['s_date']
e_date = settings[cname]['e_date']


[lat_left_top_t, lon_left_top_t] = settings[cname]['coord_top']  # Pittsburgh
[lat_right_bot_t, lon_right_bot_t] = settings[cname]['coord_bot']

size = 10
scale = abs(lat_left_top_t-lat_right_bot_t)/abs(lon_left_top_t-lon_right_bot_t)

GUF_data = settings[cname]['GUF_data'] == 'yes'

Building_data = settings[cname]['building_data']
Road_data = settings[cname]['road_data']
if Building_data == 'MICROSOFT':
    building_dir = settings[cname]['building_dir']


list_of_GUF=sorted(glob('./Data/GUF/WSF2015_v1_EPSG4326/WSF2015_v1_EPSG4326/*'))

class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                            np.logical_not(eopatch.mask['CLM'].astype(np.bool)))

class CountValid(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.name,
                            np.count_nonzero(eopatch.mask[self.what], axis=0))

        return eopatch

class NormalizedDifferenceIndex(EOTask):
    """
    The tasks calculates user defined Normalised Difference Index (NDI) between two bands A and B as:
    NDI = (A-B)/(A+B).
    """

    def __init__(self, feature_name, band_a, band_b):
        self.feature_name = feature_name
        self.band_a_fetaure_name = band_a.split('/')[0]
        self.band_b_fetaure_name = band_b.split('/')[0]
        self.band_a_fetaure_idx = int(band_a.split('/')[-1])
        self.band_b_fetaure_idx = int(band_b.split('/')[-1])

    def execute(self, eopatch):
        band_a = eopatch.data[self.band_a_fetaure_name][...,
                                                        self.band_a_fetaure_idx]
        band_b = eopatch.data[self.band_b_fetaure_name][...,
                                                        self.band_b_fetaure_idx]

        ndi = (band_a - band_b) / (band_a + band_b)

        eopatch.add_feature(
            FeatureType.DATA, self.feature_name, ndi[..., np.newaxis])

        return eopatch

class EuclideanNorm(EOTask):
    """
    The tasks calculates Euclidian Norm of all bands within an array:
    norm = sqrt(sum_i Bi**2),
    where Bi are the individual bands within user-specified feature array.
    """

    def __init__(self, feature_name, in_feature_name):
        self.feature_name = feature_name
        self.in_feature_name = in_feature_name

    def execute(self, eopatch):
        arr = eopatch.data[self.in_feature_name]
        norm = np.sqrt(np.sum(arr**2, axis=-1))

        eopatch.add_feature(
            FeatureType.DATA, self.feature_name, norm[..., np.newaxis])
        return eopatch

class LULC(Enum):
    OTHER              = (0,  'Paved+Built+other',              'red')
    VEG                = (1,  'Veg',                            'green')
    Water              = (2,  'Water',                          'blue')
    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3

class ConcatenateData(EOTask):
    
    """ Task to concatenate data arrays along the last dimension
    """
    def __init__(self, feature_name, feature_names_to_concatenate):
        self.feature_name = feature_name
        self.feature_names_to_concatenate = feature_names_to_concatenate
    def execute(self, eopatch):
        arrays = [eopatch.data[name] for name in self.feature_names_to_concatenate]
        eopatch.add_feature(FeatureType.DATA, self.feature_name, np.concatenate(arrays, axis=-1))
        return eopatch

class ValidDataFractionPredicate:
    """ Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold

class PredictPatch(EOTask):
            """
            Task to make model predictions on a patch. Provide the model and the feature,
            and the output names of labels and scores (optional)
            """
            def __init__(self, model, features_feature, predicted_labels_name, pic_n,predicted_scores_name=None):
                self.model = model
                self.features_feature = features_feature
                self.predicted_labels_name = predicted_labels_name
                self.predicted_scores_name = predicted_scores_name
                self.pic_n = pic_n

            def execute(self, eopatch):
                ftrs = eopatch[self.features_feature[0]][self.features_feature[1]][self.pic_n,:,:,:]
                #ftrs = np.mean(eopatch[self.features_feature[0]][self.features_feature[1]][:,:,:,:],axis=0)
                #ftrs = eopatch[self.features_feature[0]][self.features_feature[1]][pic_n,:,:,7]

                #t, w, h, f = ftrs.shape
                w, h, f = ftrs.shape
                #w, h = ftrs.shape
                
                #ftrs = np.moveaxis(ftrs, 0, 2).reshape(w * h, 1 * f)
                ftrs = ftrs.reshape(w * h, 1 * f)
                #ftrs = ftrs.reshape(w * h, 1 * 1)

                plabels = self.model.predict(ftrs)
                plabels = plabels.reshape(w, h)
                plabels = plabels[..., np.newaxis]
                eopatch.add_feature(FeatureType.MASK_TIMELESS, self.predicted_labels_name, plabels)

                if self.predicted_scores_name:
                    pscores = self.model.predict_proba(ftrs)
                    _, d = pscores.shape
                    pscores = pscores.reshape(w, h, d)
                    eopatch.add_feature(FeatureType.DATA_TIMELESS, self.predicted_scores_name, pscores)

                return eopatch

def clip_points(shp, clip_obj):
    '''
    Docs Here
    '''

    poly = clip_obj.geometry.unary_union
    return(shp[shp.geometry.intersects(poly)])

def clip_line_poly(shp, clip_obj):
    '''
    docs
    '''

    # Create a single polygon object for clipping
    poly = clip_obj.geometry.unary_union
    spatial_index = shp.sindex

    # Create a box for the initial intersection
    bbox = poly.bounds
    # Get a list of id's for each road line that overlaps the bounding box and subset the data to just those lines
    sidx = list(spatial_index.intersection(bbox))
    shp_sub = shp.iloc[sidx]

    # Clip the data - with these data
    clipped = shp_sub.copy()
    clipped['geometry'] = shp_sub.intersection(poly)

    # Return the clipped layer with no null geometry values
    return(clipped[clipped.geometry.notnull()])

def clip_shp(shp, clip_obj):
    '''
    '''
    if shp["geometry"].iloc[0].type == "Point":
        return(clip_points(shp, clip_obj))
    else:
        return(clip_line_poly(shp, clip_obj))

def grass_union(v1_dir,v2_dir,out_dir,patch_n):
    gsetup.init(gisbase, gisdb, location, mapset)

    gscript.run_command("g.proj",flags="c" ,proj4="+proj=longlat +datum=WGS84 +no_defs")

    gscript.run_command("v.in.ogr", 
    min_area=0.0001 ,
    snap=-1.0, 
    input=v1_dir, 
    output="vector3", 
    overwrite=True, 
    flags="o")

    gscript.run_command("v.in.ogr", 
    min_area=0.0001 ,
    snap=-1.0, 
    input=v2_dir, 
    output="vector4", 
    overwrite=True, 
    flags="o")

    raster=rasterio.open(path_out+'/predicted_tiff/patch'+str(patch_n)+'/merged_prediction.tiff')
    df=raster.bounds

    gscript.run_command("g.region",
    n=df.top,
    s=df.bottom,
    e=df.right,
    w=df.left)

    gscript.run_command("v.overlay", 
    overwrite=True,
    ainput="vector3",
    atype="area", 
    binput="vector4", 
    btype="area", 
    operator='or', 
    snap=0, 
    output="output_b")

    out_file='converted_temp.geojson'
    gscript.run_command("v.out.ogr",
    type="auto",
    input="output_b",
    output=out_file,
    format="GeoJSON",
    overwrite=True)


    temp=gpd.read_file(out_file)
    os.remove(out_file)
    temp.crs={'init' :'epsg:4326'}
    temp.to_file(out_dir)

def download_data(path_out,coords_top,coords_bot,patch_n):
    [lat_left_top, lon_left_top]=coords_top
    [lat_right_bot, lon_right_bot]=coords_bot
    # TASK FOR BAND DATA
    # add a request for B(B02), G(B03), R(B04), NIR (B08), SWIR1(B11), SWIR2(B12)
    # from default layer 'ALL_BANDS' at 10m resolution
    # Here we also do a simple filter of cloudy scenes. A detailed cloud cover
    # detection is performed in the next step
    custom_script = 'return [B02, B03, B04, B08, B11, B12];'
    add_data = S2L1CWCSInput(
        layer='BANDS-S2-L1C',
        feature=(FeatureType.DATA, 'BANDS'),  # save under name 'BANDS'
        # custom url for 6 specific bands
        custom_url_params={CustomUrlParam.EVALSCRIPT: custom_script},
        resx='10m',  # resolution x
        resy='10m',  # resolution y
        maxcc=0.1,  # maximum allowed cloud cover of original ESA tiles
    )

    # TASK FOR CLOUD INFO
    # cloud detection is performed at 80m resolution
    # and the resulting cloud probability map and mask
    # are scaled to EOPatch's resolution
    cloud_classifier = get_s2_pixel_cloud_detector(
        average_over=2, dilation_size=1, all_bands=False)
    add_clm = AddCloudMaskTask(cloud_classifier, 'BANDS-S2CLOUDLESS', cm_size_y='80m', cm_size_x='80m',
                            cmask_feature='CLM',  # cloud mask name
                            cprobs_feature='CLP'  # cloud prob. map name
                            )

    # TASKS FOR CALCULATING NEW FEATURES
    # NDVI: (B08 - B04)/(B08 + B04)
    # NDWI: (B03 - B08)/(B03 + B08)
    # NORM: sqrt(B02^2 + B03^2 + B04^2 + B08^2 + B11^2 + B12^2)
    ndvi = NormalizedDifferenceIndex('NDVI', 'BANDS/3', 'BANDS/2')
    ndwi = NormalizedDifferenceIndex('NDWI', 'BANDS/1', 'BANDS/3')
    norm = EuclideanNorm('NORM', 'BANDS')

    # TASK FOR VALID MASK
    # validate pixels using SentinelHub's cloud detection mask and region of acquisition
    add_sh_valmask = AddValidDataMaskTask(SentinelHubValidData(),
                                        'IS_VALID'  # name of output mask
                                        )

    # TASK FOR COUNTING VALID PIXELS
    # count number of valid observations per pixel using valid data mask
    count_val_sh = CountValid('IS_VALID',  # name of existing mask
                            'VALID_COUNT'  # name of output scalar
                            )

    # TASK FOR SAVING TO OUTPUT (if needed)
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    save = SaveToDisk(path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


    # Define the workflow
    workflow = LinearWorkflow(
        add_data,
        add_clm,
        ndvi,
        ndwi,
        norm,
        add_sh_valmask,
        count_val_sh,
        save
    )
    # Execute the workflow

    time_interval = [s_date, e_date]  # time interval for the SH request

    # define additional parameters of the workflow
    execution_args = []

    execution_args.append({
        add_data: {'bbox': BBox(((lon_left_top, lat_left_top), (lon_right_bot, lat_right_bot)), crs=CRS.WGS84),
                'time_interval': time_interval},
        save: {'eopatch_folder': 'eopatch_{}'.format(patch_n)}
    })

    if downloading_img:
        executor = EOExecutor(workflow, execution_args, save_logs=True)

        print('Downloading Satellite data . . .')

        executor.run(workers=5, multiprocess=True)

        executor.make_report()
        print('Satellite data is downloaded')

def save_images(path_out,patch_n):
    # Draw the RGB image
    size=20
    eopatch = EOPatch.load('{}/eopatch_{}'.format(path_out, patch_n), lazy_loading=True)
    image_dir=path_out+'/images/patch_'+str(patch_n)
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    print('saving the images into '+image_dir+'. . .')
    n_pics=eopatch.data['BANDS'].shape[0]
    for i in range(n_pics):
        
        fig = plt.figure(figsize=(size*1, size*scale))
        ax = plt.subplot(1, 1, 1)
        plt.imshow(np.clip(eopatch.data['BANDS'][i][..., [2,1,0]] * 3.5, 0, 1))
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect("auto")
        dr=image_dir+'/'+str(i)+'.png'
        print('Saving '+dr)
        plt.savefig(dr)
        plt.close()

def predict_image(path_out,patch_n):
    model_path = './model.pkl'
    model = joblib.load(model_path)

    cnt='n'
    while cnt == 'n':

        pic_n=int(input('Please choose the desired image number: '))            
        # TASK TO LOAD EXISTING EOPATCHES
        load = LoadFromDisk(path_out)

        # TASK FOR CONCATENATION
        concatenate = ConcatenateData('FEATURES', ['BANDS', 'NDVI', 'NDWI', 'NORM'])
        #concatenate = ConcatenateData('FEATURES', ['BANDS'])
        # TASK FOR FILTERING OUT TOO CLOUDY SCENES
        # keep frames with > 80 % valid coverage
        # valid_data_predicate = ValidDataFractionPredicate(0.8)
        # filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)

        save = SaveToDisk(path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

        workflow = LinearWorkflow(
            load,
            concatenate,
        #    filter_task,
        #    linear_interp,
        #     erosion,
        #     spatial_sampling,
            save
        )

        execution_args = []
        for idx in range(0,1):
            execution_args.append({
                load: {'eopatch_folder': 'eopatch_{}'.format(patch_n)},
                save: {'eopatch_folder': 'eopatch_{}'.format(patch_n)}
            })

        print('Saving the features . . .')
        executor = EOExecutor(workflow, execution_args, save_logs=False)
        executor.run(workers=5, multiprocess=True)

        executor.make_report()

            # TASK TO LOAD EXISTING EOPATCHES
        load = LoadFromDisk(path_out)

        # TASK FOR PREDICTION
        predict = PredictPatch(model, (FeatureType.DATA, 'FEATURES'), 'LBL',pic_n ,'SCR')

        # TASK FOR SAVING
        #path_out_sampled = './eopatches_sampled_small_'+cname+'_2/' if use_smaller_patches else './eopatches_sampled_large/'
        #if not os.path.isdir(path_out_sampled):
        #    os.makedirs(path_out_sampled)
        save = SaveToDisk(str(path_out), overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

        # TASK TO EXPORT TIFF
        export_tiff = ExportToTiff((FeatureType.MASK_TIMELESS, 'LBL'))
        tiff_location = path_out+'/predicted_tiff/patch'+str(patch_n)+'/'

        if not os.path.isdir(tiff_location):
            os.makedirs(tiff_location)

        workflow = LinearWorkflow(
            load,
            predict,
            export_tiff,
            save
        )


        # create a list of execution arguments for each patch
        execution_args = []
        for i in range(0,1):
            execution_args.append(
                {
                    load: {'eopatch_folder': 'eopatch_{}'.format(patch_n)},
                    export_tiff: {'filename': '{}/prediction_eopatch_{}.tiff'.format(tiff_location, i)},
                    save: {'eopatch_folder': 'eopatch_{}'.format(patch_n)}
                }
            )

        # run the executor on 2 cores
        executor = EOExecutor(workflow, execution_args)

        # uncomment below save the logs in the current directory and produce a report!
        #executor = EOExecutor(workflow, execution_args, save_logs=True)
        print('Predicting the land cover . . .')
        executor.run(workers=5, multiprocess=True)
        executor.make_report()

        PATH=path_out+'/predicted_tiff/patch'+str(patch_n)
        if os.path.exists(PATH+'/merged_prediction.tiff'):
            os.remove(PATH+'/merged_prediction.tiff')
        cmd="gdal_merge.py -o "+PATH+"/merged_prediction.tiff -co compress=LZW "+PATH+"/prediction_eopatch_*"
        os.system(cmd)



        # Reference colormap things
        lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
        lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 3, 1), lulc_cmap.N)


        size=20
        fig, ax = plt.subplots(figsize=(2*size*1, 1*size*scale), nrows=1, ncols=2)
        eopatch = EOPatch.load('{}/eopatch_{}'.format(path_out, patch_n), lazy_loading=True)
        im = ax[0].imshow(eopatch.mask_timeless['LBL'].squeeze(),cmap=lulc_cmap, norm=lulc_norm)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_aspect("auto")

        fig.subplots_adjust(wspace=0, hspace=0)
        for i in range(0,1):
            eopatch = EOPatch.load('{}/eopatch_{}'.format(path_out, patch_n), lazy_loading=True)
            ax = ax[1]
            plt.imshow(np.clip(eopatch.data['BANDS'][pic_n,:,:,:][..., [2,1,0]] * 3.5, 0, 1))
            plt.xticks([])
            plt.yticks([])
            ax.set_aspect("auto")
            del eopatch

        print('saving the predicted image . . .')    
        plt.savefig(path_out+'/predicted_vs_real_'+str(patch_n)+'.png')

        cnt=input('Is the first stage predicted image good for patch '+str(patch_n)+'? (y/n):')
        while (cnt not in ['y','n']):
            cnt=input('Is the first stage predicted image good for patch '+str(patch_n)+'? (y/n):')

def other_tasks(path_out,patch_n):
    print('Getting the boundaries of the image. . .')
    raster=rasterio.open(path_out+'/predicted_tiff/patch'+str(patch_n)+'/merged_prediction.tiff')
    df=raster.bounds
    sh_box=box(df.left, df.bottom, df.right, df.top)
    shape_box=gpd.GeoDataFrame({'geometry': sh_box, 'col':[np.nan]})
    shape_box.crs = {'init' :'epsg:4326'}
    shape_box.to_file(path_out+'/shape_box'+str(patch_n))

    if GUF_data:
        print('Clipping the GUF data . . .')
        with fiona.open(path_out+"/shape_box"+str(patch_n)+"/shape_box"+str(patch_n)+".shp", "r") as shapefile:
            features = [feature["geometry"] for feature in shapefile]

        for GUF_data_dir in list_of_GUF:
            try:
                with rasterio.open(GUF_data_dir) as src:
                        out_image, out_transform = rasterio.mask.mask(src, features,
                                                                crop=True)
                        out_meta = src.meta.copy()

                print('Overlap is found. Clipping the data..')

                out_meta.update({"driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})

                with rasterio.open(path_out+"/masked_footprint"+str(patch_n)+".tif", "w", **out_meta) as dest:
                            dest.write(out_image)
                
            
            except:
                pass

        mask = None
        with rasterio.Env():
            with rasterio.open(path_out+'/masked_footprint'+str(patch_n)+'.tif') as src:
                image = src.read(1) # first band
                results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) 
                in enumerate(
                    shapes(image, mask=mask, transform=src.transform)))
        geoms_foot = list(results)      
        urban_foot=gpd.GeoDataFrame.from_features(geoms_foot)
        
        box_domain=gpd.read_file(path_out+'/shape_box'+str(patch_n)+'/shape_box'+str(patch_n)+'.shp')
        urban_foot=clip_shp(urban_foot,box_domain)
        urban_foot=urban_foot.rename(columns={'raster_val':'GUF'})
        urban_foot=urban_foot[urban_foot.GUF!=128]

        urban_foot_temp=urban_foot.buffer(0)
        urban_foot_temp=gpd.GeoDataFrame(urban_foot_temp)
        urban_foot_temp['GUF']=urban_foot['GUF']
        urban_foot_temp=urban_foot_temp.rename(columns={0:'geometry'})
        urban_foot_temp.crs={'init' :'epsg:4326'}
        urban_foot_temp.to_file(path_out+'/urban_foot_shape'+str(patch_n))


    mask = None
    print('Converting the predicted tiff file to shapefile . . .')
    with rasterio.Env():
        with rasterio.open(path_out+'/predicted_tiff/patch'+str(patch_n)+'/merged_prediction.tiff') as src:
            image = src.read(1) # first band
            results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                shapes(image, mask=mask, transform=src.transform)))
    geoms_predict = list(results)      
    predicted=gpd.GeoDataFrame.from_features(geoms_predict)

    predicted=predicted.rename(columns={'raster_val':'predicted'})
    predicted_temp=predicted.buffer(0)
    predicted_temp=gpd.GeoDataFrame(predicted_temp)
    predicted_temp['predicted']=predicted['predicted']
    predicted_temp=predicted_temp.rename(columns={0:'geometry'})
    predicted_temp.crs={'init' :'epsg:4326'}
    predicted_temp.to_file(path_out+'/predicted_shape'+str(patch_n))


    if GUF_data:
        print('Merging the GUF data with the predicted . . .')
        time.sleep(3)
        v1_dir=path_out+'/predicted_shape'+str(patch_n)+'/predicted_shape'+str(patch_n)+'.shp'
        v2_dir=path_out+'/urban_foot_shape'+str(patch_n)+'/urban_foot_shape'+str(patch_n)+'.shp'
        out_dir=path_out+'/predict_GUF'+str(patch_n)
        grass_union(v1_dir,v2_dir,out_dir,patch_n)
        
        predict_GUF=gpd.read_file(path_out+'/predict_GUF'+str(patch_n)+'/predict_GUF'+str(patch_n)+'.shp')
        predict_GUF=predict_GUF[~np.isnan(predict_GUF.a_predicte)]
        predict_GUF['LC']=predict_GUF.a_predicte
        a=predict_GUF[(predict_GUF.b_GUF==255) & (predict_GUF.a_predicte!=1)& (predict_GUF.a_predicte!=2)]
        predict_GUF.loc[a.index,'LC']=0
        b=predict_GUF[(~np.isnan(predict_GUF.b_GUF))&(predict_GUF.b_GUF!=255) & (predict_GUF.a_predicte==0)]
        predict_GUF.loc[b.index,'LC']=3
        predict_GUF=predict_GUF.drop(['a_cat','b_GUF','b_cat','a_predicte','cat'],axis=1)
        predict_GUF_temp=predict_GUF.buffer(0)
        predict_GUF_temp=gpd.GeoDataFrame(predict_GUF_temp)
        predict_GUF_temp['LC']=predict_GUF['LC']
        predict_GUF_temp=predict_GUF_temp.rename(columns={0:'geometry'})
        predict_GUF_temp.crs={'init' :'epsg:4326'}
        predict_GUF_temp.to_file(path_out+'/predict_GUF_mod'+str(patch_n))


    lon_deg_min, lat_deg_min=(lon_left_top,lat_right_bot)
    lon_deg_max, lat_deg_max=(lon_right_bot,lat_left_top)
    lat_rad_min=lat_deg_min*np.pi/180
    lat_rad_max=lat_deg_max*np.pi/180

    zoom=15
    n = 2 **zoom

    xtile_min = n * ((lon_deg_min + 180) / 360)
    ytile_min = n * (1 - (np.log(np.tan(lat_rad_min) + (1/np.cos(lat_rad_min))) / np.pi)) / 2

    xtile_max = n * ((lon_deg_max + 180) / 360)
    ytile_max = n * (1 - (np.log(np.tan(lat_rad_max) + (1/np.cos(lat_rad_max))) / np.pi)) / 2

    if ytile_max < ytile_min:
        temp=ytile_min
        ytile_min=ytile_max
        ytile_max=temp

    ytile_min=int(np.floor(ytile_min))
    ytile_max=int(np.ceil(ytile_max))

    xtile_min=int(np.floor(xtile_min))
    xtile_max=int(np.ceil(xtile_max))
    xtile_min=xtile_min-3

    if Building_data != 'no':
        if Building_data =='OSM':
            print('Downloading the OSM data . . . ')
            to_wait=60
            path_OSM=path_out+'/OSM'+str(patch_n)
            if not os.path.isdir(path_OSM):
                os.makedirs(path_OSM)

            for i in np.arange(xtile_min,xtile_max+1):
                print('Xtile '+str(i-xtile_min+1)+' out of '+str(xtile_max-xtile_min+1))
                for j in np.arange(ytile_min,ytile_max+1):

                    if not Path(path_OSM+'/buildings'+str(i)+'-'+str(j)+'.geojson').exists():
                        url = 'https://a.data.osmbuildings.org/0.2/anonymous/tile/{}/{}/{}.json'.format(zoom,i,j)
                        try:

                            r = requests.get(url,timeout=10)
                            while r.status_code==429:
                                print('Too many requests. Waiting for '+str(to_wait)+' seconds...')
                                time.sleep(to_wait)
                                print('Trying again')
                                r = requests.get(url,timeout=10)

                            with open(path_OSM+'/buildings'+str(i)+'-'+str(j)+'.geojson', 'wb') as f:  
                                f.write(r.content)
                        except:
                            print('Passing this domain')

            print('All data are collected. DONE')

            print('Attaching the OSM data together . . .')
            counter=0
            for i in range(xtile_min,xtile_max+1):
                for j in range(ytile_min,ytile_max+1):

                    try:
                        b=gpd.read_file(path_OSM+'/buildings'+str(i)+'-'+str(j)+'.geojson')
                        b=b.buffer(0)

                        if counter==0:
                            a=b

                        if counter!=0:
                            a=a.union(b)
                        counter=1
                    except:
                        pass

            path_OSM_sh=path_OSM+'_sh'
            if not os.path.isdir(path_OSM_sh):
                os.makedirs(path_OSM_sh)

            a.to_file(path_OSM_sh)


    box_domain=gpd.read_file(path_out+'/shape_box'+str(patch_n)+'/shape_box'+str(patch_n)+'.shp')

    if Building_data != 'no':
        if Building_data =='OSM':
            buildings=gpd.read_file(path_out+'/OSM'+str(patch_n)+'_sh/OSM'+str(patch_n)+'_sh.shp')
        elif Building_data=='MICROSOFT':
            print('Reading the Microsoft building data . . .')
            buildings=gpd.read_file(building_dir)
        print('Clipping the building data to the selected domain')        
        buildings_clipped=clip_shp(buildings,box_domain)
        buildings_clipped=buildings_clipped.buffer(0)
        buildings_clipped=gpd.GeoDataFrame(buildings_clipped)
        buildings_clipped['build']=1
        buildings_clipped=buildings_clipped.rename(columns={0:'geometry'})
        buildings_clipped.crs={'init' :'epsg:4326'}
        print('Writing clipped data into '+path_out+'/'+Building_data+'_sh_clipped'+str(patch_n)+'/')
        buildings_clipped.to_file(path_out+'/'+Building_data+'_sh_clipped'+str(patch_n)+'/')

    if Road_data != 'no':
        print('Downloading road data from OSM . . .')
        G = ox.graph_from_bbox(lat_left_top, lat_right_bot, 
                                lon_right_bot, lon_left_top, 
                                network_type='all_private')
        G_projected = ox.project_graph(G)
        gdf=ox.save_load.graph_to_gdfs(G_projected)[1]
        road_kind=[]
        for rd in gdf.highway:
            if rd[0]=='[':
                rd=ast.literal_eval(rd)
            if type(rd)!=list:
                rd=[rd]
            road_kind.append(rd[0])
                
        gdf['cat']=road_kind 

        with open('./road_width.json') as setting_file:
            subset = json.load(setting_file)

        buffered=gpd.GeoDataFrame(columns=['geometry','cat'])
        for kind in subset.keys():
            temp0=gdf[gdf.cat==kind]
            temp=temp0.buffer(subset[kind])
            temp=gpd.GeoDataFrame(temp,columns=['geometry'])
            temp['cat']=temp0.cat
            buffered=buffered.append(temp)

        buffered.crs=gdf.crs
        buffered.to_file(path_out+'/roads_all_'+str(patch_n))

        buffered.cat='5'
        print('Dissolving roads . . .')
        buffered=buffered.dissolve(by='cat')
        buffered=buffered.to_crs(epsg=4326)
        buffered.to_file(path_out+'/roads_'+str(patch_n))

    if GUF_data:
        if Building_data != 'no' and Road_data == 'no':
            print('Merging the predicted-GUF to Building data . . .')
            time.sleep(3)
            v1_dir=path_out+'/predict_GUF_mod'+str(patch_n)+'/predict_GUF_mod'+str(patch_n)+'.shp'
            v2_dir=path_out+'/'+Building_data+'_sh_clipped'+str(patch_n)+'/'+Building_data+'_sh_clipped'+str(patch_n)+'.shp'
            out_dir=path_out+'/predict_GUF_'+Building_data+str(patch_n)
            grass_union(v1_dir,v2_dir,out_dir,patch_n)

            predict_GUF_bld=gpd.read_file(path_out+'/predict_GUF_'+Building_data+str(patch_n)+'/predict_GUF_'+Building_data+str(patch_n)+'.shp')
            predict_GUF_bld['LC']=predict_GUF_bld.a_LC
            a=predict_GUF_bld[predict_GUF_bld.b_build==1]
            predict_GUF_bld.loc[a.index,'LC']=4
            predict_GUF_bld=predict_GUF_bld.drop(['a_cat','b_cat','a_LC','b_build'],axis=1)
            predict_GUF_bld_temp=predict_GUF_bld.buffer(0)
            predict_GUF_bld_temp=gpd.GeoDataFrame(predict_GUF_bld_temp)
            predict_GUF_bld_temp['LC']=predict_GUF_bld['LC']
            predict_GUF_bld_temp=predict_GUF_bld_temp.rename(columns={0:'geometry'})
            predict_GUF_bld_temp.crs={'init' :'epsg:4326'}
            print('dissolving the final result . . . .')
            predict_GUF_bld_temp=predict_GUF_bld_temp.dissolve('LC')
            predict_GUF_bld_temp=predict_GUF_bld_temp.reset_index()
            predict_GUF_bld_temp.to_file(path_out+'/predict_GUF_'+Building_data+'_mod'+str(patch_n))

        elif Road_data != 'no' and Building_data == 'no':
            print('Merging the predicted-GUF to Road data . . .')
            time.sleep(3)
            v1_dir=path_out+'/predict_GUF_mod'+str(patch_n)+'/predict_GUF_mod'+str(patch_n)+'.shp'
            v2_dir=path_out+'/'+'roads_'+str(patch_n)+'/'+'roads_'+str(patch_n)+'.shp'
            out_dir=path_out+'/predict_GUF_roads_'+str(patch_n)
            grass_union(v1_dir,v2_dir,out_dir,patch_n)

            predict_GUF_rd=gpd.read_file(path_out+'/predict_GUF_roads_'+str(patch_n)+'/predict_GUF_roads_'+str(patch_n)+'.shp')
            predict_GUF_rd['LC']=predict_GUF_rd.a_LC

            a=predict_GUF_rd[predict_GUF_rd.b_cat==1]
            predict_GUF_rd.loc[a.index,'LC']=5
            b=predict_GUF_rd[predict_GUF_rd.LC==0]
            predict_GUF_rd.loc[b.index,'LC']=4
            c=predict_GUF_rd[predict_GUF_rd.LC==5]
            predict_GUF_rd.loc[c.index,'LC']=0

            predict_GUF_rd=predict_GUF_rd.drop(['cat','a_cat','b_cat','a_LC','b_FID'],axis=1)
            predict_GUF_rd_temp=predict_GUF_rd.buffer(0)
            predict_GUF_rd_temp=gpd.GeoDataFrame(predict_GUF_rd_temp)
            predict_GUF_rd_temp['LC']=predict_GUF_rd['LC']
            predict_GUF_rd_temp=predict_GUF_rd_temp.rename(columns={0:'geometry'})
            predict_GUF_rd_temp.crs={'init' :'epsg:4326'}
            print('dissolving the final result . . . .')
            predict_GUF_rd_temp=predict_GUF_rd_temp.dissolve('LC')
            predict_GUF_rd_temp=predict_GUF_rd_temp.reset_index()
            predict_GUF_rd_temp.to_file(path_out+'/predict_GUF_roads'+'_mod'+str(patch_n))

        elif Road_data != 'no' and Building_data != 'no':
            print('Both Building and Road: Merging the predicted-GUF to Road data . . .')
            time.sleep(3)
            v1_dir=path_out+'/predict_GUF_mod'+str(patch_n)+'/predict_GUF_mod'+str(patch_n)+'.shp'
            v2_dir=path_out+'/'+'roads_'+str(patch_n)+'/'+'roads_'+str(patch_n)+'.shp'
            out_dir=path_out+'/predict_GUF_roads_'+str(patch_n)
            grass_union(v1_dir,v2_dir,out_dir,patch_n)

            predict_GUF_rd=gpd.read_file(path_out+'/predict_GUF_roads_'+str(patch_n)+'/predict_GUF_roads_'+str(patch_n)+'.shp')
            predict_GUF_rd['LC']=predict_GUF_rd.a_LC

            a=predict_GUF_rd[predict_GUF_rd.b_cat==1]
            predict_GUF_rd.loc[a.index,'LC']=5
            b=predict_GUF_rd[predict_GUF_rd.LC==0]
            predict_GUF_rd.loc[b.index,'LC']=4
            c=predict_GUF_rd[predict_GUF_rd.LC==5]
            predict_GUF_rd.loc[c.index,'LC']=0

            predict_GUF_rd=predict_GUF_rd.drop(['cat','a_cat','b_cat','a_LC','b_FID'],axis=1)
            predict_GUF_rd_temp=predict_GUF_rd.buffer(0)
            predict_GUF_rd_temp=gpd.GeoDataFrame(predict_GUF_rd_temp)
            predict_GUF_rd_temp['LC']=predict_GUF_rd['LC']
            predict_GUF_rd_temp=predict_GUF_rd_temp.rename(columns={0:'geometry'})
            predict_GUF_rd_temp.crs={'init' :'epsg:4326'}
            print('dissolving the result . . . .')
            predict_GUF_rd_temp=predict_GUF_rd_temp.dissolve('LC')
            predict_GUF_rd_temp=predict_GUF_rd_temp.reset_index()
            predict_GUF_rd_temp.to_file(path_out+'/predict_GUF_roads'+'_mod'+str(patch_n))

            print('Merging the predicted-GUF-roads to Building data . . .')
            time.sleep(3)
            v1_dir=path_out+'/predict_GUF_roads_mod'+str(patch_n)+'/predict_GUF_roads_mod'+str(patch_n)+'.shp'
            v2_dir=path_out+'/'+Building_data+'_sh_clipped'+str(patch_n)+'/'+Building_data+'_sh_clipped'+str(patch_n)+'.shp'
            out_dir=path_out+'/predict_GUF_roads_'+Building_data+str(patch_n)
            grass_union(v1_dir,v2_dir,out_dir,patch_n)
            
            predict_GUF_rd_bd=gpd.read_file(path_out+'/predict_GUF_roads_'+Building_data+str(patch_n)+'/predict_GUF_roads_'+Building_data+str(patch_n)+'.shp')
            predict_GUF_rd_bd['LC']=predict_GUF_rd_bd.a_LC

            a=predict_GUF_rd_bd[predict_GUF_rd_bd.a_LC==4]
            predict_GUF_rd_bd.loc[a.index,'LC']=5
            b=predict_GUF_rd_bd[predict_GUF_rd_bd.b_build==1]
            predict_GUF_rd_bd.loc[b.index,'LC']=4

            predict_GUF_rd_bd=predict_GUF_rd_bd.drop(['cat','a_cat','b_cat','a_LC','b_build'],axis=1)
            predict_GUF_rd_bd_temp=predict_GUF_rd_bd.buffer(0)
            predict_GUF_rd_bd_temp=gpd.GeoDataFrame(predict_GUF_rd_bd_temp)
            predict_GUF_rd_bd_temp['LC']=predict_GUF_rd_bd['LC']
            predict_GUF_rd_bd_temp=predict_GUF_rd_bd_temp.rename(columns={0:'geometry'})
            predict_GUF_rd_bd_temp.crs={'init' :'epsg:4326'}
            print('dissolving the final result . . . .')
            predict_GUF_rd_bd_temp=predict_GUF_rd_bd_temp.dissolve('LC')
            predict_GUF_rd_bd_temp=predict_GUF_rd_bd_temp.reset_index()
            predict_GUF_rd_bd_temp.to_file(path_out+'/predict_GUF_roads_'+Building_data+'_mod'+str(patch_n))

    else:
        if Building_data != 'no' and Road_data == 'no':
            print('Merging the predicted to Building data . . .')
            time.sleep(3)
            v1_dir=path_out+'/predicted_shape'+str(patch_n)+'/predicted_shape'+str(patch_n)+'.shp'
            v2_dir=path_out+'/'+Building_data+'_sh_clipped'+str(patch_n)+'/'+Building_data+'_sh_clipped'+str(patch_n)+'.shp'
            out_dir=path_out+'/predict_'+Building_data+str(patch_n)
            grass_union(v1_dir,v2_dir,out_dir,patch_n)

            predict_bld=gpd.read_file(path_out+'/predict_'+Building_data+str(patch_n)+'/predict_'+Building_data+str(patch_n)+'.shp')
            predict_bld['LC']=predict_bld.a_predicte
            a=predict_bld[predict_bld.b_build==1]
            predict_bld.loc[a.index,'LC']=3
            predict_bld=predict_bld.drop(['a_cat','b_cat','a_predicte','b_build'],axis=1)
            predict_bld_temp=predict_bld.buffer(0)
            predict_bld_temp=gpd.GeoDataFrame(predict_bld_temp)
            predict_bld_temp['LC']=predict_bld['LC']
            predict_bld_temp=predict_bld_temp.rename(columns={0:'geometry'})
            predict_bld_temp.crs={'init' :'epsg:4326'}
            print('dissolving the final result . . . .')
            predict_bld_temp=predict_bld_temp.dissolve('LC')
            predict_bld_temp=predict_bld_temp.reset_index()
            predict_bld_temp.to_file(path_out+'/predict_'+Building_data+'_mod'+str(patch_n))


path_out = cname
all_lats=np.linspace(lat_right_bot_t,lat_left_top_t,num=ny+1)
all_lons=np.linspace(lon_left_top_t,lon_right_bot_t,num=nx+1)
patch_n=0
for i in range(0,len(all_lons)-1):
    lon_left_top,lon_right_bot=[all_lons[i],all_lons[i+1]]
    for j in range(0,len(all_lats)-1):

        skip_patch=input('Skip patch '+str(patch_n)+'? (y/n):')
        while (skip_patch not in ['y','n']):
            skip_patch=input('Skip patch '+str(patch_n)+'? (y/n):')

        if skip_patch!='y':
            lat_right_bot,lat_left_top=[all_lats[j],all_lats[j+1]]

            coords_top=[lat_left_top, lon_left_top]
            coords_bot=[lat_right_bot, lon_right_bot]

            download_data(path_out,coords_top,coords_bot,patch_n)
            save_images(path_out,patch_n)
            predict_image(path_out,patch_n)
            other_tasks(path_out,patch_n)
        patch_n=patch_n+1
#%%
