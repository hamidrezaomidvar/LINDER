
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
from shapely.geometry import box
import geopandas as gpd
import numpy as np
from eolearn.core import  EOPatch
import json
import cv2
import tensorflow as tf
from tqdm import tqdm

def reproject_labels(path_label,before,after):
    dst_crs = 'EPSG:4326'
    with rasterio.open(path_label / before) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(path_label / after, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return path_label / after



#def crop_to_box(path_label_reprojected,coords_bot,coords_top):
def crop_to_box(path_label_reprojected,bbox):
    
    # top,left=coords_top
    # bottom,right=coords_bot
    # sh_box = box(left,bottom, right, top)
    shape_box = gpd.GeoDataFrame({"geometry": bbox, "col": [np.nan]})
    shape_box.crs = {"init": "epsg:4326"}
    shapes = shape_box["geometry"]

    LC_data_path=path_label_reprojected

    with rasterio.open(LC_data_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    LC_data_path_masked=path_label_reprojected.parent/'tq-wgs84-masked.tif'
    with rasterio.open(LC_data_path_masked, "w", **out_meta) as dest:
        dest.write(out_image)
    return LC_data_path_masked


def get_labels_from_json(LC_data_path_masked):
    LC_data = rasterio.open(LC_data_path_masked)
    LC_data = LC_data.read()[:, :, 1:]
    
    print('getting labels from Json . . .')
    with open('map_LC.json') as file:
        classes = json.load(file)

    for i in tqdm(range(LC_data.shape[1])):
        for j in range(LC_data.shape[2]):
            x = LC_data[0:3, i, j]
            if ~(np.array(x)==np.array([0,0,0])).all():
                LC_data[3, i, j] = classes[f'{x[0]}'][f'{x[1]}'][f'{x[2]}']
            else:
                LC_data[3, i, j]=5

    label_data = LC_data[3, :, :]
    return label_data


def resize_label_data_to_sentinel(path_EOPatch, image, label_data):
    data = EOPatch.load(path_EOPatch)
    x2 = data.data['NDVI'][image].shape[0]
    x1 = data.data['NDVI'][image].shape[1]
    label_data_resized = cv2.resize(label_data, (x1, x2))
    return label_data_resized



def stack_all_data(path_EOPatch,label_data_resized,image):
    data = EOPatch.load(path_EOPatch)
    data_all=tf.convert_to_tensor(np.stack([data.data['NDVI'][image],
                                            data.data['NDWI'][image],
                                            data.data['NORM'][image],
                                            data.data["BANDS"][image, :, :, :][..., [2]],
                                            data.data["BANDS"][image, :, :, :][..., [1]],
                                            data.data["BANDS"][image, :, :, :][..., [0]],
                                           ]
                                           ,axis=2)[:,:,:,0])
    masks=tf.convert_to_tensor(label_data_resized[...,np.newaxis])
    stacked_all=tf.stack([data_all[:,:,0],
                          data_all[:,:,1],
                          data_all[:,:,2],
                          data_all[:,:,3],
                          data_all[:,:,4],
                          data_all[:,:,5],
                          tf.cast(masks[:,:,0],tf.float32),
                         ],
                         axis=2)
    return data_all,stacked_all



def get_sub_stackes(stacked_all):
    sub_stack=tf.image.random_crop(stacked_all,size = [128,128, 7])
    return sub_stack


def prepare_data_for_model(stacked_all,n_data):
    images=[]
    labels=[]
    rgbs=[]
    for i in range(n_data):
        cropped=get_sub_stackes(stacked_all)

        features=cropped[:,:,0:3]
        #features=cropped[:,:,3:6]
        rgb=cropped[:,:,3:6]
        label=tf.cast(cropped[:,:,6:7],tf.int8)

        images.append(features)
        labels.append(label)
        rgbs.append(rgb)
    
    
    dataset=tf.stack(images)
    labelset=tf.stack(labels)
    rgbset=tf.stack(rgbs)
    
    return dataset,labelset,rgbset