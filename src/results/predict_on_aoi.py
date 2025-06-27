
import os
import shutil
from datetime import datetime

import geopandas as gpd
import hydra
import numpy as np
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from osgeo import gdal
from pyproj import Transformer
from pytorch_lightning import Trainer
from shapely.geometry import box
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import glob
from rasterio.transform import from_bounds

from lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import random


# from torchvision import transforms
from tqdm import tqdm



def predict_on_aoi(config: DictConfig) -> None:
   
    from src import train_utils as utils
    log = utils.get_logger(__name__)
    
    if "seed" in config:
        seed_everything(config.seed, workers=True)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    save_dir_tmp = os.path.join(config.save_dir, "tmp")
    save_dir_data = os.path.join(config.save_dir, "data")


    if os.path.exists(config.save_dir):
        shutil.rmtree(config.save_dir)
    os.makedirs(config.save_dir)

    target_saved = config.model.instance.save_target
    two_input = ("change" in config.datamodule.dataset.geometries_path.lower())

    if os.path.exists(save_dir_data):
        shutil.rmtree(save_dir_data)
    os.makedirs(save_dir_data)
    os.makedirs(os.path.join(save_dir_data, "pred"))
    if two_input :
        os.makedirs(os.path.join(save_dir_data, "input_1"))
        os.makedirs(os.path.join(save_dir_data, "input_2"))
    else :
        os.makedirs(os.path.join(save_dir_data, "input"))
    if target_saved:
        os.makedirs(os.path.join(save_dir_data, "target"))
    
    
    if os.path.exists(save_dir_tmp):
        shutil.rmtree(save_dir_tmp)
    os.makedirs(save_dir_tmp)
    os.makedirs(os.path.join(save_dir_tmp, "pred"))
    os.makedirs(os.path.join(save_dir_tmp, "input"))
    if target_saved :
        os.makedirs(os.path.join(save_dir_tmp, "target"))
    os.makedirs(os.path.join(save_dir_tmp, "bounds"))
    os.makedirs(os.path.join(save_dir_tmp, "number_geometry"))



    # save the config next to the data
    OmegaConf.save(config, os.path.join(config.save_dir, "predict_on_aoi_config.yaml"))

    if config.via_bounds is not None and config.via_geojson is None:
        row = {name : data for name, data in config.via_bounds.items() if name!="bounds"}
        bounds = config.via_bounds["bounds"]
        bounds = bounds["bottom"], bounds["left"], bounds["top"], bounds["right"]
        row["geometry"] = box(*bounds)
        aoi_gdf = gpd.GeoDataFrame([row], crs="EPSG:2154")
    elif config.via_bounds is None and config.via_geojson is not None :
        aoi_gdf = gpd.read_file(config.via_geojson["path"])
        aoi_gdf = aoi_gdf[aoi_gdf['split'] == config.via_geojson["split"]].reset_index(drop=True) 
    else:
        Exception("Either via_bouns or via_geojson must be null")

    geometries_path = os.path.join(config.save_dir, "geometries.geojson")
    aoi_gdf.to_file(
            geometries_path,
            driver="GeoJSON",
        )
    
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.instance._target_}>")
    config.datamodule.dataset.geometries_path = geometries_path
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule.instance)

    # Init lightning model
    log.info(f"Instantiating model <{config.model.instance._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model.instance)
    model.predictions_save_dir = save_dir_tmp

    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger)


    # Train the model
    if config.get("ckpt_path"):
        ckpt_path = config.get("ckpt_path")
        if config.load_just_weights :
            log.info(f"Start of training from checkpoint {ckpt_path} using only the weights !")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            ckpt_path = None
        else :
            log.info(f"Start of training from checkpoint {ckpt_path} !")
    else :
        log.info("Starting training from scratch!")
        ckpt_path = None

  
    # Preprocess aoi_gdf to merge geometries that form rectangles
    # !!!WARNING!!! this results in a potentially larger predict area than demanded,
    # as we're taking bounds of adjacent geometries to form new (larger) geometries
    # (geometries may be adjacent but not form a full rectangle)


    # Predict
    trainer.predict(model, ckpt_path=ckpt_path, datamodule=datamodule)


    nb_predicted_files = len([
        os.path.join(save_dir_tmp, "pred", x)
        for x in os.listdir( os.path.join(save_dir_tmp, "pred"))
        if x.endswith(".npy")
    ])


    # XXX The following could be parallelized / included in the prediction step (especially for small patch size),
    # to make it faster
    # XXX gdal can handle a limited number of files to concat (need to increase file limit / it is slow)
    # Create a callback called after predict to reconcat preds -> need to save params for that
    for i_file in tqdm(range(nb_predicted_files), desc="Saving preds for each batch"):
        pred_file = os.path.join(save_dir_tmp, "pred", str(i_file) + ".npy")
        input_file = os.path.join(save_dir_tmp, "input", str(i_file) + ".npy")
        bounds_file = os.path.join(save_dir_tmp, "bounds", str(i_file) + ".npy")
        number_geometry_file = os.path.join(save_dir_tmp, "number_geometry", str(i_file) + ".npy")
        target_file = os.path.join(save_dir_tmp, "target", str(i_file) + ".npy")
        
        # Each pred_file stores one batch of preds 
        batch_pred = np.load(pred_file)
        batch_input = np.load(input_file)
        batch_bounds = np.load(bounds_file)
        batch_number_geometry = np.load(number_geometry_file)
        if target_saved :
            batch_target = np.load(target_file)
        
        for idx, pred in enumerate(batch_pred):
            input = batch_input[idx]
            bounds = batch_bounds[idx]
            nb_geometry = int(batch_number_geometry[idx])
            if target_saved :
                target = batch_target[idx]

            #pred save
            save_image_in_tif(pred, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="pred")

            #input save
            if two_input :
                separation_input = int(input.shape[0]/2)
                input_1 = input[:separation_input, ...]
                input_2 = input[separation_input:, ...]
                
                if len(input.shape) == 4 :
                    input_1 = np.median(input_1, axis=0)
                    input_2 = np.median(input_2, axis=0)

                save_image_in_tif(input_1, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="input_1")
                save_image_in_tif(input_2, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="input_2")
            else :
                if len(input.shape) == 4 :
                    input = np.mean(input, axis=0)
                save_image_in_tif(input, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="input")

            #target save
            if target_saved :
                save_image_in_tif(target, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="target")
                    

    group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="pred")
    
    if two_input :
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="input_1")
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="input_2")
    else :
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="input")

    if target_saved :
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="target")

    # remove old config.save_dir
    shutil.rmtree(save_dir_tmp)


def save_image_in_tif(image, save_dir, bounds, nb_geometry, image_type="pred"):    
    count, height, width = image.shape
    transform = from_bounds(*bounds, width, height)
    
    save_dir_geometry = os.path.join(save_dir, f"geometry", image_type, f"geometry_{str(nb_geometry)}")
    if not os.path.exists(save_dir_geometry): 
        os.makedirs(save_dir_geometry)
    save_image_path = os.path.join(save_dir_geometry, f"{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}" + ".tif")
    with rasterio.open(
        save_image_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=image.dtype,
        crs="EPSG:2154",  # Remplacer par le CRS approprié si nécessaire
        transform=transform,
    ) as dst:
        dst.write(image)


def create_vrt(files_list, output_path, return_vrt=False):
    # Create a VRT
    vrt_options = gdal.BuildVRTOptions(
        separate=False,
        srcNodata="nan",
        VRTNodata="nan",
        )  
   
    vrt = gdal.BuildVRT(output_path, files_list, options=vrt_options)
    vrt.FlushCache()  # Ensure all data is written
    if return_vrt :
        return vrt    
    else :
        del vrt


def concat_tif_from_folder(folder_path, output_path, pattern=".jp2"):
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_CACHEMAX", "10000")

    tif_files = glob.glob(folder_path + "/*" + pattern)
    vrt = create_vrt(tif_files, output_path, return_vrt=True)

    warp_options = gdal.WarpOptions(
        resampleAlg="cubicspline",        
        warpOptions=["CUTLINE_ALL_TOUCHED=TRUE", "COMPRESS=LZW", "TILED=YES"]  
    )
    gdal.Warp(output_path, vrt, format="GTiff", options=warp_options)


def group_tif(save_tmp_dir, final_save_dir, model_name, image_type="pred") :
    path_geometries = os.path.join(save_tmp_dir, f"geometry", image_type)
    list_save_dir_tmp_geometry = sorted([os.path.join(path_geometries, d) for d in os.listdir(path_geometries) if os.path.isdir(os.path.join(path_geometries, d))])
    for i, dir_tmp_geometry in tqdm(enumerate(list_save_dir_tmp_geometry), total=len(list_save_dir_tmp_geometry), desc=f"Grouping {image_type} tifs"):
        concat_tif_from_folder(
            dir_tmp_geometry, os.path.join(final_save_dir, image_type, f"geo_{i}_{model_name}_{image_type}.tif"), pattern=".tif"
        )
    vrt_path = os.path.join(final_save_dir, image_type, f"full_{model_name}_{image_type}.vrt")
    files_list = [
        os.path.join(final_save_dir, image_type, x) for x in os.listdir(os.path.join(final_save_dir, image_type)) if x.endswith(".tif")
    ]
    create_vrt(files_list, vrt_path)