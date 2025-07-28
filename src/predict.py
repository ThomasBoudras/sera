import hydra
import shutil
from pathlib import Path
import geopandas as gpd 
import numpy as np
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from osgeo import gdal
from shapely.geometry import box
import glob
from rasterio.transform import from_bounds
from lightning import seed_everything
from tqdm import tqdm

from src import global_utils as utils

log = utils.get_logger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def predict(config: DictConfig) -> None:
   
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    OmegaConf.set_struct(config, True)


    if "seed" in config:
        seed_everything(config.seed, workers=True)
        torch.use_deterministic_algorithms(True)

    save_dir = Path(config.save_dir).resolve()
    save_dir_tmp = save_dir / "tmp"
    save_dir_data = save_dir / "data"

    if save_dir.exists() :
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    target_saved = config.module.instance.save_target
    two_input = ("change" in config.datamodule.dataset.geometries_path.lower())

    if save_dir_data.exists() :
        shutil.rmtree(save_dir_data)
    save_dir.mkdir(parents=True)

    (save_dir_data / "pred").mkdir()

    if two_input :
        (save_dir_data / "input_t1").mkdir()
        (save_dir_data / "input_t2").mkdir()
    else :
        (save_dir_data / "input").mkdir()

    if target_saved:
        (save_dir_data / "target").mkdir()
    
    
    if save_dir_tmp.exists() :
        shutil.rmtree(save_dir_tmp)
    save_dir_tmp.mkdir(parents=True)
    (save_dir_tmp / "pred").mkdir()
    (save_dir_tmp / "input").mkdir()
    if target_saved :
        (save_dir_tmp / "target").mkdir()
    (save_dir_tmp / "bounds").mkdir()
    (save_dir_tmp / "geometry_id").mkdir()

    # save the config next to the data
    OmegaConf.save(config, (save_dir /"predict_on_aoi_config.yaml"))

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

    geometries_path = save_dir / "geometries.geojson"
    aoi_gdf.to_file(
            geometries_path,
            driver="GeoJSON",
        )
    
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.instance._target_}>")
    config.datamodule.dataset.geometries_path = geometries_path
    datamodule = hydra.utils.instantiate(config.datamodule.instance)

    # Init lightning module
    log.info(f"Instantiating module <{config.module.instance._target_}>")
    module = hydra.utils.instantiate(config.module.instance)
    module.predictions_save_dir = save_dir_tmp

    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Train the model
    if config.get("ckpt_path"):
        ckpt_path = config.get("ckpt_path")
        if config.load_just_weights :
            log.info(f"Start of training from checkpoint {ckpt_path} using only the weights !")
            checkpoint = torch.load(ckpt_path)
            module.load_state_dict(checkpoint['state_dict'], strict=False)
            ckpt_path = None
        else :
            log.info(f"Start of training from checkpoint {ckpt_path} !")
    else :
        log.info("Starting training from scratch!")
        ckpt_path = None

    # Predict
        trainer.predict(module, ckpt_path=ckpt_path, datamodule=datamodule)


    nb_predicted_files = len([
        save_dir_tmp / "pred" / file
        for file in (save_dir_tmp / "pred").iterdir()
        if file.suffix == "npy"
    ])

    for i_file in tqdm(range(nb_predicted_files), desc="Saving preds for each batch"):
        pred_file = save_dir_tmp / "pred" / f"{i_file}.npy"
        input_file = save_dir_tmp / "input" / f"{i_file}.npy"
        bounds_file = save_dir_tmp / "bounds" / f"{i_file}.npy"
        geometry_id_file = save_dir_tmp / "geometry_id" / f"{i_file}.npy"
        target_file = save_dir_tmp / "target" / f"{i_file}.npy"
        
        # Each pred_file stores one batch of preds 
        batch_pred = np.load(pred_file)
        batch_input = np.load(input_file)
        batch_bounds = np.load(bounds_file)
        batch_geometry_id = np.load(geometry_id_file)
        if target_saved :
            batch_target = np.load(target_file)
        
        for idx, pred in enumerate(batch_pred):
            input = batch_input[idx]
            bounds = batch_bounds[idx]
            nb_geometry = int(batch_geometry_id[idx])
            if target_saved :
                target = batch_target[idx]

            #pred save
            save_image_in_tif(pred, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="pred")

            #input save
            if two_input :
                separation_input = int(input.shape[0]/2)
                input_t1 = input[:separation_input, ...]
                input_t2 = input[separation_input:, ...]
                
                if len(input.shape) == 4 :
                    input_t1 = np.median(input_t1, axis=0)
                    input_t2 = np.median(input_t2, axis=0)

                save_image_in_tif(input_t1, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="input_t1")
                save_image_in_tif(input_t2, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="input_t2")
            else :
                if len(input.shape) == 4 :
                    input = np.mean(input, axis=0)
                save_image_in_tif(input, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="input")

            #target save
            if target_saved :
                save_image_in_tif(target, save_dir=save_dir_tmp, bounds=bounds, nb_geometry=nb_geometry, image_type="target")
                    

    group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="pred")
    
    if two_input :
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="input_t1")
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="input_t2")
    else :
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="input")

    if target_saved :
        group_tif(save_tmp_dir=save_dir_tmp, final_save_dir=save_dir_data, model_name=config.run_name, image_type="target")

    # remove old save_dir
    shutil.rmtree(save_dir_tmp)


def save_image_in_tif(image, save_dir, bounds, nb_geometry, image_type="pred"):    
    count, height, width = image.shape
    transform = from_bounds(*bounds, width, height)
    
    save_dir_geometry = save_dir / f"geometry" / image_type / f"geometry_{nb_geometry}"
    if not save_dir_geometry.exists() : 
        save_dir_geometry.mkdir(parents=True)
    save_image_path = save_dir_geometry / f"{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}.tif"
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
    geometries_path = save_tmp_dir / f"geometry" / image_type
    list_save_dir_tmp_geometry = sorted([geometries_path / file for file in geometries_path.iterdir() if (geometries_path / file).is_dir()])
    for i, dir_tmp_geometry in tqdm(enumerate(list_save_dir_tmp_geometry), total=len(list_save_dir_tmp_geometry), desc=f"Grouping {image_type} tifs"):
        geo_path = final_save_dir / image_type / f"geo_{i}_{model_name}_{image_type}.tif"
        concat_tif_from_folder(
            dir_tmp_geometry, 
            geo_path, 
            pattern=".tif"
        )

    vrt_path = final_save_dir / image_type / f"full_{model_name}_{image_type}.vrt"
    files_list = [
        final_save_dir / image_type / file for file in (final_save_dir, image_type).iterdir() if file.endswith(".tif")
    ]
    create_vrt(files_list, vrt_path)