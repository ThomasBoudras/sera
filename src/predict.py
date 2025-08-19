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
from src.global_utils import get_window
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box

log = utils.get_logger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def predict(config: DictConfig) -> None:
   
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    OmegaConf.set_struct(config, True)


    if "seed" in config:
        seed_everything(config.seed, workers=True)

    save_dir = Path(config.save_dir).resolve()
    save_dir_tmp = save_dir / "tmp"
    save_dir_data = save_dir / "data"
    
    if save_dir.exists() :
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    target_saved = config.module.instance.save_target
    two_inputs = ("change" in config.datamodule.dataset.predict_dataset._target_.lower())

    if save_dir_data.exists() :
        shutil.rmtree(save_dir_data)
    save_dir_data.mkdir(parents=True)

    (save_dir_data / "preds").mkdir()

    if two_inputs :
        (save_dir_data / "inputs_t1").mkdir()
        (save_dir_data / "inputs_t2").mkdir()
    else :
        (save_dir_data / "inputs").mkdir()

    if target_saved:
        (save_dir_data / "targets").mkdir()
    
    
    if save_dir_tmp.exists() :
        shutil.rmtree(save_dir_tmp)
    save_dir_tmp.mkdir(parents=True)
    (save_dir_tmp / "preds").mkdir()
    (save_dir_tmp / "inputs").mkdir()
    if target_saved :
        (save_dir_tmp / "targets").mkdir()
    (save_dir_tmp / "bounds").mkdir()

    if config.via_bounds is not None and config.via_geojson is None:
        row = {name : data for name, data in config.via_bounds.items() if name!="bounds"}
        bounds = config.via_bounds["bounds"]
        bounds = bounds["bottom"], bounds["left"], bounds["top"], bounds["right"]
        row["geometry"] = box(*bounds)
        aoi_gdf = gpd.GeoDataFrame([row], crs="EPSG:2154")

    elif config.via_bounds is None and config.via_geojson is not None :
        aoi_gdf = gpd.read_file(config.via_geojson["path"])
        aoi_gdf = aoi_gdf[aoi_gdf['split'] == config.via_geojson["split"]].reset_index(drop=True) 
        # aoi_gdf = aoi_gdf[:100] # for debug
    else:
        Exception("via_bounds and via_geojson : one of them must be provided and the other must be null")

    # Group geometries that are adjacent or touching each other
    unioned = unary_union(aoi_gdf['geometry'])
    if isinstance(unioned, MultiPolygon):
        polygons = list(unioned.geoms)
    elif isinstance(unioned, Polygon):
        polygons = [unioned]
    else:
        polygons = []

    grouped_aoi = gpd.GeoDataFrame({'geometry': polygons}, crs=aoi_gdf.crs)
    gdf_path = save_dir / "predict_gdf.geojson"
    aoi_gdf.to_file(
            gdf_path,
            driver="GeoJSON",
        )
 
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.instance._target_}>")
    config.datamodule.dataset.gdf_path = gdf_path # we take the new gdf_path
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
    if config.get("ckpt_path") is not None or config.get("ckpt_path") != "last":
        ckpt_path = config.get("ckpt_path")
        if config.load_just_weights :
            log.info(f"Start of training from checkpoint {ckpt_path} using only the weights !")
            checkpoint = torch.load(ckpt_path)

            if "state_dict" in checkpoint:
                missing_keys, unexpected_keys = module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                missing_keys, unexpected_keys = module.load_state_dict(checkpoint, strict=False)
            
                log.warning(f"Missing keys in checkpoint: {missing_keys}")
                log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            ckpt_path = None
        
        else :
            log.info(f"Start of training from checkpoint {ckpt_path} !")
    
    elif ckpt_path == "last" :
        log.info(f"Starting training from last checkpoint {ckpt_path} !")
    
    else :
        log.info("Starting training from scratch!")
        ckpt_path = None

    # Predict
    trainer.predict(module, ckpt_path=ckpt_path, datamodule=datamodule)

    # Save predictions properly
    nb_predicted_files = len([
        file
        for file in (save_dir_tmp / "preds").iterdir()
        if file.suffix == ".npy"
    ])

    for i_file in tqdm(range(nb_predicted_files), desc="Saving preds for each batch"):
        pred_file = save_dir_tmp / "preds" / f"batch_{i_file}.npy"
        inputs_file = save_dir_tmp / "inputs" / f"batch_{i_file}.npy"
        bounds_file = save_dir_tmp / "bounds" / f"batch_{i_file}.npy"
        target_file = save_dir_tmp / "targets" / f"batch_{i_file}.npy"
        
        # Each pred_file stores one batch of preds 
        batch_preds = np.load(pred_file)
        batch_inputs = np.load(inputs_file)
        batch_bounds = np.load(bounds_file)
        if target_saved :
            batch_target = np.load(target_file)

        for idx, preds in enumerate(batch_preds):
            inputs = batch_inputs[idx]
            bounds = batch_bounds[idx]
            if target_saved :
                targets = batch_target[idx]

            #preds save
            save_image_in_tif(
                preds, 
                save_dir=save_dir_tmp, 
                bounds=bounds, 
                image_type="preds",
            )

            #inputs save
            if two_inputs :
                separation_inputs = int(inputs.shape[0]/2)
                inputs_t1 = inputs[:separation_inputs, ...]
                inputs_t2 = inputs[separation_inputs:, ...]
                
                if len(inputs.shape) == 4 :
                    inputs_t1 = np.median(inputs_t1, axis=0)
                    inputs_t2 = np.median(inputs_t2, axis=0)

                save_image_in_tif(
                    inputs_t1, 
                    save_dir=save_dir_tmp, 
                    bounds=bounds, 
                    image_type="inputs_t1",
                )
                save_image_in_tif(
                    inputs_t2, 
                    save_dir=save_dir_tmp, 
                    bounds=bounds, 
                    image_type="inputs_t2",
                )
            else :
                if len(inputs.shape) == 4 :
                    inputs = np.median(inputs, axis=0)
                save_image_in_tif(
                    inputs, 
                    save_dir=save_dir_tmp, 
                    bounds=bounds, 
                    image_type="inputs",
                )

            #targets save
            if target_saved :
                save_image_in_tif(
                    targets, 
                    save_dir=save_dir_tmp, 
                    bounds=bounds, 
                    image_type="targets",
                )
                    
    get_grouped_tif(
        grouped_aoi=grouped_aoi, 
        save_tmp_dir=save_dir_tmp, 
        final_save_dir=save_dir_data, 
        model_name=config.run_name, 
        image_type="preds",
        merge_method=config.merge_method_outputs
    )
    
    if two_inputs :
        get_grouped_tif(
            grouped_aoi=grouped_aoi, 
            save_tmp_dir=save_dir_tmp, 
            final_save_dir=save_dir_data, 
            model_name=config.run_name, 
            image_type="inputs_t1",
            merge_method=config.merge_method_inputs
        )
        get_grouped_tif(
            grouped_aoi=grouped_aoi, 
            save_tmp_dir=save_dir_tmp, 
            final_save_dir=save_dir_data, 
            model_name=config.run_name, 
            image_type="inputs_t2",
            merge_method=config.merge_method_inputs
        )
    else :
        get_grouped_tif(
            grouped_aoi=grouped_aoi, 
            save_tmp_dir=save_dir_tmp, 
            final_save_dir=save_dir_data, 
            model_name=config.run_name, 
            image_type="inputs",
            merge_method=config.merge_method_inputs
        )

    if target_saved :
        get_grouped_tif(
            grouped_aoi=grouped_aoi, 
            save_tmp_dir=save_dir_tmp, 
            final_save_dir=save_dir_data, 
            model_name=config.run_name, 
            image_type="targets",
            merge_method=config.merge_method_outputs
        )

    # remove old save_dir
    shutil.rmtree(save_dir_tmp)

def save_image_in_tif(image, save_dir, bounds, image_type):    
    count, height, width = image.shape
    transform = from_bounds(*bounds, width, height)
    
    save_dir_geometry = save_dir / f"patch_tifs" / image_type
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


def merge_tif(list_tif, bounds, output_path, merge_method):
    """
    Concatenate all tif files from a folder into a single output GeoTIFF using GDAL.

    Args:
        folder_path (Path): Path to the folder containing tif files.
        output_path (str or Path): Path to the output GeoTIFF.
    """

    bounds_poly = box(*bounds)

    # Filtrer les tifs qui intersectent les bounds
    list_tif_intersect = []
    for tif in list_tif:
        tif_bounds = tif.stem.split("_")
        tif_bounds = [int(tif_bounds[0]), int(tif_bounds[1]), int(tif_bounds[2]), int(tif_bounds[3])]
        tif_poly = box(*tif_bounds)
        if tif_poly.intersects(bounds_poly):
            list_tif_intersect.append(tif)

    # Read all images using get_window and stack them for mean calculation
    list_images = []
    for tif_file in list_tif_intersect:
        image, _ = get_window(
            tif_file,
            bounds,
            resolution=None,
            resampling_method=None,
            open_even_oob=True,
        )
        list_images.append(image.astype(np.float32))
    
    # Compute the mean across all images, ignoring NaNs
    merge_method = getattr(np, merge_method)
    tif_merged = merge_method(list_images, axis=0)
    count, height, width = tif_merged.shape

    transform = from_bounds(*bounds, width, height)
    # Write the mean image to the output path using rasteri
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        count=count,
        height=height,
        width=width,
        dtype=tif_merged.dtype,
        crs="EPSG:2154",
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(tif_merged)


def get_grouped_tif(grouped_aoi, save_tmp_dir, final_save_dir, model_name, image_type, merge_method) :
    patches_path = save_tmp_dir / f"patch_tifs" / image_type
    list_tif = [file for file in patches_path.iterdir() if file.suffix == ".tif"]

    for id, row in tqdm(grouped_aoi.iterrows(), total=len(grouped_aoi), desc=f"Merging {image_type} tifs"):
        bounds = row.geometry.bounds

        final_geo_path = final_save_dir / image_type / f"{model_name}_{image_type}_{id}.tif"
        merge_tif(
            list_tif=list_tif, 
            bounds=bounds,
            output_path=final_geo_path, 
            merge_method=merge_method
        )
        
        
    vrt_path = final_save_dir / image_type / f"{model_name}_{image_type}_full.vrt"
    files_list = [
         str(file) for file in (final_save_dir / image_type).iterdir() if file.suffix == ".tif"
    ]

    gdal.UseExceptions()
    vrt_options = gdal.BuildVRTOptions(
        separate=False,
        srcNodata="nan",
        VRTNodata="nan",
    )  
   
    vrt = gdal.BuildVRT(vrt_path, files_list, options=vrt_options)
    vrt.FlushCache()

if __name__ == "__main__":
    predict()



