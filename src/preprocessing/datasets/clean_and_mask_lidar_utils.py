import numpy as np
from osgeo import gdal
import shutil
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import from_bounds as from_bounds_window
from rasterio.transform import from_bounds as from_bounds_transform
from tqdm import tqdm
import logging
from pathlib import Path
import shutil
import geopandas as gpd
from rasterio import features
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape
import matplotlib.pyplot as plt
from rasterio.plot import show

from src.global_utils import get_window

def create_grid(bounds, tile_size, resolution, out_of_bounds_avoided):
    """
    Create a grid of square tiles within the given bounds.

    Args:
        bounds (tuple): A tuple of (min_x, min_y, max_x, max_y) in the same CRS as tile_size
        tile_size (float): The edge length of the square tile in meters
        resolution (float): The resolution of the tiles in meters

    Returns:
        list: A list of tuples (min_x, min_y, max_x, max_y) representing the grid tiles
    """
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    # Calculate the number of tiles in each dimension
    x_count = int(np.ceil(width / tile_size))
    y_count = int(np.ceil(height / tile_size))

    # Generate the tiles
    bounds_tiles = []
    for x in range(x_count):
        for y in range(y_count):
            # Calculate the tile's bounds
            tile_min_x = min_x + x * tile_size
            tile_min_y = min_y + y * tile_size 
            if resolution is not None:
                tile_min_x = tile_min_x - ((tile_min_x - min_x) % resolution) # align to resolution
                tile_min_y = tile_min_y - ((tile_min_y - min_y) % resolution) # align to resolution
            
            tile_max_x = tile_min_x + tile_size
            tile_max_y = tile_min_y + tile_size
            if out_of_bounds_avoided:
                tile_max_x = min(tile_max_x, max_x) 
                tile_max_y = min(tile_max_y, max_y) 

            # Create the tile as a a tuple of bounds and add it to the list
            bounds_tile = (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
            bounds_tiles.append(bounds_tile)

    return bounds_tiles


def create_vrt(files_list, output_path):
    # Create a VRT
    dataset = gdal.BuildVRT(output_path, files_list)
    dataset.FlushCache()  # Ensure all data is written
    dataset = None  # Close the dataset
    

def get_lidar_image_and_mask(
    lidar_path: str,
    bounds: tuple,
    replace_zero: bool,
    min_area,
    resolution_target: float,
    no_data_value: any,
):
    """
    Generate a NaN mask from a lidar file for a given area.
    """
    image, profile = get_window(
        image_path=lidar_path,
        bounds=bounds,
        resolution=resolution_target,
        resampling_method="max",
        open_even_oob=True,
    )
    if image is None:
        return None, None, None, None

    image = image.astype(np.float32).squeeze()
    if np.isfinite(image).sum() == 0:
        return None, None, None, None
    
    nodata_value_num = no_data_value if no_data_value != "nan" else np.nan

    image[~np.isfinite(image)] = np.nan
    image[image == nodata_value_num] = np.nan
    image[image < 0] = np.nan
    if "nodata" in profile and not np.isnan(profile["nodata"]):
        image[image == profile["nodata"]] = np.nan

    height, width = image.shape[-2], image.shape[-1]
    image = np.pad(image, pad_width=((1,1), (1,1)), mode='constant', constant_values=np.nan)

    transform_image = profile["transform"]
    mask = ~np.isnan(image)
    if replace_zero : 
        mask = np.logical_and(mask, image > 0)

    tif_meta = profile.copy()
    tif_meta.update({
        "height": height,
        "width": width,
        "dtype": "float32",
        "driver": "GTiff"
    })

    polygons = [
        shape(geom)
        for geom, value in features.shapes(mask.astype(np.int16), transform=transform_image)
        if value == 1
    ]

    
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=2154)
    gdf = gdf[gdf.geometry.area > min_area]
    if not gdf.empty:
        final_mask = geometry_mask(
            geometries=gdf.geometry,
            transform=transform_image,
            invert=True,
            out_shape=image.shape
        )
    else:
        final_mask = np.zeros(image.shape, dtype=bool)

    return image, final_mask, transform_image, tif_meta




def get_masked_lidar_tiles(
    bounds,
    vrt_path_t1,
    vrt_path_t2,
    output_lidar_path_t1,
    output_lidar_path_t2,
    min_area,
    replace_zero_t1,
    replace_zero_t2,
    lidar_unit_t1,
    lidar_unit_t2,
    resolution_target,
    no_data_t1,
    no_data_t2,
):
    """Process a single tile and return the results"""
    image_t1, valid_mask_t1, transform_t1, tif_meta_t1 = get_lidar_image_and_mask(
        lidar_path=vrt_path_t1,
        bounds=bounds,
        replace_zero=replace_zero_t1,
        min_area=min_area,
        resolution_target=resolution_target,
        no_data_value=no_data_t1,
    )
    if image_t1 is None:
        logging.info(f"image_t1 is None : {bounds}")
        return

    image_t2, valid_mask_t2, transform_t2, tif_meta_t2 = get_lidar_image_and_mask(
        lidar_path=vrt_path_t2,
        bounds=bounds,
        replace_zero=replace_zero_t2,
        min_area=min_area,
        resolution_target=resolution_target,
        no_data_value=no_data_t2,
    )
    if image_t2 is None:
        logging.info(f"image_t2 is None : {bounds}")
        return

    if transform_t1 != transform_t2:
        logging.info(f"### transform 1 :{transform_t1},\n transform 2 {transform_t2}")

    if image_t1.shape != image_t2.shape:
        logging.info(f"Shape different : image 1 {image_t1.shape}, image 2 : {image_t2.shape}")


    valid_mask = valid_mask_t1 & valid_mask_t2    

    # Apply the mask to the TIFF data
    masked_image_t1 = image_t1.copy()
    masked_image_t2 = image_t2.copy()
    masked_image_t1[~valid_mask] = np.nan  # Using NaN to indicate masked areas
    masked_image_t2[~valid_mask] = np.nan  # Using NaN to indicate masked areas
    
    if np.isfinite(masked_image_t1).sum() == 0 or np.isfinite(masked_image_t2).sum() == 0:
        logging.info(f"masked_image_t1 or masked_image_t2 is all nan : {bounds}")
        return

    # Delete the padding
    masked_image_t1 = masked_image_t1[..., 1:-1, 1:-1]
    masked_image_t2 = masked_image_t2[..., 1:-1, 1:-1]

    scaling_factor = {"m": 1, "dm": 10, "cm": 100, "mm": 1000}
    masked_image_t1 = masked_image_t1 / scaling_factor[lidar_unit_t1]
    masked_image_t2 = masked_image_t2 / scaling_factor[lidar_unit_t2]

    # Save the masked output to a new TIFF file        
    masked_image_t1_path = output_lidar_path_t1 / f"masked_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}.tif"
    with rasterio.open(masked_image_t1_path, "w", **tif_meta_t1) as dest:
        dest.write(masked_image_t1.astype(np.float32), 1)
    logging.info(f"Saved in {masked_image_t1_path}")

    masked_image_t2_path = output_lidar_path_t2 / f"masked_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}.tif"
    with rasterio.open(masked_image_t2_path, "w", **tif_meta_t2) as dest:
        dest.write(masked_image_t2.astype(np.float32), 1)
    logging.info(f"Saved in {masked_image_t2_path}")
        

def check_valid_proportion(
    idx, 
    row, 
    vrt_path, 
    min_non_nan_proportion, 
    min_non_zero_proportion
):
    """
    Check if a single geometry has sufficient valid data proportion.
    
    Parameters:
    - idx_row: Tuple of (index, row) from GeoDataFrame
    - vrt_path: Path to the VRT file
    - min_non_nan_proportion: Minimum proportion of non-nan data required
    - min_non_zero_proportion: Minimum proportion of non-zero data required
    
    Returns:
    - Tuple of (index, is_valid, valid_proportion)
    """
    geometry = row["geometry"]
    
    # Get bounds for this geometry
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    
    # Get data for this geometry using get_window
    data, _ = get_window(
        vrt_path,
        bounds=bounds,
        resolution=None,
        resampling_method=None,
        open_even_oob=False,
    )

    if data is None :
        return idx, False
    # Calculate valid data proportion
    total_pixels = data.size
    if total_pixels == 0:
        return idx, False
    
    # Count valid pixels (not NaN and not nodata)
    valid_non_nan_mask = np.isfinite(data)  
    valid_non_zero_mask = data > 0

    valid_non_nan_pixels = np.sum(valid_non_nan_mask)
    valid_non_zero_pixels = np.sum(valid_non_zero_mask)
    
    valid_non_nan_proportion = valid_non_nan_pixels / total_pixels
    valid_non_zero_proportion = valid_non_zero_pixels / total_pixels
    
    # Check if it meets the minimum threshold
    is_valid_non_nan_proportion = valid_non_nan_proportion >= min_non_nan_proportion
    is_valid_non_zero_proportion = valid_non_zero_proportion >= min_non_zero_proportion
    
    is_valid = is_valid_non_nan_proportion and is_valid_non_zero_proportion
    return idx, is_valid
