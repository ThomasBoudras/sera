import numpy as np
from osgeo import gdal
import shutil
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import from_bounds
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
import os


def create_grid(bounds, tile_size):
    """
    Create a grid of square tiles within the given bounds.

    Parameters:
    - bounds: A tuple of (min_x, min_y, max_x, max_y) in the same CRS as tile_size.
    - tile_size: The edge length of the square tile in meters.

    Returns:
    - A list of tuples (min_x, min_y, max_x, max_y) representing the grid tiles.
    """
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    # Calculate the number of tiles in each dimension
    x_count = int(np.ceil(width / tile_size))
    y_count = int(np.ceil(height / tile_size))

    # Generate the tiles
    tiles = []
    for x in range(x_count):
        for y in range(y_count):
            # Calculate the tile's bounds
            tile_min_x = min_x + x * tile_size 
            tile_min_y = min_y + y * tile_size 
            tile_max_x = min(tile_min_x + tile_size, max_x) 
            tile_max_y = min(tile_min_y + tile_size, max_y) 

            # Create the tile as a a tuple of bounds and add it to the list
            tile = (int(tile_min_x), int(tile_min_y), int(tile_max_x), int(tile_max_y))
            tiles.append(tile)

    return tiles


def create_vrt(files_list, output_path):
    # Create a VRT
    dataset = gdal.BuildVRT(output_path, files_list)
    dataset.FlushCache()  # Ensure all data is written
    dataset = None  # Close the dataset
    

def regroup_tiles(
    cfg,
    initial_tifs_path,
    regrouped_tifs_path,
):
        
    list_tif = [file for file in initial_tifs_path.iterdir() if file.endswith(".tif")]
    vrt_path = initial_tifs_path / "full.vrt"
    create_vrt(list_tif, vrt_path)

    if regrouped_tifs_path.exists():
        shutil.rmtree(regrouped_tifs_path)
    regrouped_tifs_path.mkdir(parents=True)

    with rasterio.open(vrt_path) as src:
        vrt_bounds = src.bounds
        bounds = vrt_bounds.left, vrt_bounds.bottom, vrt_bounds.right, vrt_bounds.top

    # define new tiles regions, so as to match original tiles without overlaps
    grid_tiles = create_grid(bounds, cfg.tile_size)

    new_files_list = []
    with rasterio.open(vrt_path) as src:
        nodata_value = cfg.nodata if cfg.nodata != "nan" else np.nan
        vrt_bounds = src.bounds
        original_crs = src.crs
        target_crs = "EPSG:2154"

        for new_tile_geometry in tqdm(grid_tiles, desc="Processing each new tile"):
            (min_x, min_y, max_x, max_y) = new_tile_geometry
            out_path = regrouped_tifs_path / f"reshaped_tif_{min_x}_{min_y}_{max_x}_{max_y}.tif"
                        
            # Check if the window is within the VRT
            if (
                new_tile_geometry[0] >= vrt_bounds.left
                and new_tile_geometry[2] <= vrt_bounds.right
                and new_tile_geometry[1] >= vrt_bounds.bottom
                and new_tile_geometry[3] <= vrt_bounds.top
            ):
                window = from_bounds(*new_tile_geometry, src.transform)
                window_data = src.read(window=window).astype(np.float32)
                if not (window_data == nodata_value).all():
                    # Define the transform for the new (windowed) dataset
                    window_transform = src.window_transform(window)
                    # Create a new dataset based on the window
                    new_files_list.append(out_path)
                    
                    if original_crs.to_string() != target_crs:
                        raise Exception(f"Tiff {vrt_path} has a different SRC from EPSG:2154")
                    
                    with rasterio.open(
                        out_path,
                        "w",
                        driver="GTiff",
                        height=window_data.shape[1],
                        width=window_data.shape[2],
                        count=src.count,
                        dtype="float32",
                        crs=src.crs,
                        transform=window_transform,
                        nodata=nodata_value
                    ) as dst:
                        dst.write(window_data)
            else:
                raise ValueError(f"Tile {new_tile_geometry} is outside the VRT bounds.")

    vrt_path = regrouped_tifs_path / "full.vrt"
    create_vrt(new_files_list, vrt_path)


def get_nan_mask_from_lidar(
    lidar_path: str,
    bounds: tuple,
    replace_zero: bool,
    min_area
):
    """
    Generate a NaN mask from a lidar file for a given area.
    """
    with rasterio.open(lidar_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        transform_image = rasterio.windows.transform(window, src.transform)

        image = src.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float32)
        image[image == src.nodata] = np.nan
        image[image < 0] = np.nan
        
        image = np.pad(image, pad_width=1, mode='constant', constant_values=np.nan)
        
        mask = ~np.isfinite(image)
        if replace_zero : 
            mask = np.logical_or(mask, image == 0)

        tif_meta = src.meta.copy()
        tif_meta.update({
            "height": window.height,
            "width": window.width,
            "transform": transform_image,
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
    
    final_mask = np.zeros(image.shape, dtype=bool)
    for geometry in gdf.geometry:         
        mask_geo = geometry_mask(
        [mapping(geometry)], 
        transform=transform_image, 
        invert=False, 
        out_shape=image.shape
        )
        final_mask = mask_geo | final_mask

    return image, final_mask, transform_image, tif_meta, gdf


def plot_masking_results(
    plot_folder,
    bounds,
    image_1,
    masked_image_1,
    transform_image_1,
    gdf_1,
    image_2,
    masked_image_2,
    transform_image_2,
    gdf_2
):
    """Plots and saves the results of the masking process."""
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Plot for image 1 before mask
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    show(
        image_1,
        ax=axes,
        transform=transform_image_1,
        vmin=0,
        cmap="Greens",
    )
    gdf_1.boundary.plot(ax=axes, edgecolor="red", linewidth=1)
    path_image_1_before = plot_folder / f"plot_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}_image_1_before.png"
    fig.savefig(path_image_1_before, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Plot for image 1 after mask
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    show(
        masked_image_1,
        ax=axes,
        transform=transform_image_1,
        vmin=0,
        cmap="Greens",
    )
    gdf_1.boundary.plot(ax=axes, edgecolor="red", linewidth=1)
    gdf_2.boundary.plot(ax=axes, edgecolor="blue", linewidth=1)
    path_image_1_after = plot_folder / f"plot_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}_image_1_after.png"
    fig.savefig(path_image_1_after, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Plot for image 2 before mask
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    vmax_2 = np.nanmean(masked_image_2) * 2 if not np.isnan(masked_image_2).all() else 0
    show(
        image_2,
        ax=axes,
        transform=transform_image_2,
        vmin=0,
        vmax=vmax_2,
        cmap="Greens",
    )
    gdf_2.boundary.plot(ax=axes, edgecolor="blue", linewidth=1)
    path_image_2_before = plot_folder / f"plot_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}_image_2_before.png"
    fig.savefig(path_image_2_before, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Plot for image 2 after mask
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    show(
        masked_image_2,
        ax=axes,
        transform=transform_image_2,
        vmin=0,
        vmax=vmax_2,
        cmap="Greens",
    )
    gdf_1.boundary.plot(ax=axes, edgecolor="red", linewidth=1)
    gdf_2.boundary.plot(ax=axes, edgecolor="blue", linewidth=1)
    path_image_2_after = plot_folder / f"plot_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}_image_2_after.png"
    fig.savefig(path_image_2_after, bbox_inches="tight", pad_inches=0)
    plt.close()
