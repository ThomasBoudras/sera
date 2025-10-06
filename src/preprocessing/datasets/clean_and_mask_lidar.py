import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import from_bounds
from tqdm import tqdm
import logging
from pathlib import Path
import shutil
from functools import partial
from joblib import Parallel, delayed
import geopandas as gpd
from shapely.geometry import box

from src.preprocessing.datasets.clean_and_mask_lidar_utils import create_grid, check_valid_proportion, get_masked_lidar_tiles, create_vrt

@hydra.main(version_base=None, config_path="../../../configs/preprocessing/datasets", config_name="clean_and_mask_lidar")
def main(cfg: DictConfig) -> None:
    
    logging.info(OmegaConf.to_yaml(cfg))
    
    initial_lidar_path_t1 = Path(cfg.t1.initial_lidar_path)
    list_tif_t1 = [file for file in initial_lidar_path_t1.iterdir() if file.suffix == ".tif"]
    vrt_path_t1 = initial_lidar_path_t1 / "full.vrt"
    if not vrt_path_t1.exists():
        create_vrt(list_tif_t1, str(vrt_path_t1))


    initial_lidar_path_t2 = Path(cfg.t2.initial_lidar_path)
    list_tif_t2 = [file for file in initial_lidar_path_t2.iterdir() if file.suffix == ".tif"]
    vrt_path_t2 = initial_lidar_path_t2 / "full.vrt"
    if not vrt_path_t2.exists():
        create_vrt(list_tif_t2, str(vrt_path_t2))


    with rasterio.open(vrt_path_t1) as src1, rasterio.open(vrt_path_t2) as src2:
        # Récupérer les bounds (xmin, ymin, xmax, ymax)
        bounds1 = src1.bounds
        bounds2 = src2.bounds
        crs1 = src1.crs
        crs2 = src2.crs
        assert crs1 == "EPSG:2154", "CRS is not EPSG:2154"
        assert crs1 == crs2, "CRS are not the same"
        
        # Calculer les bounds union
        union_bounds = (
            min(bounds1.left, bounds2.left),  # xmin
            min(bounds1.bottom, bounds2.bottom),  # ymin
            max(bounds1.right, bounds2.right),  # xmax
            max(bounds1.top, bounds2.top)  # ymax
        )

    list_bounds  = create_grid(union_bounds, tile_size=cfg.tile_size_grouped_tif, resolution=cfg.resolution_target, out_of_bounds_avoided=True)

    output_lidar_path_t1 = Path(cfg.t1.output_lidar_path)
    if output_lidar_path_t1.exists():
        shutil.rmtree(output_lidar_path_t1)
    output_lidar_path_t1.mkdir(parents=True)
    output_lidar_path_t2 = Path(cfg.t2.output_lidar_path)
    if output_lidar_path_t2.exists():
        shutil.rmtree(output_lidar_path_t2)
    output_lidar_path_t2.mkdir(parents=True)

    get_masked_lidar_tiles_partial = partial(
        get_masked_lidar_tiles, 
        vrt_path_t1=Path(vrt_path_t1), 
        vrt_path_t2=Path(vrt_path_t2), 
        output_lidar_path_t1=output_lidar_path_t1,
        output_lidar_path_t2=output_lidar_path_t2,
        min_area=cfg.min_area,
        replace_zero_t1=cfg.t1.replace_zero,
        replace_zero_t2=cfg.t2.replace_zero,
        lidar_unit_t1=cfg.t1.lidar_unit,
        lidar_unit_t2=cfg.t2.lidar_unit,
        resolution_target=cfg.resolution_target,
        no_data_t1=cfg.t1.no_data,
        no_data_t2=cfg.t2.no_data
        )

    Parallel(n_jobs=cfg.n_jobs)(
        delayed(get_masked_lidar_tiles_partial)(bounds)
        for bounds in tqdm(list_bounds, desc="Processing mask on tiles")
    )
    logging.info(f"Masking finished")
        
    output_lidar_path_t1 = Path(cfg.t1.output_lidar_path)
    list_masked_tifs_t1 = [tif for tif in output_lidar_path_t1.iterdir() if tif.suffix == ".tif"]
    vrt_masked_path_t1 = Path(cfg.t1.output_lidar_path) / "full.vrt"
    create_vrt(list_masked_tifs_t1, str(vrt_masked_path_t1))
    
    output_lidar_path_t2 = Path(cfg.t2.output_lidar_path)
    list_masked_tifs_t2 = [tif for tif in output_lidar_path_t2.iterdir() if tif.suffix == ".tif"]
    vrt_masked_path_t2 = output_lidar_path_t2 / "full.vrt"
    create_vrt(list_masked_tifs_t2, str(vrt_masked_path_t2))
    

    # Create grid tiles as geometries instead of loading from file
    grid_bounds = create_grid(union_bounds, tile_size=cfg.tile_size_geojson, resolution=None, out_of_bounds_avoided=False)
    
    # Convert bounds to polygon geometries
    geometries = [box(*bounds) for bounds in grid_bounds]

    if cfg.get("acquisition_date_gdf_path") is not None: 
        gdf_acquisition_date = gpd.read_file(cfg.acquisition_date_gdf_path)
    
    if cfg.t1.acquisition_date == "gdf_lidar_hd" or cfg.t2.acquisition_date == "gdf_lidar_hd":
        
        # Filter gdf to keep only geometries that intersect with union_bounds
        geometries_inter = []
        for idx, row in tqdm(gdf_acquisition_date.iterrows(), total=len(gdf_acquisition_date), desc="Filtering gdf to keep only geometries that intersect with union_bounds"):
            bounds_geometry = row.geometry.bounds
            if (
                union_bounds[0] <= bounds_geometry[2] and 
                union_bounds[1] <= bounds_geometry[3] and 
                bounds_geometry[0] <= union_bounds[2] and 
                bounds_geometry[1] <= union_bounds[3] 
            ):
                geometries_inter.append(row.copy())
        gdf_acquisition_date = gpd.GeoDataFrame(geometries_inter, crs=2154)
        
        gdf_rows = []
        for geometry in tqdm(geometries, total=len(geometries), desc="Finding closest acquisition date"):
            # Calculate distances to all geometries in gdf
            distances = gdf_acquisition_date.geometry.distance(geometry)

            # If minimum distance is 0 (overlapping geometries), check for multiple overlaps
            if distances.min() == 0:
                # Find all geometries with distance 0 (overlapping)
                overlapping_indices = distances[distances == 0].index
                
                if len(overlapping_indices) > 1:
                    # Calculate intersection area for each overlapping geometry
                    max_intersection_area = 0
                    best_idx = overlapping_indices[0]
                    
                    for idx in overlapping_indices:
                        intersection = geometry.intersection(gdf_acquisition_date.loc[idx, 'geometry'])
                        intersection_area = intersection.area
                        
                        if intersection_area > max_intersection_area:
                            max_intersection_area = intersection_area
                            best_idx = idx
                    
                    closest_idx = best_idx
                else:
                    closest_idx = overlapping_indices[0]
            else:
                # Find the index of the closest geometry
                closest_idx = distances.idxmin()

            
            # Get the closest geometry
            lidar_acquisition_date = gdf_acquisition_date.loc[closest_idx, 'lidar_acquisition_date']
            
            if cfg.t1.acquisition_date == "gdf_lidar_hd":
                gdf_rows.append(
                        {
                            'geometry': geometry, 
                            'lidar_acquisition_date_t1': lidar_acquisition_date, 
                            'lidar_acquisition_date_t2': cfg.t2.acquisition_date,
                            'lidar_unit_t1': "m",
                            'lidar_unit_t2': "m",
                        }
                    )
            else:
                gdf_rows.append(
                    {
                        'geometry': geometry, 
                        'lidar_acquisition_date_t1': cfg.t1.acquisition_date, 
                        'lidar_acquisition_date_t2': lidar_acquisition_date,
                        'lidar_unit_t1': "m",
                        'lidar_unit_t2': "m",
                    }
                )
                        
        gdf = gpd.GeoDataFrame(gdf_rows, crs=2154)
    
    else :
        # Create GeoDataFrame from grid geometries
        gdf = gpd.GeoDataFrame(
            {
                'geometry': geometries, 
                'lidar_acquisition_date_t1': cfg.t1.acquisition_date, 
                'lidar_acquisition_date_t2': cfg.t2.acquisition_date,
                'lidar_unit_t1': "m",
                'lidar_unit_t2': "m",
            }, 
            crs=2154
        )
    
    logging.info(f"Loaded GeoJSON with {len(gdf)} features")
    
    # Process in parallel
    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(check_valid_proportion)(idx, row, vrt_masked_path_t1, cfg.min_non_nan_proportion, cfg.min_non_zero_proportion) 
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Checking valid data proportion")
    )    
    # Extract valid indices and log statistics
    valid_indices = []
    for idx, is_valid in results:
        if is_valid:
            valid_indices.append(idx)
    gdf_inter = gdf.loc[valid_indices].reset_index(drop=True).copy()
    logging.info(f"Filtered to {len(valid_indices)} features (from {len(gdf)})")
    
    # Process in parallel
    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(check_valid_proportion)(idx, row, vrt_masked_path_t2, cfg.min_non_nan_proportion, cfg.min_non_zero_proportion) 
        for idx, row in tqdm(gdf_inter.iterrows(), total=len(gdf_inter), desc="Checking valid data proportion")
    )    
    # Extract valid indices and log statistics
    valid_indices = []
    for idx, is_valid in results:
        if is_valid:
            valid_indices.append(idx)
    gdf_final = gdf_inter.loc[valid_indices].reset_index(drop=True).copy()
    logging.info(f"Filtered to {len(valid_indices)} features (from {len(gdf_inter)})")
            
    # Save the filtered GeoJSON
    output_gdf_path = Path(cfg.output_gdf_path)
    output_gdf_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_final.to_file(output_gdf_path, driver='GeoJSON')
    logging.info(f"Saved filtered GeoJSON to {output_gdf_path}")

if __name__ == "__main__":
    main()