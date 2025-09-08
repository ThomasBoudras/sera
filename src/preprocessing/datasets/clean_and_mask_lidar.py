import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import from_bounds
from tqdm import tqdm
import logging
from pathlib import Path
import shutil

from src.preprocessing.datasets.clean_and_mask_lidar_utils import create_grid, regroup_tiles, get_nan_mask_from_lidar, plot_masking_results

@hydra.main(version_base=None, config_path="../../configs/preprocessing/datasets", config_name="clean_and_mask_lidar")
def main(cfg: DictConfig) -> None:
    
    logging.info(OmegaConf.to_yaml(cfg))
    """Combine .tif files into larger files in order to decrease the number of stored files"""
    # TODO for lidar, additional operations needed to handle the json files

    print(OmegaConf.to_yaml(cfg))
    
    initial_path_1 = Path(cfg.t1.initial_path)
    regroup_tiles(cfg, initial_path_1)

    initial_path_2 = Path(cfg.t2.initial_path)
    regroup_tiles(cfg, initial_path_2)

    with rasterio.open(cfg.t1.initial_path) as src1, rasterio.open(cfg.t2.initial_path) as src2:
        # Récupérer les bounds (xmin, ymin, xmax, ymax)
        bounds1 = src1.bounds
        bounds2 = src2.bounds
        
        # Calculer les bounds union
        union_bounds = (
            min(bounds1.left, bounds2.left),  # xmin
            min(bounds1.bottom, bounds2.bottom),  # ymin
            max(bounds1.right, bounds2.right),  # xmax
            max(bounds1.top, bounds2.top)  # ymax
        )

    list_bounds  = create_grid(union_bounds, tile_size=cfg.tile_size)

    for i, bounds in tqdm(enumerate(list_bounds),total=len(list_bounds)) :
        image_1, mask_1, transform_1, tif_meta_1, gdf_1 = get_nan_mask_from_lidar(
            lidar_path=cfg.t1.initial_path,
            bounds=bounds,
            replace_zero=cfg.replace_zero_1,
            min_area=cfg.min_area
        )
        
        image_2, mask_2, transform_2, tif_meta_2, gdf_2 = get_nan_mask_from_lidar(
            lidar_path=cfg.t2.initial_path,
            bounds=bounds,
            replace_zero=cfg.replace_zero_2,
            min_area=cfg.min_area
        )
        
        if transform_1 != transform_2 :
            print(f"### transform 1 :{transform_1},\n transform 2 {transform_2}")

        if image_1.shape != image_2.shape :
            Exception(f"Shape different : image 1 {image_1.shape}, image 2 : {image_2.shape}")

        print(f"nb nan mask 1 {(~mask_1).sum()}, mask 2 {(~mask_2).sum()}")

        mask = np.logical_and(mask_1, mask_2)
        # Apply the mask to the TIFF data
        masked_image_1 = np.where(mask, image_1, np.nan)  # Using NaN to indicate masked areas
        masked_image_2 = np.where(mask, image_2, np.nan)  # Using NaN to indicate masked areas

        if cfg.mask_plot_folder is not None :
            plot_masking_results(
                plot_folder=cfg.mask_plot_folder,
                i=i,
                bounds=bounds,
                image_1=image_1,
                masked_tif_data_1=masked_image_1,
                transform_image_1=transform_1,
                gdf_1=gdf_1,
                image_2=image_2,
                masked_tif_data_2=masked_image_2,
                transform_image_2=transform_2,
                gdf_2=gdf_2
            )
        
        # Delete the padding
        masked_image_1 = masked_image_1[..., 1:-1, 1:-1]
        masked_image_2 = masked_image_2[..., 1:-1, 1:-1]

        final_folder_1 = Path(cfg.t1.final_folder)
        final_folder_2 = Path(cfg.t2.final_folder)

        if not final_folder_1.exists():
            final_folder_1.mkdir(parents=True)

        if not final_folder_2.exists():
            final_folder_2.mkdir(parents=True)
        
        # Save the masked output to a new TIFF file        
        path_image_masked_1 = final_folder_1 / f"masked_{bounds[0]}_{bounds[1]}_{bounds[2]}_{bounds[3]}.tif"
        with rasterio.open(path_image_masked_1, "w", **tif_meta_1) as dest:
            dest.write(masked_image_1.astype(np.float32), 1)
        print(f"Saved in {path_image_masked_1}")

        path_image_masked_2 = final_folder_2 / f"masked_{bounds[0]}_{bounds[1]}_{bounds[2]}_{bounds[3]}.tif"
        with rasterio.open(path_image_masked_2, "w", **tif_meta_2) as dest:
            dest.write(masked_image_2.astype(np.float32), 1)
        print(f"Saved in {path_image_masked_2}")

if __name__ == "__main__":
    main()





