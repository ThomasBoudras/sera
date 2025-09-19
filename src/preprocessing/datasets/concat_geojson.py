import geopandas as gpd
from typing import List
import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np

@hydra.main(version_base="1.3", config_path="../../../configs/preprocessing/datasets", config_name="concat_geojson")
def concatenate_geojson_files(cfg: DictConfig) -> None:
    """
    Concatenate multiple GeoJSON files that have the same columns using Hydra configuration.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing:
        - geojson_paths: List of paths to GeoJSON files to concatenate
        - output_path: Path where to save the concatenated GeoJSON file
    """
    geojson_paths = cfg.geojson_paths
    output_path = cfg.output_path
    
    if not geojson_paths:
        raise ValueError("The list of GeoJSON paths cannot be empty")
    
    # Read all GeoDataFrames
    gdfs = []
    for path in geojson_paths:
        print(f"Reading GeoJSON file: {path}")
        gdf = gpd.read_file(path)
        gdfs.append(gdf)
    
    # Concatenate all GeoDataFrames
    concatenated_gdf = gpd.pd.concat(gdfs, ignore_index=True)

    # Add a 'split' column with train/val/test distribution     
    n_total = len(concatenated_gdf)
    n_train = int(cfg.train_proportion * n_total)
    n_val = int(cfg.val_proportion * n_total)
    n_test = n_total - n_train - n_val  # Remaining goes to test to ensure exact total
    
    # Create split array
    split_array = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    
    # Shuffle the split assignments randomly
    np.random.shuffle(split_array)
    
    # Add the split column to the GeoDataFrame
    concatenated_gdf['split'] = split_array
        
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the concatenated GeoDataFrame to a new GeoJSON file
    print(f"Saving concatenated GeoJSON to: {output_path}")
    concatenated_gdf.to_file(output_path, driver="GeoJSON")
    
    print(f"Successfully concatenated {len(geojson_paths)} GeoJSON files into {output_path}")
    print(f"Total number of features: {len(concatenated_gdf)}")


if __name__ == "__main__":
    concatenate_geojson_files()
