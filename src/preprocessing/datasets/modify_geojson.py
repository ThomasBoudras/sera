import json
import sys
import hydra
from omegaconf import DictConfig
import geopandas as gpd
from pathlib import Path

@hydra.main(config_path="../../configs/preprocessing/datasets", config_name="get_clean_geojson")
def main(cfg: DictConfig):
    data = gpd.read_file(cfg.gdf_path)

    # Drop columns
    for col in cfg.columns_to_drop:
        data.drop(col, axis=1, inplace=True)
    
    # Add columns   
    for col, value in cfg.columns_to_add.items():
        data[col] = value
    
    output_path = Path(cfg.gdf_path).resolve().parent /   f"{cfg.gdf_output_name}.geojson"
    data.to_file(output_path, driver="GeoJSON")
    

if __name__ == "__main__":
    main()