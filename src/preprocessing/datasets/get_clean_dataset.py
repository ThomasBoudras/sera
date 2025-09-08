import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

@hydra.main(version_base=None, config_path="../../configs/preprocessing/datasets", config_name="get_clean_dataset")
def main(cfg: DictConfig) -> None:    
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    create_vrts = hydra.utils.get_method(cfg.create_vrts)
    get_valid_vrts = hydra.utils.get_method(cfg.get_valid_vrts)
    data_dir = Path(cfg.data_dir).resolve()
    initial_gdf = gpd.read_file(cfg.initial_gdf_path)
    initial_gdf["grouping_dates"] = initial_gdf[cfg.grouping_dates].astype(str).str[:6] + "15"
    grouping_dates = initial_gdf["grouping_dates"].drop_duplicates().tolist()
    
    #create vrts for the dataframe
    # Parallelize the creation of VRTs for each date
    Parallel(n_jobs=cfg.n_jobs_parrallelized)(
        delayed(create_vrts)(data_dir, date)
        for date in tqdm(grouping_dates, desc="Create VRT for each date", total=len(grouping_dates))
    )
    
    logging.info("Starting to compute 'vrt_list_timeseries' with parallel processing.")
    tqdm_desc = "Find correct VRT for each geometries"
    initial_gdf[cfg.validation_column] = Parallel(n_jobs=cfg.n_jobs_parrallelized)(
        delayed(get_valid_vrts)(data_dir, row["geometry"], row["grouping_dates"]) 
        for _, row in tqdm(initial_gdf.iterrows(), total=len(initial_gdf), desc=tqdm_desc)
    )
    
    tqdm.pandas(desc="Delete bounds without data")
    logging.info("Filtering out rows with no valid vrt'.")
    clean_gdf = initial_gdf[initial_gdf[cfg.validation_column].notnull()].reset_index(drop=True).copy()
    unclean_gdf = initial_gdf[initial_gdf[cfg.validation_column].isnull()].reset_index(drop=True).copy()
    
    clean_gdf.to_file(cfg.gdf_clean_path, driver="GeoJSON")
    unclean_gdf.to_file(cfg.gdf_unclean_path, driver="GeoJSON")
    logging.info(f"Cleaned geodataframe saved to {cfg.gdf_clean_path}. Process complete.")


if __name__ == "__main__":
    main()




