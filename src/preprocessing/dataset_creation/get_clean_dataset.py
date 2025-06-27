import geopandas as gpd
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed

from src.datamodules.dataset_utils import get_window


@hydra.main(version_base=None, config_path="config", config_name="get_clean_dataset")
def main(cfg: DictConfig) -> None:
    wandb.init(
        project="download_data",
        name="clean_dataset"
    )
    
    logging.info(OmegaConf.to_yaml(cfg))
    
    gdf = gpd.read_file(cfg.gdf_in_path)
    
    logging.info("Starting to compute 'vrt_list_timeseries' with parallel processing.")
    tqdm_desc = "Find correct VRT for geometries"
    geometries['vrt_list_timeseries'] = Parallel(n_jobs=-1)(
        delayed(get_correct_vrt)(row["geometry"], row["lidar_acquisition_date"], cfg.input_path) 
        for _, row in tqdm(geometries.iterrows(), total=len(geometries), desc=tqdm_desc)
    )
    
    tqdm.pandas(desc="Delete bounds without data")
    logging.info("Filtering out rows with empty 'vrt_list_timeseries'.")
    cleaned_geometries = geometries[geometries['vrt_list_timeseries'].apply(lambda x: len(x) > 0)]
    uncleaned_geometries = geometries[geometries['vrt_list_timeseries'].apply(lambda x: len(x) == 0)]

    cleaned_geometries.to_file(cfg.geometries_cleaned_path, driver="GeoJSON")
    uncleaned_geometries.to_file(cfg.geometries_uncleaned_path, driver="GeoJSON")
    logging.info(f"Cleaned geometries saved to {cfg.geometries_cleaned_path}. Process complete.")

    wandb.finish()
    


def get_date_from_vrt_name(vrt_name) :
    return datetime.strptime(vrt_name.split('_')[1].split('.')[0], "%Y%m%d")
        
def sort_by_proximity(target_file, file_list):
    target_date = get_date_from_vrt_name(target_file)
    return sorted(file_list, key=lambda file: abs((get_date_from_vrt_name(file) - target_date).days))

def get_correct_vrt(geometry, lidar_month, input_path):
    bounds = geometry.bounds
    lidar_month = str(lidar_month)[:6]
        
    list_correct_vrt = []

    s1_vrt_list_path = os.path.join(input_path, f"lidar_date_{lidar_month}/s1/vrt_files")
    s2_vrt_list_path = os.path.join(input_path, f"lidar_date_{lidar_month}/s2/vrt_files")
    s1_vrt_list = [file for file in os.listdir(s1_vrt_list_path) if file.endswith('.vrt')]
    s2_vrt_list = sorted([file for file in os.listdir(s2_vrt_list_path) if file.endswith('.vrt')])
    
    for s2_vrt in s2_vrt_list :
        try :
            s2_image = get_window(
                image_path=os.path.join(s2_vrt_list_path, s2_vrt),
                bounds=bounds,
                )
        except Exception as e:
            logging.info(f"Problem with {os.path.join(s2_vrt_list_path, s2_vrt)} : {e}")
            s2_image = np.array([])

        if s2_image is None or s2_image.size == 0 or not np.isfinite(s2_image).any() :
            continue

        sorted_s1_list = sort_by_proximity(s2_vrt, s1_vrt_list) #We are looking for the nearest tensor s1 in terms of date 
        for s1_vrt in sorted_s1_list :
            try:
                s1_image = get_window(
                    image_path=os.path.join(s1_vrt_list_path, s1_vrt),  
                    bounds=bounds,
                    )
            except Exception as e:
                logging.info(f"Problem with {os.path.join(s1_vrt_list_path, s1_vrt)} : {e}")
                s1_image = np.array([])

            if s1_image is not None and len(s1_image) > 0  and np.isfinite(s1_image).any():
                list_correct_vrt.append([s2_vrt, s1_vrt])
                break
        
    return list_correct_vrt


if __name__ == "__main__":
    main()



def create_vrt_timeseries(data_dir: Path, reference_date):
    data_dir = data_dir / f"lidar_date_{reference_date}"
    
    s1_path = data_dir / "s1"
    dict_vrt_s1_asc = defaultdict(list) 
    dict_vrt_s1_dsc = defaultdict(list) 
    for path in s1_path.glob("*tif") :
        date_s1 = path.stem.split("_")[4][:8]
        new_file_path = ".." / Path(path.parent.stem) / path.stem
        with rasterio.open(path) as src:
            s1_orbit = src.tags()["orbitProperties_pass"]
        if s1_orbit == "ASCENDING" :
            dict_vrt_s1_asc[date_s1].append(new_file_path)
        else :
            dict_vrt_s1_dsc[date_s1].append(new_file_path)
        
        

    s2_path = data_dir / "s2" 
    dict_vrt_s2 = defaultdict(list)
    for path in s2_path.glob("*tif") :
        date_s2 = path.stem[:8]
        new_file_path = ".." / Path(path.parent.stem) / path.stem
        dict_vrt_s2[date_s2].append(new_file_path)
   