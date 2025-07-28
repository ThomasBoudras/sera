
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import geopandas as gpd
import hydra
from pathlib import Path
from lightning import seed_everything
from src import global_utils as utils

def from_float_to_str(number: int) -> str:
    number_str = str(number)
    formatted_str = number_str.replace('.', '-')
    return formatted_str

def mul(factor, number) :
    return factor * number

# Register the resolver

@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def compute_metrics(config: DictConfig) -> None:
    
    OmegaConf.register_new_resolver("from_float_to_str", from_float_to_str)
    OmegaConf.register_new_resolver("mul", mul) 
    if config.get("print_config"):
        utils.print_config(config, resolve=True, fields={"get_images", "get_metrics_local", "get_metrics_global", "get_plots"})
    OmegaConf.set_struct(config, True)

    if "seed" in config:
        seed_everything(config.seed)

    geometries = gpd.read_file(config.geometries_path)

    get_images =  hydra.utils.instantiate(config.get_images)
    get_metrics_local =  hydra.utils.instantiate(config.get_metrics_local)
    get_metrics_global =  hydra.utils.instantiate(config.get_metrics_global)

    if config.get_plots is not None :
        plot_images =  hydra.utils.instantiate(config.get_plots)
        nb_plots = min(len(geometries), plot_images.nb_plots) 
        indice_plot = np.random.randint(0, len(geometries)+1, nb_plots)

    metrics_global = {}
    for idx_row, row in tqdm(geometries.iterrows(), total=len(geometries), desc="Computes metrics"):
        bounds = row["geometry"].bounds

        images = get_images(bounds=bounds)

        metrics_local = get_metrics_local(images)

        if (config.get_plots is not None) and (idx_row in indice_plot) :
            plot_images(images, metrics_local, idx_plot=np.where(indice_plot == idx_row)[0][0])

        for name__metrics, value_metrics in metrics_local.items() :
            if name__metrics not in metrics_global :
                metrics_global[name__metrics] = [value_metrics]
            else :
                metrics_global[name__metrics].append(value_metrics)

    # Compute the average for each metric
    metrics_global = get_metrics_global(metrics_global)
    df = pd.DataFrame(list(metrics_global.items()), columns=['Metrics', 'Value'])
    print(df)
    output_path_xlsx = Path(config.output_path_xlsx).resolve()
    if output_path_xlsx.exists() :
        with pd.ExcelWriter(output_path_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=config.version_metrics_type)
    else :
        df.to_excel(output_path_xlsx, index=False, sheet_name=config.version_metrics_type)

