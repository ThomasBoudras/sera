
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import geopandas as gpd
import hydra
import os

from lightning import seed_everything

def compute_metrics(cfg: DictConfig) -> None:
    if "seed" in cfg:
        seed_everything(cfg.seed)

    geometries = gpd.read_file(cfg.geometries_path)

    get_images =  hydra.utils.instantiate(cfg.get_images)
    get_metrics_local =  hydra.utils.instantiate(cfg.get_metrics_local)
    get_metrics_global =  hydra.utils.instantiate(cfg.get_metrics_global)

    if cfg.get_plots is not None :
        plot_images =  hydra.utils.instantiate(cfg.get_plots)
        nb_plots = min(len(geometries), plot_images.nb_plots) 
        indice_plot = np.random.randint(0, len(geometries)+1, nb_plots)

    metrics_global = {}
    for idx_row, row in tqdm(geometries.iterrows(), total=len(geometries), desc="Computes metrics"):
        bounds = row["geometry"].bounds

        images = get_images(bounds=bounds)

        metrics_local = get_metrics_local(images)

        if (cfg.get_plots is not None) and (idx_row in indice_plot) :
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
    if os.path.exists(cfg.output_path_xlsx) :
        with pd.ExcelWriter(cfg.output_path_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, index=False, sheet_name=cfg.version_metrics_type)
    else :
        df.to_excel(cfg.output_path_xlsx, index=False, sheet_name=cfg.version_metrics_type)

