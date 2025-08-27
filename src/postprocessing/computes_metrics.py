
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import geopandas as gpd
import hydra
from pathlib import Path
from lightning import seed_everything
from src import global_utils as utils
from shapely.ops import unary_union

@hydra.main(version_base="1.3", config_path="../../configs/postprocessing", config_name="height_predictions.yaml")
def compute_metrics(config: DictConfig) -> None:
    
    if config.get("print_config"):
        utils.print_config(config, resolve=True, fields={"get_images", "get_metrics_local", "get_metrics_global", "get_plots"})
    OmegaConf.set_struct(config, True)

    if "seed" in config:
        seed_everything(config.seed)

    aoi_gdf = gpd.read_file(config.geometries_path)
    crs = aoi_gdf.crs
    # Regrouper les géométries par mois commun
    aoi_gdf["grouping_dates"] = aoi_gdf[config.grouping_dates].astype(str).str[:6] + "15"
    
    # Group the geometries by the grouping dates
    aoi_gdf = (
        aoi_gdf
        .groupby("grouping_dates")
        .agg({ 'geometry': lambda x: unary_union(x) })
        .reset_index()
    )
    aoi_gdf = aoi_gdf.set_geometry('geometry').explode(index_parts=False).reset_index(drop=True)
    aoi_gdf = gpd.GeoDataFrame(aoi_gdf, geometry='geometry', crs=crs)
    aoi_gdf_path = Path(config.save_dir) / "gdf_metrics.geojson"
    aoi_gdf.to_file(aoi_gdf_path, driver="GeoJSON")
    
    get_images =  hydra.utils.instantiate(config.get_images)
    get_metrics_local =  hydra.utils.instantiate(config.get_metrics_local)
    get_metrics_global =  hydra.utils.instantiate(config.get_metrics_global)

    if config.get_plots is not None :
        plot_images =  hydra.utils.instantiate(config.get_plots)
        nb_plots = min(len(aoi_gdf), plot_images.nb_plots) 
        indice_plot = np.random.choice(len(aoi_gdf), nb_plots, replace=False)
        count_plot = 0

    metrics_global = {}
    for idx_row, row in tqdm(aoi_gdf.iterrows(), total=len(aoi_gdf), desc="Computes metrics"):
        bounds = row["geometry"].bounds
        date = row["grouping_dates"]
        images = get_images(bounds=bounds, date=date)

        metrics_local = get_metrics_local(images)
        if (config.get_plots is not None) and (idx_row in indice_plot) :
            plot_images(images, metrics_local, idx_plot=count_plot)
            count_plot += 1

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
            df.to_excel(writer, index=False, sheet_name=config.version_metrics)
    else :
        df.to_excel(output_path_xlsx, index=False, sheet_name=config.version_metrics)

if __name__ == "__main__":
    compute_metrics()