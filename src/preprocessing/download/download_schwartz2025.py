
import geopandas as gpd
import hydra
import pystac_client
import rasterio
import teledetection as tld
from omegaconf import DictConfig
from retry import retry
import logging
from shapely.ops import unary_union
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

@hydra.main(version_base=None, config_path="../../../configs/preprocessing/download", config_name="dwd_schwartz2025")
@retry(exceptions=Exception, delay=10, tries=100)
def main(cfg: DictConfig) -> None:    

    gdf_path = Path(cfg.gdf_path).resolve()
    data_dir = Path(cfg.data_dir).resolve()
    
    initial_gdf = gpd.read_file(gdf_path)

    save_gdf_path = data_dir / gdf_path.name
    initial_gdf.to_file(save_gdf_path, driver="GeoJSON")

    if cfg.split is not None:
        initial_gdf = initial_gdf[initial_gdf["split"] == cfg.split]

    # To simplify the download process and reduce the number of images, 
    # each date is rounded to the 15th day of its respective month.
    initial_gdf["grouping_year"] = initial_gdf[cfg.grouping_dates].astype(str).str[:4]
    
    # Group the geometries by the grouping dates
    grouped_by_df = (
        initial_gdf
        .groupby("grouping_year")
        .agg({ 'geometry': lambda x: unary_union(x) })
        .reset_index()
    )
    grouped_gdf = gpd.GeoDataFrame(grouped_by_df, geometry='geometry', crs=initial_gdf.crs)

    grouped_gdf_filename = gdf_path.stem + "_grouped.geojson"
    grouped_gdf_path = data_dir / grouped_gdf_filename
    grouped_gdf.to_file(grouped_gdf_path, driver="GeoJSON")

    # Parallelisation 
    total_rows = len(grouped_gdf)
    logging.info("Starting parallel download.")
    Parallel(n_jobs=cfg.n_jobs_parrallelized)(
        delayed(process_geometry)(row_gdf, cfg)
        for _, row_gdf in tqdm(grouped_gdf.iterrows(), total=total_rows, desc="Downloading data")
    )
    logging.info("Download process completed.")


def process_geometry(row_gdf, cfg):
    """Process a single geometry row and perform the download from STAC API."""
    # Log in is required through Data Terra (possible with ORCID)
    api = pystac_client.Client.open(
        "https://api.stac.teledetection.fr",
        modifier=tld.sign_inplace,
    )
    year = row_gdf["grouping_year"]
    geometry = row_gdf["geometry"]

    stac_item = (
        api.get_collection("forms-t").get_item(f"FORMS-T-{year}").get_assets()[cfg.asset_type]
    )
    asset_url = stac_item.href

    
    # We ensure that each spatially disjointed polygon is processed on a sepatly of the gdf, to avoid downloading data between the two that we're not interested in.
    if geometry.geom_type == 'MultiPolygon':
            polygons = [polygon for polygon in geometry.geoms]
    else :
        polygons = [geometry]

    for i, polygon in enumerate(tqdm(polygons, desc =f"Download polygons",total=len(polygons))) :
        bounds = polygon.bounds
        
        try:
            with rasterio.open(asset_url) as src:
                # get the window for the aoi
                window = src.window(*bounds)
                data = src.read(1, window=window)

                # Get metadata for writing
                profile = src.profile
                profile.update({
                    'driver': 'GTiff',
                    'height': window.height,
                    'width': window.width,
                    'transform': src.window_transform(window),
                    'compress': 'lzw'
                })

                # Define output path
                output_dir = Path(cfg.data_dir).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / str(year) / f"{cfg.asset_type}_{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}.tif"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data, 1)
        except Exception as e:
            logging.error(f"Failed to download and save for polygon {i} on year {year}, bounds {bounds}. Error: {e}")


if __name__ == "__main__":
    main()