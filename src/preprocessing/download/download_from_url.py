
import geefetch
import geefetch.utils
import geefetch.utils.gee
import geopandas as gpd
import hydra
from omegaconf import DictConfig
from pandas.io.formats.style import save_to_buffer
from retry import retry
import logging
from shapely.ops import unary_union
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from geobbox import GeoBoundingBox
from rasterio.crs import CRS
from geefetch.cli.download_implementation import load_country_filter_polygon
from geefetch.data.get import download_custom
from geefetch.data.satellites.custom import CustomSatellite
from geefetch.utils.enums import CompositeMethod

@hydra.main(version_base=None, config_path="../../configs/preprocessing/download", config_name="dwd_from_url_pauls2025")
# @retry(exceptions=Exception, delay=10, tries=100)
def main(cfg: DictConfig) -> None:    

    gdf_path = Path(cfg.gdf_path).resolve()
    data_dir = Path(cfg.data_dir).resolve()
    
    initial_gdf = gpd.read_file(gdf_path)
    if cfg.split is not None:
        initial_gdf = initial_gdf[initial_gdf["split"] == cfg.split]

    save_gdf_path = data_dir / gdf_path.name
    initial_gdf.to_file(save_gdf_path, driver="GeoJSON")

    

    # To simplify the download process and reduce the number of images, 
    # each date is rounded to the 15th day of its respective month.
    initial_gdf["grouping_dates"] = initial_gdf[cfg.grouping_dates].astype(str).str[:6] + "15"
    
    # Group the geometries by the grouping dates
    grouped_by_df = (
        initial_gdf
        .groupby("grouping_dates")
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
        delayed(process_geometry)(row_gdf, cfg, data_dir)
        for _, row_gdf in tqdm(grouped_gdf.iterrows(), total=total_rows, desc="Downloading data")
    )
    logging.info("Download process completed.")


def process_geometry(row_gdf, cfg, data_dir):
    """Process a single geometry row and perform the download."""
    geefetch.utils.gee.auth("ee-thomasboudras04")
    
    date = row_gdf["grouping_dates"]
    geometry = row_gdf["geometry"]
    
    # Convert the date to get the first and last day of the year
    year = date[:4]
    year = max(int(year), int(cfg.min_year))
    year = str(min(int(year), int(cfg.max_year)))
    # We ensure that each spatially disjointed polygon is processed on a sepatly of the gdf, to avoid downloading data between the two that we're not interested in.
    if geometry.geom_type == 'MultiPolygon':
            polygons = [polygon for polygon in geometry.geoms]
    else :
        polygons = [geometry]

    print(f"date {date}, nb polygons {len(polygons)}")
    for polygon in polygons :
        bounds = polygon.bounds
        if "<year>" in cfg.url:
            url = cfg.url.replace("<year>", year)
            data_dir_year = data_dir / year
        else:
            url = cfg.url
        
        if not data_dir_year.exists():
            data_dir_year.mkdir(parents=True)

        left, bottom, right, top = bounds[0], bounds[1], bounds[2], bounds[3]
        tile_shape = int(min(cfg.tile_shape, max(abs(right - left), abs(top - bottom)))) # either tile_size is larger than WH: we want the whole area in one tile without necessarily downloading more, or tile size is smaller: we follow the size of tile_size
        bbox = GeoBoundingBox(left=left, right=right, top=top, bottom=bottom, crs=CRS.from_epsg(2154))

        class Satellelit_from_url(CustomSatellite):
            def __init__(self, url, name, selected_bands, pixel_range, resolution):
                super().__init__(url=url, pixel_range=pixel_range, name=name)
                self.selected_bands = list(selected_bands)
                self.pixel_range = tuple(pixel_range)
                self.resolution = resolution
            
            def bands(self):
                print(f"ceci est une selected_bands {type(self.selected_bands)}")
                return self.selected_bands

            def default_selected_bands(self):
                print(f"ceci est une selected_bands {type(self.selected_bands)}")
                return self.selected_bands

            def pixel_range(self):
                return self.pixel_range

            def resolution(self):
                return self.resolution

            def is_raster(self):
                return True

        composite_method = getattr(CompositeMethod, cfg.composite_method)
        
        satellite_custom = Satellelit_from_url(
            url=url, 
            name=cfg.name,
            pixel_range=cfg.pixel_range,
            selected_bands=cfg.selected_bands,
            resolution=cfg.resolution,
        )

        filter_polygon = (None if cfg.country is None else load_country_filter_polygon(cfg.country))
        download_custom(
            satellite_custom=satellite_custom,
            data_dir=data_dir_year,
            bbox=bbox,
            start_date=None,
            end_date=None,
            crs=CRS.from_epsg(2154),
            selected_bands=list(cfg.selected_bands),
            resolution=cfg.resolution,
            tile_shape=tile_shape,
            max_tile_size=cfg.max_tile_size,
            composite_method=composite_method,
            filter_polygon=filter_polygon,
        )


if __name__ == "__main__":
    main()


