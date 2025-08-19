import hydra
from omegaconf import DictConfig, OmegaConf
import geopandas as gpd
from pathlib import Path

@hydra.main(version_base="1.3", config_path="../../configs/postprocessing", config_name="get_predict_dataset")
def filter_and_save_predict_geojson(cfg: DictConfig):
    """
    Main function to filter a reference GeoJSON using Hydra config.
    The config should contain:
        reference_geojson_path: str or Path
        unwanted_geojson_paths: list of str or Path
        output_geojson_path: str or Path
    """
    print(OmegaConf.to_yaml(cfg))
    reference_geojson_path = Path(cfg.reference_geojson_path)
    unwanted_geojson_paths = [Path(p) for p in cfg.unwanted_geojson_paths]
    output_geojson_path = Path(cfg.output_geojson_path)

    # Load the reference GeoJSON
    gdf_ref = gpd.read_file(reference_geojson_path)

    # Collect unwanted geometries from all unwanted GeoJSONs
    unwanted_geometries = set()
    for unwanted_geojson_path in unwanted_geojson_paths:
        gdf_unwanted = gpd.read_file(unwanted_geojson_path)
        unwanted_geometries.update(gdf_unwanted['geometry'].apply(lambda geom: geom.wkb))

    # Filter the reference GeoDataFrame by geometry
    gdf_filtered = gdf_ref[~gdf_ref['geometry'].apply(lambda geom: geom.wkb).isin(unwanted_geometries)]

    # Save the filtered GeoDataFrame to a new GeoJSON
    gdf_filtered.to_file(output_geojson_path, driver='GeoJSON')
    print(f"Filtered GeoJSON saved to {output_geojson_path}")

if __name__ == "__main__":
    filter_and_save_predict_geojson()
