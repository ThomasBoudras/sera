from tkinter import W
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import binary_dilation, binary_erosion
from src.global_utils  import get_window
from shapely.geometry import shape
from shapely.geometry import box
from rasterio.features import rasterize

class get_images:
    def __init__(self, image_loaded_set, image_computed_set) :
        self.image_loaded_set = image_loaded_set
        self.image_computed_set = image_computed_set

    def __call__(self, row):
        bounds = row["geometry"].bounds
        date = row["grouping_dates"]

        res_images = {}
        res_profiles = {}

        #Retrieving images to be loaded 
        for name_image, image_loader in self.image_loaded_set.items() :
            image, profile = image_loader.load_image(bounds, date)
            res_images[name_image] = image 
            res_profiles[name_image] = profile

        #Retrieving images to be calculated if they are not None
        if self.image_computed_set is not None:         
            for name_image, image_computer in self.image_computed_set.items() :
                image = image_computer.compute_image(res_images, res_profiles)
                res_images[name_image] = image

        return res_images


class input_image_loader :
    def __init__(self, path, resolution, resampling_method, open_even_oob, channel_to_keep):
        self.path = path
        self.resolution = resolution
        self.resampling_method = resampling_method
        self.open_even_oob = open_even_oob
        self.channel_to_keep = channel_to_keep

    def load_image(self, bounds, date):
        if "<date>" in self.path :
            rounded_date = date[:6] + "15"
            path = self.path.replace("<date>", rounded_date)
        elif "<year>" in self.path :
            path = self.path.replace("<year>", date[:4])
        else:
            path = self.path

        image, profile = get_window(
            path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob
            )
        if self.channel_to_keep is not None:
            image = image[self.channel_to_keep, ...]
        image = image.astype(np.float32)
        # Normalisation min-max
        image[~np.isfinite(image)] = 0
        image_min = np.nanmin(image)
        image_max = np.nanmax(image)
        image = (image - image_min) / (image_max - image_min)

        return image, profile


class output_image_loader :
    def __init__(self, path, resolution, resampling_method, scaling_factor, min_image, max_image, open_even_oob, max_date, min_date):
        self.path = path
        self.resolution = resolution
        self.resampling_method = resampling_method
        self.scaling_factor = scaling_factor
        self.min_image = min_image
        self.max_image = max_image
        self.open_even_oob = open_even_oob
        self.max_date = max_date 
        self.min_date = min_date

    def load_image(self, bounds, date):
        limited_date = date
        if self.max_date is not None:
            limited_date = str(min(int(date), int(self.max_date)))

        if self.min_date is not None:
            limited_date = str(max(int(limited_date), int(self.min_date)))

        path = self.path
        if "<year>" in self.path :
            path = path.replace("<year>", limited_date[:4])
        if "<date>" in self.path :
            path = path.replace("<date>", limited_date)

        image, profile = get_window(
            path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob
            )
        image = image.astype(np.float32)*self.scaling_factor
        image = np.clip(image, self.min_image, self.max_image).squeeze()

        return image, profile


class mask_image_loader :
    def __init__(self, classification_mask_path, forest_mask_path, resolution, classes_to_keep):
        self.classification_mask_path = classification_mask_path
        self.forest_mask_gdf = gpd.read_parquet(forest_mask_path) if forest_mask_path is not None else None
        self.resolution = resolution
        self.classes_to_keep = classes_to_keep
        
    def load_image(self, bounds, date):
        year = date[:4]
        raster_bounds = box(*bounds)
        mask_path = self.classification_mask_path.replace("<year>", year)
        classification, profile = get_window(
            mask_path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method="nearest",
        )
        classification = classification.squeeze()

        classif_mask = classification == self.classes_to_keep[0]
        if len(self.classes_to_keep) > 1:
            for aclass in self.classes_to_keep[1::]:
                classif_mask = classif_mask | (classification == aclass)
    
        if self.forest_mask_gdf is not None:
            clipped_gdf = gpd.clip(self.forest_mask_gdf, raster_bounds)     
            geometries = [(geom, 1) for geom in clipped_gdf.geometry]
            if len(geometries):
                mask_forest = rasterize(
                    geometries,
                    out_shape=classification.shape,
                    transform=profile["transform"],
                    fill=0,
                    default_value=1,
                    dtype=np.uint8,
                ).astype(bool)
            else:
                mask_forest = np.zeros_like(classif_mask, dtype=bool)
        else :
            mask_forest = np.zeros_like(classif_mask, dtype=bool)
        
        final_mask = classif_mask | mask_forest
        return final_mask, None


class masked_image_computer :
    def __init__(self, input_name, mask_name):
        self.input_name = input_name
        self.mask_name = mask_name

    def compute_image(self, res_images, res_profiles):
        if self.input_name not in res_images:
            Exception(f"You must first load {self.input_name}")
            
        image = res_images[self.input_name].copy()
        mask = res_images[self.mask_name].copy()

        image[~mask] = np.nan
        return image


class difference_computer :
    def __init__(self, input_name_1, input_name_2, min_image, max_image):
        self.input_name_1 = input_name_1
        self.input_name_2 = input_name_2
        self.min_image = min_image
        self.max_image = max_image

    def compute_image(self, res_images, res_profiles):
        if self.input_name_1 not in res_images or  self.input_name_2 not in res_images :
            Exception(f"You must first load {self.input_name_1} and {self.input_name_2}")
            
        image_1 = res_images[self.input_name_1].copy()
        image_2 = res_images[self.input_name_2].copy()
        difference = image_2 - image_1
        difference = np.clip(difference, self.min_image, self.max_image).astype(np.float32)
        return difference


class change_threshold_computer :
    def __init__(self, input_name, threshold, min_area, profile_name = None):
        self.input_name = input_name
        self.profile_name = self.input_name if profile_name is None else profile_name
        self.threshold = threshold
        self.min_area = min_area

    def compute_image(self, res_images, res_profiles):
        if self.input_name not in res_images  :
            Exception(f"You must first load {self.input_name}")
            
        difference = res_images[self.input_name].copy()
        profile = res_profiles[self.profile_name]

        nan_mask = np.isnan(difference)
        changes = difference < self.threshold

        return apply_min_area(changes=changes, profile=profile, min_area=self.min_area, nan_mask=nan_mask)
  
   
class change_percentage_computer :
    def __init__(self, name_image_1, name_image_2, threshold, min_area, profile_name = None):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        self.profile_name = self.name_image_1 if profile_name is None else profile_name
        self.threshold = threshold
        self.min_area = min_area

    def compute_image(self, res_images, res_profiles):
        if (self.name_image_1 not in res_images) or  (self.name_image_2 not in res_images):
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = res_images[self.name_image_1].copy()
        image_2 = res_images[self.name_image_2].copy()
        profile = res_profiles[self.profile_name]
        nan_mask = np.isnan(image_1) | np.isnan(image_2)
        changes =  (image_2 - image_1) / (image_1 + 1e-6) < self.threshold
        #avoid the noise of tree less than 2 meters
        changes[image_1 < 2] = False

        return apply_min_area(changes=changes, profile=profile, min_area=self.min_area, nan_mask=nan_mask)
    

class change_treecover_computer :
    def __init__(self, name_image_1, name_image_2, threshold, min_area, profile_name = None):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        self.profile_name = self.name_image_1 if profile_name is None else profile_name
        self.threshold = threshold
        self.min_area = min_area

    def compute_image(self, res_images, res_profiles):
        if (self.name_image_1 not in res_images) or  (self.name_image_2 not in res_images):
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = res_images[self.name_image_1].copy()
        image_2 = res_images[self.name_image_2].copy()
        profile = res_profiles[self.profile_name]

        nan_mask = np.isnan(image_1) | np.isnan(image_2)
        binary_map_1 = image_1 >= self.threshold
        binary_map_2 = image_2 >= self.threshold
        changes = np.logical_and(binary_map_1, ~binary_map_2)

        return apply_min_area(changes=changes, profile=profile, min_area=self.min_area, nan_mask=nan_mask)


def apply_min_area(changes, profile, min_area, nan_mask):
    size = 3
    # Apply morphology to filter out noise
    changes = binary_erosion(changes, structure=np.ones((size, size)))
    changes = binary_dilation(changes, structure=np.ones((size, size)), iterations=2)
    changes = binary_erosion(changes, structure=np.ones((size, size)))

    # Transform array into polygons
    polygons = [
        shape(geom)
        for geom, value in features.shapes(changes.astype(np.int16), transform=profile["transform"])
        if value == 1
    ]

    # Convert polygons to GeoDataFrame
    gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=profile["crs"])

    # Filter with min_area (not necessary since already filtered)
    gdf_filtered = gdf[gdf.geometry.area > min_area]

    # Mask the changes
    # Create a mask based on the geometry
    if len(gdf_filtered["geometry"].values):
        changes = rasterio.features.geometry_mask(
            gdf_filtered["geometry"].values,
            transform=profile["transform"],
            invert=True,
            out_shape=changes.shape,
        )
    else:
        changes = np.zeros(changes.shape)
    changes = changes.astype(np.float32)
    changes[nan_mask] = np.nan
    return changes