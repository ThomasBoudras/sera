import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import binary_dilation, binary_erosion
from src.global_utils  import get_window
from shapely.geometry import shape


class get_images:
    def __init__(self, image_loaded_set, image_computed_set) :
        self.image_loaded_set = image_loaded_set
        self.image_computed_set = image_computed_set

    def __call__(self, bounds):
        res_images = {}
        res_profiles = {}

        #Retrieving images to be loaded 
        for name_image, image_loader in self.image_loaded_set.items() :
            image, profile = image_loader.load_image(bounds)
            res_images[name_image] = image 
            res_profiles[name_image] = profile

        #Retrieving images to be calculated 
        for name_image, image_computer in self.image_computed_set.items() :
            image = image_computer.compute_image(res_images, res_profiles)
            res_images[name_image] = image

        return res_images


class image_loader_model :
    def __init__(self, path, resolution=None, resampling_method=None, scaling_factor=1, min_image=None, max_image=None, classification_mask_path = None, classes_to_keep = None, open_even_oob=False):
        self.path = path
        self.resolution = resolution
        self.resampling_method = resampling_method
        self.scaling_factor = scaling_factor
        self.min_image = min_image
        self.max_image = max_image
        self.classification_mask_path = classification_mask_path
        self.classes_to_keep = classes_to_keep
        self.open_even_oob = open_even_oob

    def load_image(self, bounds):
        image, profile = get_window(
            self.path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob
            )
        
        if self.classification_mask_path:
                mask, _= get_window(
                    self.classification_mask_path,
                    geometry=None,
                    bounds=bounds,
                    resolution=self.resolution,
                    resampling_method=self.resampling_method,
                ).squeeze()
                for ix in np.unique(mask):
                    if ix not in self.classes_to_keep:
                        image[mask == ix] = np.nan

        image = image.astype(np.float32)*self.scaling_factor
        image = np.clip(image, self.min_image, self.max_image).squeeze()

        return image, profile


class difference_computer :
    def __init__(self, input_name_1, input_name_2, min_image, max_image):
        self.input_name_1 = input_name_1
        self.input_name_2 = input_name_2
        self.min_image = min_image
        self.max_image = max_image


    def compute_image(self, res_images, res_profiles):
        if self.input_name_1 not in res_images or  self.input_name_2 not in res_images :
            Exception(f"You must first load {self.input_name_1} and {self.input_name_2}")
            
        image_1 = res_images[self.input_name_1]
        image_2 = res_images[self.input_name_2]
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
            
        difference = res_images[self.input_name]
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

        image_1 = res_images[self.name_image_1]
        image_2 = res_images[self.name_image_2]
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

        image_1 = res_images[self.name_image_1]
        image_2 = res_images[self.name_image_2]
        profile = res_profiles[self.profile_name]

        nan_mask = np.isnan(image_1) | np.isnan(image_2)
        binary_map_1 = image_1 > self.threshold
        binary_map_2 = image_2 > self.threshold
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