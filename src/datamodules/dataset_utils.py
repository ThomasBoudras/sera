import numpy as np
import math
import rasterio
from affine import Affine
from rasterio.enums import Resampling
import rasterio.transform
from rasterio.windows import from_bounds
from datetime import datetime, timedelta
import torchvision.transforms.functional as F
from torch.utils.data import Sampler
from skimage.measure import block_reduce
from rasterio.windows import from_bounds
from shapely.geometry import box
import geopandas as gpd

def get_window(
    image_path,
    geometry=None,
    bounds=None,
    resolution=None,
    return_profile=False,
    resampling_method=None,
    open_even_oob= False
):
    """Retrieve a window from an image, within given bounds or within the bounds of a geometry"""
    with rasterio.open(image_path) as src:
        profile = src.profile
        if bounds is None:
            if geometry is not None:
                bounds = geometry.bounds
            else:
                bounds = src.bounds

        init_resolution = profile["transform"].a
        real_height, real_width = bounds[2] - bounds[0], bounds[3] - bounds[1]
        assert real_height % resolution == 0, f"Please choose a patchsize divisible by the requested resolution : {resolution}m"

        src_bounds = src.bounds
        bounds_within_vrt = (
            bounds[0] >= src_bounds.left and  
            bounds[1] >= src_bounds.bottom and 
            bounds[2] <= src_bounds.right and 
            bounds[3] <= src_bounds.top  
        )
        
        if not bounds_within_vrt and not open_even_oob:
            print(f"no intersection betwween vrt and bounds : path {image_path}, bounds : {bounds}, vrt bounds {vrt_bounds}") 
            return np.array([])
        
        window = from_bounds(*bounds, transform=src.transform)
        transform = src.window_transform(window)

        if (resolution is not None) and (init_resolution != resolution):
            window_height, window_width = int(real_height/resolution), int(real_width/resolution)
            resampling_method = resampling_method if resampling_method is not None else "bilinear"
            resampling_method = getattr(Resampling, resampling_method)
            transform = Affine(
                resolution,
                transform.b,
                transform.c,
                transform.d,
                -resolution,
                transform.f,
            )
        else:
            window_height, window_width = window.height, window.width
            resampling_method = None

        data = src.read(
            out_shape=(
                src.count,
                int(window_height),
                int(window_width),
                ),
            window=window,
            resampling=resampling_method,
            boundless=True,
            fill_value=np.nan
            )

    if return_profile:
        new_profile = profile.copy()
        count = data.shape[0] if len(data.shape) == 3 else 1
        height = data.shape[-2]
        width = data.shape[-1]

        new_profile.update(
            {
                "transform": transform,
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": count,
            }
        )
        return data, new_profile
    else:
        return data
    
    
class BottomLeftCrop:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        return F.crop(img, 0, 0, self.patch_size, self.patch_size)
        

def found_nearest_date(date, min_year, max_year):
    current_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
    
    possible_final_dates = {
        2021 : ["05-01", "07-01", "09-01", "11-01"],
        2022 : ["01-01", "03-01", "05-01", "07-01", "09-01", "11-01"],
        2023 : ["01-01", "03-01", "05-01"],
        }
    
    final_date = None
    final_year = None
    min_deltatime = timedelta.max  

    for year in range(min_year, max_year + 1):
        for possible_date in possible_final_dates[year]:
            possible_final_date = datetime.strptime(f"{year}-{possible_date}", "%Y-%m-%d")
            delta_time = abs(current_date - possible_final_date)
            if delta_time < min_deltatime:
                final_date = possible_date
                final_year = year
                min_deltatime = delta_time

    return final_date, final_year
    

class SubsetSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))

    def __iter__(self):
        indices = list(range(self.num_samples))
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
def get_patch_bounds(bounds, patch_size, position):
    if position not in {"bottom-left", "top-left", "top-right", "bottom-right"}:
        raise ValueError("La position doit être 'bottom-left', 'top-left', 'top-right' ou 'bottom-right'.")

    xmin, ymin, xmax, ymax = bounds

    # Calcul de nouvelles bornes en fonction de la position
    if position == "bottom-left":
        x_start = xmin
        y_start = ymin
    elif position == "top-left":
        x_start = xmin
        y_start = ymax - patch_size
    elif position == "top-right":
        x_start = xmax - patch_size
        y_start = ymax - patch_size
    elif position == "bottom-right":
        x_start = xmax - patch_size
        y_start = ymin

    # Calcul des coordonnées finales
    x_end = x_start + patch_size
    y_end = y_start + patch_size

    # S'assurer que les bounds restent dans les limites globales
    if x_start < xmin or y_start < ymin or x_end > xmax or y_end > ymax:
        raise ValueError("Le patch dépasse les limites globales.")

    return (x_start, y_start, x_end, y_end)


def generate_random_bounds(bounds, patch_size, resolution_x, resolution_y):
    bounds = list(bounds)
    available_width = bounds[2] - bounds[0] - patch_size
    available_height = bounds[3] - bounds[1] - patch_size

    x_offset = np.random.randint(0, available_width // resolution_x + 1)
    y_offset = np.random.randint(0, available_height // resolution_y + 1)

    bounds[0] += x_offset * resolution_x
    bounds[1] += y_offset * resolution_y
    bounds[2] = bounds[0] + patch_size
    bounds[3] = bounds[1] + patch_size

    return bounds


def get_grid(patch_size, crop_size, global_bounds):
    patch_bounds = []
    
    min_x = global_bounds[0] - crop_size
    min_y = global_bounds[1] - crop_size
    max_x = global_bounds[2] + crop_size
    max_y = global_bounds[3] + crop_size
    assert ((max_x - min_x) > patch_size) and ((max_y - min_y) > patch_size)
    x_start = min_x 
    y_start = min_y

    while (x_start < max_x - 3 * crop_size) :
        x_stop = x_start + patch_size
        y_stop = y_start + patch_size
        # Save pixel_patch_bounds
        if x_stop <= max_x and y_stop <= max_y :
            patch_bounds.append((x_start, y_start, x_stop, y_stop))
        elif x_stop > max_x and y_stop <= max_y :
            patch_bounds.append((max_x - patch_size, y_start, max_x, y_stop))
        elif x_stop <= max_x and y_stop > max_y :
            patch_bounds.append((x_start, max_y - patch_size, x_stop, max_y))
        elif x_stop > max_x and y_stop > max_y :
            patch_bounds.append((max_x - patch_size, max_y - patch_size, max_x, max_y))

        if y_stop  < max_y:
            y_start = y_stop - 3 * crop_size
        else:
            y_start = min_y
            x_start = x_stop - 3 * crop_size
    return patch_bounds

def expand_geometries(gdf, patch_size, crop_size) :
    new_gdf = []
    for i, row in gdf.iterrows() :
        bounds = row['geometry'].bounds
        list_bounds = get_grid(patch_size=patch_size, crop_size=crop_size, global_bounds=bounds)
        for sub_bounds in list_bounds :
            new_row = row.copy()
            new_row["geometry"] = box(*sub_bounds)
            new_row["number_geometry"] = i
            new_gdf.append(new_row)
    return gpd.GeoDataFrame(new_gdf, crs=gdf.crs).reset_index()
