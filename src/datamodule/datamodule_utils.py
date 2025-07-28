import numpy as np
from torch.utils.data import Sampler
from shapely.geometry import box
import geopandas as gpd
       

class SubsetSampler(Sampler):
    def __init__(self, data_source, num_samples, shuffle=True):
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples



def generate_random_bounds(bounds, patch_size, resolution):
    bounds = list(bounds)
    available_width = bounds[2] - bounds[0] - patch_size
    available_height = bounds[3] - bounds[1] - patch_size

    x_offset = np.random.randint(0, available_width // resolution + 1)
    y_offset = np.random.randint(0, available_height // resolution + 1)

    bounds[0] += x_offset * resolution
    bounds[1] += y_offset * resolution
    bounds[2] = bounds[0] + patch_size
    bounds[3] = bounds[1] + patch_size

    return bounds


def get_grid(global_bounds, patch_size, margin_size):
    patch_bounds = []
    
    #We subtract the margin we are going to cut in 
    #order to be sure of predicting at least all the 
    #bounds initially requested.
    min_x = global_bounds[0] - margin_size
    min_y = global_bounds[1] - margin_size
    max_x = global_bounds[2] + margin_size
    max_y = global_bounds[3] + margin_size

    assert ((max_x - min_x) > patch_size) and ((max_y - min_y) > patch_size)
    x_start = min_x 
    y_start = min_y

    while (x_start < max_x - 3 * margin_size) : 
        # We set 3 margins: two for the edges to reduce border effects, 
        # and a third to create an overlap between neighbouring patches. 
        # This allows the values to be averaged over the overlap area, 
        # thus reducing discontinuities between patches.

        x_stop = x_start + patch_size 
        y_stop = y_start + patch_size

        # Save pixel_patch_bounds
        if x_stop <= max_x and y_stop <= max_y :
            patch_bounds.append((x_start, y_start, x_stop, y_stop))

        elif x_stop > max_x and y_stop <= max_y :
            x_stop = max_x
            patch_bounds.append((max_x - patch_size, y_start, max_x, y_stop))
        
        elif x_stop <= max_x and y_stop > max_y :
            y_stop = max_y
            patch_bounds.append((x_start, max_y - patch_size, x_stop, max_y))
        
        elif x_stop > max_x and y_stop > max_y :
            y_stop, x_stop = max_y, max_x
            patch_bounds.append((max_x - patch_size, max_y - patch_size, max_x, max_y))

        if y_stop  < max_y :
            y_start = y_stop - 3 * margin_size
        else:
            y_start = min_y
            x_start = x_stop - 3 * margin_size
    return patch_bounds

def expand_gdf(gdf, patch_size, margin_size) :
    new_gdf = []
    for i, row in gdf.iterrows() :
        bounds = row['geometry'].bounds
        list_bounds = get_grid( global_bounds=bounds, patch_size=patch_size, margin_size=margin_size)
        for sub_bounds in list_bounds :
            new_row = row.copy()
            new_row["geometry"] = box(*sub_bounds)
            new_row["geometry_id"] = i
            new_gdf.append(new_row)
    return gpd.GeoDataFrame(new_gdf, crs=gdf.crs).reset_index()
