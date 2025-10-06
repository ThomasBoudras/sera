import rasterio
from pathlib import Path
from typing import List, Tuple
import logging
import shutil
from tqdm import tqdm

def find_tifs_within_bounds(bounds_list: List[Tuple[float, float, float, float]], tif_directory: Path) -> List[Path]:
    """
    Find all TIF files in a directory that have bounds included within any of the given bounds.
    
    Parameters:
    - bounds_list: List of bounds tuples (minx, miny, maxx, maxy)
    - tif_directory: Path to directory containing TIF files
    
    Returns:
    - List of TIF file paths that have bounds within the given bounds
    """
    tif_directory = Path(tif_directory)
    tif_files = [file for file in tif_directory.iterdir() if file.suffix.lower() == ".tif"]
    
    matching_tifs = []
    
    for tif_file in tqdm(tif_files, total=len(tif_files), desc="Checking TIF files"):
        try:
            with rasterio.open(tif_file) as src:
                tif_bounds = src.bounds  # (left, bottom, right, top)
                
                # Check if tif bounds are included in any of the given bounds
                for bounds in bounds_list:
                    minx, miny, maxx, maxy = bounds
                    
                    # Check if tif bounds are completely within the given bounds
                    if (tif_bounds.left >= minx and 
                        tif_bounds.bottom >= miny and 
                        tif_bounds.right <= maxx and 
                        tif_bounds.top <= maxy):
                        matching_tifs.append(tif_file)
                        print(f"TIF {tif_file.name} has bounds {tif_bounds} included in bounds {bounds}")
                        break  # No need to check other bounds for this tif
                        
        except Exception as e:
            logging.warning(f"Could not read TIF file {tif_file}: {e}")
    
    return matching_tifs


def main():
    """
    Example usage of the find_tifs_within_bounds function
    """
    # Example bounds list - replace with your actual bounds
    bounds_list = [
        #(990712,6838291, 995562,6842301),
        #(1006905.0,6857919.8, 1010303.3,6862201.7),
        (976520,6832497,986007,6850993)
    ]
    
    # Example directory path - replace with your actual directory
    
    tif_directory = Path("/lustre/fsn1/projects/rech/ego/uof45xi/data/lidar/uncleaned_lidar/vosges_2022")
    output_directory = tif_directory.parent / "bad_tifs"
    output_directory.mkdir(exist_ok=True)

    matching_tifs = find_tifs_within_bounds(bounds_list, tif_directory)

    
    print(f"\nFound {len(matching_tifs)} TIF files with bounds included in the given bounds:")
    for tif_file in matching_tifs:
        print(f"  {tif_file}")
        # Copy the TIF file to the output directory
        destination = output_directory / tif_file.name
        shutil.copy2(tif_file, destination)
        print(f"    Copied to {destination}")
    
    print(f"\nAll bad TIF files have been copied to: {output_directory}")


if __name__ == "__main__":
    main()
