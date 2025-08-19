import geopandas as gpd
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import hydra
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import numpy as np
from scipy.stats import t   
from collections import defaultdict
import json

@hydra.main(config_path="../../configs/preprocessing", config_name="data_on_geojson")
def main(config):
    save_dir = Path(config.save_dir).resolve()
    gdf_path = Path(config.geojson_path).resolve()
    min_n_days = config.min_n_days
    max_n_days = config.max_n_days
    resolution = config.resolution
    list_date_field = config.list_date_field
    ref_date_field = config.ref_date_field
    confidence_level = config.confidence_level
    gdf = load_geojson(gdf_path, list_date_field, ref_date_field)
    df = get_data_on_geojson(gdf, min_n_days, max_n_days, list_date_field, ref_date_field, resolution)
    plot_figures(df, min_n_days, max_n_days, confidence_level, save_dir, resolution)

def load_geojson(gdf_path, list_date_field, ref_date_field) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gdf_path)
    gdf[ref_date_field] = pd.to_datetime(gdf[ref_date_field])
    gdf[list_date_field] = gdf[list_date_field].apply(json.loads) 
    # For each list of tuples in the vrt_list_field, keep only the first element (date) of each tuple and convert it to datetime
    gdf[list_date_field] = gdf[list_date_field].apply(
        lambda lst: pd.to_datetime([t[0] for t in lst])
    )
    return gdf

def count_within_n_days(list_dates, ref_date, max_n_days) -> int:
    delta = timedelta(days=max_n_days//2)
    return sum(abs(d - ref_date) <= delta for d in list_dates)


def get_data_on_geojson(gdf, min_n_days, max_n_days, list_date_field, ref_date_field, resolution) -> pd.DataFrame:
    results = {"mean_nb_date": [], "std_nb_date": [], "percentage_samples": defaultdict(list)}
    for n_days in tqdm(range(min_n_days, max_n_days + 3*resolution +1, resolution), desc="Computing mean number of valid dates "):
        nb_dates = gdf.apply(
            lambda row: count_within_n_days(row[list_date_field], row[ref_date_field], n_days),
            axis=1
        )
        results["mean_nb_date"].append(nb_dates.mean())
        results["std_nb_date"].append(nb_dates.std())
        results["percentage_samples"][1].append(len(nb_dates)/len(gdf))

        for min_dates in range(2, 9, 2):
            nb_dates_with_min_dates = nb_dates >= min_dates
            results["percentage_samples"][min_dates].append(nb_dates_with_min_dates.sum()/len(gdf)) 

    return results

def plot_figures(df, min_n_days, max_n_days, confidence_level, save_dir, resolution):
    """
    Plot the mean number of valid dates per day with standard deviation,
    and also plot the number of samples per day for several minimum image count thresholds.
    Also, print the results in a formatted table.
    """

    # Prepare the data for the mean number of valid dates and its standard deviation
    nb_days = np.arange(min_n_days, max_n_days + 3*resolution + 1, resolution)
    means = np.array(df["mean_nb_date"])
    stds = np.array(df["std_nb_date"])
    min_dates_list = sorted([k for k in df["percentage_samples"].keys()])
    percentage_samples = {}
    for min_dates in min_dates_list:
        percentage_samples[min_dates] = np.round(np.array(df["percentage_samples"][min_dates])*100, 2)
    # DataFrame for the mean and standard deviation
    plot_df = pd.DataFrame({
        "nb_days": nb_days,
        "mean_nb_date": means,
        "std_nb_date": stds
    })

    # Print the results in a formatted table (showing std instead of interval)
    # Create the first dataset: mean and std of valid dates, with n_days as index
    summary_df = pd.DataFrame({
        "mean_nb_date": means,
        "std_nb_date": stds
    }, index=nb_days)
    summary_df.index.name = "nb_days"
    print("\nSummary Table: Mean number of valid dates and standard deviation")
    print(summary_df.to_string())

    # Save the first dataset to CSV
    summary_csv_path = save_dir / "summary_mean_std_valid_dates.csv"
    summary_df.to_csv(summary_csv_path)
    print(f"\nSaved summary table to {summary_csv_path}")

    # Create the second dataset: percentage of samples for each min_dates threshold, with n_days as index
    samples_df = pd.DataFrame(percentage_samples, index=nb_days)
    samples_df.index.name = "nb_days"
    print("\nNumber of samples per ±N days for different minimum image thresholds:")
    print(samples_df.to_string())

    # Save the second dataset to CSV
    samples_csv_path = save_dir / "samples_per_n_days_min_images.csv"
    samples_df.to_csv(samples_csv_path)
    print(f"\nSaved samples per n_days table to {samples_csv_path}")

    # Plot mean number of valid dates with standard deviation as shaded area
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="nb_days",
        y="mean_nb_date",
        label="Mean number of valid dates"
    )
    plt.fill_between(
        plot_df["nb_days"],
        plot_df["mean_nb_date"] - plot_df["std_nb_date"],
        plot_df["mean_nb_date"] + plot_df["std_nb_date"],
        color='b',
        alpha=0.2,
        label='±1 std'
    )
    plt.xlabel('nb days of acquisition')
    plt.ylabel('Mean number of valid dates')
    plt.title('Mean number of valid dates as a function of ±N days interval')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_dir / "data_on_geojson_mean_nb_dates.png")

    # Plot the number of samples per n_days for several minimum image thresholds
    plt.figure(figsize=(10, 6))
    for min_dates in min_dates_list:
        # For each min_dates, we need to compute the number of samples per n_days
        plt.plot(
            nb_days,
            np.round(np.array(df["percentage_samples"][min_dates])*100, 2),
            label=f"min {min_dates} images"
        )

    plt.xlabel('Nb days of acquisition')
    plt.ylabel('Percentage of samples used in the dataset')
    plt.title('Percentage of samples in the dataset used per nb days of acquisition for different minimum image thresholds')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_dir / "data_on_geojson_%_samples_used.png")

if __name__ == "__main__":
    main()
    
