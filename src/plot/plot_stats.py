import argparse
from tqdm import tqdm
import pandas as pd
import os
from config.utils import get_boundary_box, calc_percent_valid, dist_bwt_two_points, calc_std_wind_direction
import xarray as xr
import matplotlib.pyplot as plt

PATH_INFO = r'data/processed/annotations.txt'
PATH_DATASET = r'data/processed/dataset'
PATH_SAVE = r'images/figures'


def get_stats(radius: float) -> None:
    columns = ['percent', 'lat', 'lon', 'label',
               'width(km)', 'height(km)', 'std', 'rows', 'cols']
    info = pd.DataFrame(columns=columns)
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Plotting all cyclones in dataset"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            mask = ds['wvc_index'].notnull()
            ds = ds.where(mask, drop=True)
            min_lat, min_lon, max_lat, max_lon = get_boundary_box(
                ds, row['lat'], row['lon'], radius)
            percentage = calc_percent_valid(
                ds, min_lat, min_lon, max_lat, max_lon)
            dist_width = dist_bwt_two_points(
                min_lat, min_lon, min_lat, max_lon)
            dist_height = dist_bwt_two_points(
                min_lat, min_lon, max_lat, min_lon)
            std = calc_std_wind_direction(
                ds, min_lat, min_lon, max_lat, max_lon)
            info.loc[len(info)] = {'percent': percentage, 'lat': row['lat'], 'lon': row['lon'],  # type: ignore
                                   'label': row['label'], 'width(km)': dist_width, 'height(km)': dist_height, 'std': std, 'rows': ds['lon'].shape[0], 'cols': ds['lon'].shape[1]}

    plot_percentages(info)
    print(f'Average width (km): {info["width(km)"].mean()}')
    print(f'Average height (km): {info["height(km)"].mean()}')
    print(
        f'Average standard deviation of wind direction: {info["std"].mean()}')
    print(
    f'Average number of rows: {info["rows"].mean()}')
    print(
    f'Average number of columns: {info["cols"].mean()}')
    print(f'Unique column lengths: {info["cols"].unique()}')


def plot_percentages(df: pd.DataFrame) -> None:
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
            50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    fig = plt.figure(figsize=(12, 10))
    plt.hist(df['percent'], bins=bins, edgecolor='k')
    plt.xticks(bins)
    plt.xlabel(
        f'Percent of points in boundary box over ocean (%)')
    plt.ylabel('Number of cyclones')
    plt.title(
        f'Percentage of wind-speed points over ocean')
    plt.tight_layout()
    plt.grid()
    folder_path = os.path.join(PATH_SAVE, "percentages.png")
    plt.savefig(folder_path, format="png")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("radius", type=float)
    args = parser.parse_args()
    get_stats(args.radius)
