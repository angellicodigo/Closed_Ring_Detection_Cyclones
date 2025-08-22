import os
import xarray as xr
import argparse
import numpy as np
from typing import Optional
import matplotlib.patches as patches
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from pathlib import Path
from config.utils import get_center, nearest_neighbors_indices, dist_bwt_two_points, get_mean_info, get_boundary_box, get_num_points, calc_percent_valid, get_segmentation_map
import pandas as pd
from tqdm import tqdm

PATH_FOLDER = r"C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\images"
PATH_CENTERS = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\external\TRACKS_CL7.dat'
PATH_DATASET = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\dataset'
PATH_PLOT = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info\dataset_final.txt'
PATH_PLOT_ALL_SAVE = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\images\dataset_final_angles'


def search_dataset(file_path: Optional[Path], query_lon: Optional[float], query_lat: Optional[float], radius: float, window_size: float, isBBox: bool) -> None:

    if file_path is None:
        df = pd.read_csv(PATH_PLOT, sep=r'\t', engine='python')
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Plotting all cyclones in dataset"):
            # Need Path() to not get a warning
            path = Path(os.path.join(PATH_DATASET, row['file_name']))
            open_file(path, row['lon'], row['lat'],
                      radius, window_size, True, isBBox)
    else:
        path = Path(os.path.join(PATH_DATASET, file_path))
        open_file(path, query_lon, query_lat,
                  radius, window_size, False, isBBox)


def open_file(file_path: Path, query_lon: Optional[float], query_lat: Optional[float], radius: float, window_size: float, plotAll: bool, isBBox: bool):
    with xr.open_dataset(file_path) as ds:
        average_time, year, month, day = get_mean_info(ds)
        hour = average_time.hour
        title = str(os.path.basename(file_path))

        if (query_lon == None) or (query_lat == None):
            cyclone_id = int(title.split('_')[1][9:])
            query_lat, query_lon = get_center(
                cyclone_id, year, month, day, average_time)

        row_indices, col_indices = nearest_neighbors_indices(
            ds, query_lat, query_lon)
        nearest_row = row_indices[0]
        nearest_col = col_indices[0]
        dim = list(ds.sizes)
        row_dim = dim[0]
        col_dim = dim[1]
        nearest_point = ds.isel({row_dim: nearest_row, col_dim: nearest_col})
        
        plot(ds, query_lat, query_lon, year, month, day, hour, radius, window_size, isBBox)

        if plotAll:
            folder_path = os.path.join(PATH_PLOT_ALL_SAVE, f"{title}.png")
        else:
            folder_path = os.path.join(PATH_FOLDER, f"{title}.png")
        plt.savefig(
            folder_path, format="png", dpi=1200)

        if not plotAll:
            print(f'Name: {title}')
            print(
                f'Center (lat, lon): {query_lat} {query_lon}')
            print(
                f'Nearest Neighbor (lat, lon): {nearest_point["lat"].values} {nearest_point["lon"].values}')
            print(
                f'Distance from Nearnest Neighbor: {dist_bwt_two_points(query_lat, query_lon, nearest_point["lat"].values, nearest_point["lon"].values)}')  # type: ignore
            print(
                f'Does nearest neighbor have wind speed? {nearest_point["wind_speed"].values}')
            print(f'Average time: {average_time}')
            plt.show()
        
        plt.close()
    

def plot(ds: xr.Dataset, query_lat: float, query_lon: float,  year: int, month: int, day: int, hour: int, radius: float, window_size: float, isBBox: bool) -> None:
    plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    boundaries = np.arange(0, 32.6, 2.5)
    cmap = plt.get_cmap("turbo")
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)
    U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
    V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))
    quiver = ax.quiver(ds['lon'], ds['lat'], U, V, ds['wind_speed'],
                       cmap='turbo', transform=ccrs.PlateCarree(), scale=500, pivot='mid', norm=norm) # type: ignore
    cbar = plt.colorbar(quiver)
    cbar.set_label("Wind Speed (m/s)")
    cbar.set_ticks(boundaries)  # type: ignore
    ax.coastlines()  # type: ignore
    gridlines = ax.gridlines(draw_labels=True)  # type: ignore
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.bottom_labels = True
    gridlines.left_labels = True
    plt.tight_layout()
    plt.xlim(round(query_lon-window_size), round(query_lon+window_size))
    plt.ylim(round(query_lat-window_size), round(query_lat+window_size))
    plt.plot(query_lon, query_lat, 'x', markersize=12,
             color="purple", transform=ccrs.PlateCarree())

    if isBBox:
        min_lat, min_lon, max_lat, max_lon = get_boundary_box(query_lat, query_lon, radius)

        plot_boundary_box(ax, min_lat, min_lon, max_lat,
                          max_lon, 'black')

        plot_percent_valid(ds, min_lat, min_lon, max_lat,
                           max_lon, 'deepskyblue', 'green')

        num = get_num_points(ds, query_lat, query_lon, radius, True)
        percent = calc_percent_valid(ds, query_lat, query_lon, radius, True)
        plt.title(
            f'{year}-{month}-{day} {hour} UTC (N={num}; {round(percent, 1)}% over ocean)')
    elif radius != 0:
        plot_semantic_segmentation(ds, query_lat, query_lon, radius, 'grey')
        num = get_num_points(ds, query_lat, query_lon, radius, False)
        percent = calc_percent_valid(ds, query_lat, query_lon, radius, False)
        plt.title(
            f'{year}-{month}-{day} {hour} UTC (N={num}; {round(percent, 1)}% over ocean)')
    else:
        plt.title(f'{year}-{month}-{day} {hour} UTC')


def plot_semantic_segmentation(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float, color: str) -> None:
    mask = get_segmentation_map(ds, query_lat, query_lon, radius)
    mask_lats = ds['lat'].where(mask).values
    mask_lons = ds['lon'].where(mask).values

    plt.scatter(mask_lons, mask_lats, c=color, s=100, marker='o', alpha=0.75, transform=ccrs.PlateCarree())

def plot_percent_valid(ds: xr.Dataset, min_lat: float, min_lon: float, max_lat: float, max_lon: float, color_ocean: str, color_land: str) -> None:
    mask = (min_lon <= ds.lon) & (ds.lon <= max_lon) & (
        min_lat <= ds.lat) & (ds.lat <= max_lat)
    lon_mask = ds['lon'].values[mask]
    lat_mask = ds['lat'].values[mask]
    wind_speed_mask = ds['wind_speed'].values[mask]
    non_nan = ~np.isnan(wind_speed_mask)
    plt.scatter(lon_mask[non_nan], lat_mask[non_nan],
                s=8, color=color_ocean, transform=ccrs.PlateCarree())
    plt.scatter(lon_mask[~non_nan], lat_mask[~non_nan],
                s=8, color=color_land, transform=ccrs.PlateCarree())


def plot_boundary_box(ax: Axes, min_lat: float, min_lon: float, max_lat: float, max_lon: float, color: str) -> None:
    width = max_lon - min_lon
    height = max_lat - min_lat

    mid_lat = (min_lat + max_lat) / 2
    mid_lon = (min_lon + max_lon) / 2

    dist_width = dist_bwt_two_points(min_lat, min_lon, min_lat, max_lon)
    dist_height = dist_bwt_two_points(min_lat, min_lon, max_lat, min_lon)

    bbox = patches.Rectangle(
        (min_lon, min_lat), width, height, linewidth=2, edgecolor=color, facecolor='none')
    plt.annotate(f"{dist_width:.2f} km", xy=(mid_lon, min_lat), xytext=(
        0, -5), textcoords='offset pixels', ha='center', va='top', color=color)
    plt.annotate(f"{dist_height:.2f} km", xy=(min_lon, mid_lat), xytext=(
        -5, 0), textcoords='offset pixels', ha='right', va='center', rotation=90, color=color)

    ax.add_patch(bbox)
    plt.plot(min_lon, max_lat, 'x', markersize=12,
             color='crimson', transform=ccrs.PlateCarree())
    plt.plot(max_lon, min_lat, 'x', markersize=12,
             color='teal', transform=ccrs.PlateCarree())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=Path)
    parser.add_argument("--query_lat", type=float)
    parser.add_argument("--query_lon", type=float)
    parser.add_argument("--radius", type=float, default=0)
    parser.add_argument("--window_size", type=float, default=3)
    parser.add_argument("--isBBox", type=float, default=False)

    args = parser.parse_args()
    search_dataset(args.file_name, args.query_lon,
                   args.query_lat, args.radius, args.window_size, args.isBBox)
