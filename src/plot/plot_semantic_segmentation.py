import argparse
from tqdm import tqdm
import pandas as pd
import os
from utils import get_segmentation_map, get_mean_info
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import ListedColormap

PATH_INFO = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info\dataset_preprocessed.txt'
PATH_DATASET = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\dataset'
PATH_SAVE = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info'
PATH_SAVE_PLOTS = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\images\segmentation'


def get_stats(radius: float, window_size: float) -> None:
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Plotting semantic segmentation images"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            mask = get_segmentation_map(ds, row['lat'], row['lon'], radius)
            plt.figure(figsize=(12, 12))
            ax = plt.axes(projection=ccrs.PlateCarree())
            U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
            V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))
            quiver = ax.quiver(ds['lon'], ds['lat'], U, V, ds['wind_speed'],
                            cmap='turbo', transform=ccrs.PlateCarree(), scale=500, pivot='mid')
            cbar = plt.colorbar(quiver)
            cbar.set_label("Wind Speed")
            mask_lats = ds['lat'].where(mask).values
            mask_lons = ds['lon'].where(mask).values

            ax.scatter(mask_lons, mask_lats, c='grey', s=100, marker='o',
                          alpha=0.75, transform=ccrs.PlateCarree())
            ax.coastlines()  # type: ignore
            gridlines = ax.gridlines(draw_labels=True)  # type: ignore
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.bottom_labels = True
            gridlines.left_labels = True
            plt.xlim(round(row['lon']-window_size), round(row['lon']+window_size))
            plt.ylim(round(row['lat']-window_size), round(row['lat']+window_size))
            plt.plot(row['lon'], row['lat'], 'x', markersize=15,
             color="black", transform=ccrs.PlateCarree())
            plt.tight_layout()

            average_time, year, month, day = get_mean_info(ds)
            plt.title(f'{year}-{month}-{day} {average_time.hour} UTC')
            title = os.path.basename(file_path)
            folder_path = os.path.join(PATH_SAVE_PLOTS, f"{title}.png")
            plt.savefig(folder_path, format="png")
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("radius", type=float)
    parser.add_argument("window_size", type=float)
    args = parser.parse_args()
    get_stats(args.radius, args.window_size)
