import argparse
from tqdm import tqdm
import pandas as pd
import os
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import BoundaryNorm
from matplotlib.axes import Axes
import cartopy.crs as ccrs
import numpy as np
from config.utils import get_segmentation_map

# Path to the folder where the ASCAT files are stored
PATH_DATASET = r'data\processed\dataset'

MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]


def view(path_info: str, window_size: float, radius: float) -> None:
    """
    Args:
        path_info: path to the variant of annotations.txt
        window_size: how much you want to see the data in the GUI

    Returns:
        Loads all the data in the txt file given by path_info and displays it in a GUI where there are a back and next buttons.

    """
    df = pd.read_csv(path_info, sep=r'\t', engine='python')
    data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading each file into memory first"):
        cyclone_id = row['cyclone_id']
        if (cyclone_id not in MEDICANES) and (row['label'] == 0):
            file_path = os.path.join(PATH_DATASET, row['file_name'])
            ds = xr.open_dataset(file_path)
            data.append((row, ds))

    print(f'How many files? {len(data)}')

    index = [0]  # A list with 0 because list is global but integer is not
    cbar_prev = [None]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.05, 0.15, 0.9, 0.8],  # type: ignore
                      projection=ccrs.PlateCarree())

    area1 = plt.axes([0.1, 0.02, 0.15, 0.07])  # type: ignore
    area2 = plt.axes([0.3, 0.02, 0.15, 0.07])  # type: ignore

    button_back = Button(area1, 'Back',
                         color='gray', hovercolor='lightgray')
    button_next = Button(area2, 'Next',
                         color='green', hovercolor='lightgreen')

    def update_plot():
        ax.clear()
        boundaries = np.arange(0, 32.6, 2.5)
        cmap = plt.get_cmap("turbo")
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)
        row, ds = data[index[0]]
        U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
        V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))
        quiver = ax.quiver(ds['lon'], ds['lat'], U, V, ds['wind_speed'],
                           cmap=cmap, transform=ccrs.PlateCarree(), scale=500, pivot='mid', norm=norm)
        # This code is needed because I was having issues of the colorbar not clearing as it should be
        if cbar_prev[0] is not None:
            cbar_prev[0].update_normal(quiver)
        else:
            cbar_prev[0] = fig.colorbar(quiver, ax=ax)  # type: ignore
        cbar_prev[0].set_label("Wind Speed (m/s)")
        cbar_prev[0].set_ticks(boundaries)  # type: ignore

        ax.coastlines()  # type: ignore
        gridlines = ax.gridlines(draw_labels=True)  # type: ignore
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.bottom_labels = True
        gridlines.left_labels = True
        ax.set_xlim(round(row['lon'] - window_size),
                    round(row['lon'] + window_size))
        ax.set_ylim(round(row['lat'] - window_size),
                    round(row['lat'] + window_size))
        ax.plot(row['lon'], row['lat'], 'x', markersize=10,
                color="black", transform=ccrs.PlateCarree())
        plot_top_five(ax, ds)
        plot_semantic_segmentation(
            ax, ds, row['lat'], row['lon'], radius, 'black')
        fig.suptitle(f"{row['file_name']}")
        fig.canvas.draw()

    def back(event):
        if index[0] > 0:
            index[0] -= 1
            update_plot()

    def next(event):
        if index[0] < len(data) - 1:
            index[0] += 1
            update_plot()

    button_back.on_clicked(back)
    button_next.on_clicked(next)
    update_plot()
    plt.show()


def plot_top_five(ax: Axes, ds: xr.Dataset) -> None:
    wind_speed = ds['wind_speed'].values.flatten()
    wind_speed = np.nan_to_num(wind_speed, nan=-np.inf)
    lon = ds['lon'].values.flatten()
    lat = ds['lat'].values.flatten()
    points = np.argsort(wind_speed)[::-1]
    indices = points[:5]
    lon = lon[indices]
    lat = lat[indices]
    ax.scatter(lon, lat, s=100, marker='x', color="purple",
               transform=ccrs.PlateCarree())


def plot_semantic_segmentation(ax: Axes, ds: xr.Dataset, query_lat: float, query_lon: float, radius: float, color: str) -> None:
    mask = get_segmentation_map(ds, query_lat, query_lon, radius)
    mask_lats = ds['lat'].where(mask).values
    mask_lons = ds['lon'].where(mask).values

    ax.scatter(mask_lons, mask_lats, c=color, s=1, marker='o',
               alpha=0.25, transform=ccrs.PlateCarree())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_info")
    parser.add_argument("--window_size", type=float, default=3.5)
    parser.add_argument("--radius", type=float, default=100)
    args = parser.parse_args()
    view(args.path_info, args.window_size, args.radius)
