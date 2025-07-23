import argparse
from tqdm import tqdm
import pandas as pd
import os
from utils import get_mean_info, get_boundary_box, get_bearing_indices, calc_bearing, dist_bwt_two_points, coords_to_pixels, get_distances_indices
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Tuple
from pyproj import Geod


PATH_INFO = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info\dataset_preprocessed.txt'
PATH_DATASET = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\dataset'
PATH_SAVE_PLOTS = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\images\angles_distances_10'


def get_stats_and_plot(radius: float, window_size: float, km: float, round_to: int) -> None:
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Plotting angles and distances"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            min_lat, min_lon, max_lat, max_lon = get_boundary_box(
                ds, row['lat'], row['lon'], radius)
            top_left = (min_lon, max_lat)
            bot_right = (max_lon, min_lat)
            angles = get_bearing_indices(
                ds, row['lat'], row['lon'])
            distances = get_distances_indices(
                ds, row['lat'], row['lon'])

            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(
                21, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            plot(fig, ax1, ds, angles,  # type: ignore
                 row['lat'], row['lon'], window_size, top_left, bot_right, True, km, round_to)
            plot(fig, ax2, ds, distances,  # type: ignore
                 row['lat'], row['lon'], window_size, top_left, bot_right, False, km, round_to)
            plot_four_points(ax1, radius, row['lat'], row['lon'], True)
            plot_four_points(ax2, radius, row['lat'], row['lon'], False)
            average_time, year, month, day = get_mean_info(ds)
            fig.suptitle(f'{year}-{month}-{day} {average_time.hour} UTC')
            title = os.path.basename(file_path)
            folder_path = os.path.join(PATH_SAVE_PLOTS, f"{title}.png")
            plt.tight_layout()
            plt.savefig(folder_path, format="png")
            plt.close()


def plot(fig: Figure, ax: Axes, ds: xr.Dataset, data: np.ndarray, center_lat: float, center_lon: float, window_size: float, top_left: Tuple[float, float], bot_right: Tuple[float, float], isAngle: bool, km: float, round_to: int) -> None:

    if isAngle:
        cmap = 'twilight'
        title = f'Bearing (degrees from N) with degrees rounded to {round_to}'
        vmax = 360
        top_text = str(
            round(calc_bearing(top_left[1], top_left[0], center_lat, center_lon), 2))
        bot_text = str(
            round(calc_bearing(bot_right[1], bot_right[0], center_lat, center_lon), 2))
    else:
        cmap = 'viridis'
        title = f'Distance (km) with contour lines every {km} km'
        vmax = None
        top_text = str(round(dist_bwt_two_points(
            top_left[1], top_left[0], center_lat, center_lon), 2))  # type: ignore
        bot_text = str(round(dist_bwt_two_points(
            bot_right[1], bot_right[0], center_lat, center_lon), 2))  # type: ignore

    im = ax.imshow(data, extent=(center_lon - window_size, center_lon + window_size,
                   center_lat - window_size, center_lat + window_size), cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax)

    row_top, col_top = coords_to_pixels(
        ds, top_left[1], top_left[0], center_lat, center_lon)
    pixel_top_left = np.full(data.shape, np.nan)
    pixel_top_left[row_top, col_top] = 1

    row_bot, col_bot = coords_to_pixels(
        ds, bot_right[1], bot_right[0], center_lat, center_lon)
    pixel_bot_right = np.full(data.shape, np.nan)
    pixel_bot_right[row_bot, col_bot] = 1

    im = ax.imshow(pixel_top_left, extent=(center_lon - window_size, center_lon + window_size,
                                           center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['white']))
    im = ax.imshow(pixel_bot_right, extent=(center_lon - window_size, center_lon + window_size,
                                            center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['grey']))

    if isAngle:
        mask = np.round(data, round_to) == round(calc_bearing(
            top_left[1], top_left[0], center_lat, center_lon), round_to)
        top_left_data = np.where(mask, 1, np.nan)
        mask = np.round(data, round_to) == round(calc_bearing(
            bot_right[1], bot_right[0], center_lat, center_lon), round_to)
        bot_right_data = np.where(mask, 1, np.nan)

        im = ax.imshow(top_left_data, extent=(center_lon - window_size, center_lon + window_size,
                       center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['red']), vmin=0, vmax=360)
        im = ax.imshow(bot_right_data, extent=(center_lon - window_size, center_lon + window_size,
                       center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['green']), vmin=0, vmax=360)

        mask = np.round(data, round_to) == np.round(calc_bearing(
            top_left[1], top_left[0], center_lat, center_lon), round_to)
        top_left_data = np.where(mask, 1, np.nan)
        mask = np.round(data, round_to) == np.round(calc_bearing(
            bot_right[1], bot_right[0], center_lat, center_lon), round_to)
        bot_right_data = np.where(mask, 1, np.nan)

        im = ax.imshow(top_left_data, extent=(center_lon - window_size, center_lon + window_size,
                       center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['red']))
        im = ax.imshow(bot_right_data, extent=(center_lon - window_size, center_lon + window_size,
                       center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['green']))

    if not isAngle:
        max_dist = np.max(data)
        levels = np.arange(0, max_dist + km, km)
        contours = ax.contour(data, colors='white', levels=levels, linestyles='--', extent=(center_lon - window_size, center_lon + window_size,
                                                                                            center_lat - window_size, center_lat + window_size))
        ax.clabel(contours, fmt='%d km', fontsize=10)

    ax.coastlines()  # type: ignore
    ax.plot(center_lon, center_lat, 'x', markersize=15,
            color="blue", transform=ccrs.PlateCarree())
    ax.plot(top_left[0], top_left[1], 'x', markersize=15,
            color="red", transform=ccrs.PlateCarree())
    ax.annotate(top_text, xy=(top_left[0], top_left[1]), xytext=(
        0, 10), textcoords='offset pixels', ha='center', va='bottom')
    ax.plot(bot_right[0], bot_right[1], 'x', markersize=15,
            color="green", transform=ccrs.PlateCarree())
    ax.annotate(bot_text, xy=(bot_right[0], bot_right[1]), xytext=(
        0, 10), textcoords='offset pixels', ha='center', va='bottom')

    gridlines = ax.gridlines(draw_labels=True)  # type: ignore
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.bottom_labels = True
    gridlines.left_labels = True
    ax.set_title(title)
    ax.set_aspect('equal')


def plot_four_points(ax: Axes, radius: float, center_lat: float, center_lon: float, isAngle: bool) -> None:
    radius = radius * 1000
    g = Geod(ellps="WGS84")
    lonN, latN, _ = g.fwd(center_lon, center_lat, 0,   radius)
    lonE, latE, _ = g.fwd(center_lon, center_lat, 90,  radius)
    lonS, latS, _ = g.fwd(center_lon, center_lat, 180, radius)
    lonW, latW, _ = g.fwd(center_lon, center_lat, 270, radius)
    if isAngle:
        top_text = str(
            round(calc_bearing(latN, lonN, center_lat, center_lon), 2))
        left_text = str(
            round(calc_bearing(latW, lonW, center_lat, center_lon), 2))
        right_text = str(
            round(calc_bearing(latE, lonE, center_lat, center_lon), 2))
        bot_text = str(
            round(calc_bearing(latS, lonS, center_lat, center_lon), 2))
    else:
        top_text = str(round(dist_bwt_two_points(
            latN, lonN, center_lat, center_lon), 2))  # type: ignore
        left_text = str(round(dist_bwt_two_points(
            latW, lonW, center_lat, center_lon), 2))  # type: ignore
        right_text = str(round(dist_bwt_two_points(
            latE, lonE, center_lat, center_lon), 2))  # type: ignore
        bot_text = str(round(dist_bwt_two_points(
            latS, lonS, center_lat, center_lon), 2))  # type: ignore

    ax.plot(lonN, latN, 'x', markersize=15,
            color="yellow", transform=ccrs.PlateCarree())
    ax.annotate(top_text, xy=(lonN, latN), xytext=(
        0, 10), textcoords='offset pixels', ha='center', va='bottom')

    ax.plot(lonW, latW, 'x', markersize=15,
            color="yellow", transform=ccrs.PlateCarree())
    ax.annotate(left_text, xy=(lonW, latW), xytext=(
        0, 10), textcoords='offset pixels', ha='center', va='bottom')

    ax.plot(lonE, latE, 'x', markersize=15,
            color="yellow", transform=ccrs.PlateCarree())
    ax.annotate(right_text, xy=(lonE, latE), xytext=(
        0, 10), textcoords='offset pixels', ha='center', va='bottom')

    ax.plot(lonS, latS, 'x', markersize=15,
            color="yellow", transform=ccrs.PlateCarree())
    ax.annotate(bot_text, xy=(lonS, latS), xytext=(
        0, 10), textcoords='offset pixels', ha='center', va='bottom')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("radius", type=float)
    parser.add_argument("window_size", type=float)
    parser.add_argument("--km", type=float, default=10)
    parser.add_argument("--round_to", type=int, default=0)
    args = parser.parse_args()
    get_stats_and_plot(args.radius, args.window_size, args.km, args.round_to)
