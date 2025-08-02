from pathlib import Path
import argparse
import xarray as xr
import pandas as pd
from tqdm import tqdm
import os
from utils import get_boundary_box, get_mean_info, coords_to_pixels
from train_OD import init_model, z_score_norm, collate_fn
import torch
from torch.utils.data import DataLoader
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
import numpy as np
from Dataset.dataset import CycloneDatasetOD
from typing import Optional, List, dict
import cv2

PATH_SAVE_IMAGES = r'images\output'
PATH_INFO = r'medicanes_info\annotations.txt'
PATH_DATASET = r'dataset'


def generate_dataset() -> DataLoader:
    dataset = CycloneDatasetOD(PATH_INFO, PATH_DATASET, transform=z_score_norm)
    return DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


def plot(model_path: Path, window_size: float, radius: float) -> None:
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = init_model()
    model.load_state_dict(torch.load(
        model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    with torch.no_grad():
        DataLoader = generate_dataset()
        for index, (datas, _) in tqdm(enumerate(DataLoader, start=0), total=len(DataLoader), desc="Plotting all predictions"):
            # for index, (datas, _) in enumerate(DataLoader, start=0):
            datas = [d.to(device) for d in datas]
            # Returns List[dict[Tensor]] with len(List) = 1
            output = model(datas)
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(
                21, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            row = df.iloc[index]
            file_path = os.path.join(PATH_DATASET, row['file_name'])
            with xr.open_dataset(file_path) as ds:
                plot_subplot(ax1, ds, row['lat'],
                             row['lon'], radius, window_size)
                plot_subplot(ax2, ds, row['lat'], row['lon'],
                             radius, window_size, output, device)
                plt.tight_layout()
                average_time, year, month, day = get_mean_info(ds)
                fig.suptitle(f'{year}-{month}-{day} {average_time.hour} UTC')
                title = os.path.basename(file_path)
                folder_path = os.path.join(PATH_SAVE_IMAGES, f"{title}.png")
                plt.savefig(folder_path, format="png")
                plt.close()


def plot_subplot(ax: Axes, ds: xr.Dataset, center_lat: float, center_lon: float, radius: float, window_size: float,    output: Optional[List[dict[str, torch.Tensor]]] = None, device: Optional[torch.device] = None):
    title = ''
    if output != None:
        title = 'Prediction from the model'

        canvas = np.ones(
            (ds['lon'].shape[0], ds['lon'].shape[1], 3), dtype=np.uint8) * 255
        highest_score = output[0]['scores'].argmax()
        boxes = output[0]['boxes'][highest_score].detach().to(device)
        x1 = int(boxes[0].item())
        y1 = int(boxes[1].item())
        x2 = int(boxes[2].item())
        y2 = int(boxes[3].item())

        width = x2 - x1
        height = y2 - y1
        im = cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(
            255, 0, 0), thickness=1)  # type: ignore
        window_size += 10
        ax.imshow(im, extent=(center_lon - window_size, center_lon + window_size,
                              center_lat - window_size, center_lat + window_size))

    if output == None:
        title = 'True answer with given pixels in red and green'

        U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
        V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))
        quiver = ax.quiver(ds['lon'], ds['lat'], U, V, ds['wind_speed'],
                           cmap='turbo', transform=ccrs.PlateCarree(), scale=500, pivot='mid')
        cbar = plt.colorbar(quiver)
        cbar.set_label("Wind Speed")

        min_lat, min_lon, max_lat, max_lon = get_boundary_box(
            ds, center_lat, center_lon, radius)
        width = max_lon - min_lon
        height = max_lat - min_lat
        bbox = patches.Rectangle(
            (min_lon, min_lat), width, height, linewidth=2, edgecolor='black', facecolor='none')

        top_left = (min_lon, max_lat)
        bot_right = (max_lon, min_lat)

        row_top, col_top = coords_to_pixels(
            ds, top_left[1], top_left[0], center_lat, center_lon)
        pixel_top_left = np.full(
            (ds['lat'].shape[0], ds['lat'].shape[1]), np.nan)
        pixel_top_left[row_top, col_top] = 1

        row_bot, col_bot = coords_to_pixels(
            ds, bot_right[1], bot_right[0], center_lat, center_lon)
        pixel_bot_right = np.full(
            (ds['lat'].shape[0], ds['lat'].shape[1]), np.nan)
        pixel_bot_right[row_bot, col_bot] = 1

        ax.imshow(pixel_top_left, extent=(center_lon - window_size, center_lon + window_size,
                                          center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['red']))
        ax.imshow(pixel_bot_right, extent=(center_lon - window_size, center_lon + window_size,
                                           center_lat - window_size, center_lat + window_size), cmap=ListedColormap(['green']))
        ax.add_patch(bbox)

    ax.coastlines()  # type: ignore
    ax.set_title(title)
    ax.set_aspect('equal')
    gridlines = ax.gridlines(draw_labels=True)  # type: ignore
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.bottom_labels = True
    gridlines.left_labels = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("window_size", type=float)
    parser.add_argument("--radius", type=float, default=100)
    args = parser.parse_args()
    plot(args.model_path, args.window_size, args.radius)
