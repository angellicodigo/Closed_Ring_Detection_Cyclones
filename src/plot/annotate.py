import argparse
from tqdm import tqdm
import pandas as pd
import os
from config.utils import get_mean_info
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cartopy.crs as ccrs
import numpy as np

PATH_INFO = r'data\processed\annotations_SS copy.txt'
PATH_DATASET = r'data\processed\dataset'
PATH_SAVE_IMAGES = r'images\annotated'

def annotate(window_size: float) -> None:
    results = []

    def isTrue(event):
        results.append(1)
        plt.close()

    def isFalse(event):
        results.append(0)
        plt.close()

    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Annotate all cyclones in txt file"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            fig = plt.figure(figsize=(12, 8))
            fig.subplots_adjust(bottom=0.25)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
            V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))
            quiver = ax.quiver(ds['lon'], ds['lat'], U, V, ds['wind_speed'],
                               cmap='turbo', transform=ccrs.PlateCarree(), scale=500, pivot='mid')
            cbar = plt.colorbar(quiver)
            cbar.set_label("Wind Speed")
            ax.coastlines()  # type: ignore
            gridlines = ax.gridlines(draw_labels=True)  # type: ignore
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.bottom_labels = True
            gridlines.left_labels = True
            plt.tight_layout()
            plt.xlim(round(row['lon']-window_size),
                     round(row['lon']+window_size))
            plt.ylim(round(row['lat']-window_size),
                     round(row['lat']+window_size))
            plt.plot(row['lon'], row['lat'], 'x', markersize=12,
                     color="black", transform=ccrs.PlateCarree())
            maximum_wind = ds['wind_speed'].max()
            sel = ds.where(ds['wind_speed'] == maximum_wind, drop=True)
            plt.plot(sel['lon'], sel['lat'], 'x', markersize=12,
                     color="purple", transform=ccrs.PlateCarree())
            area1 = fig.add_axes([0.20, 0.05, 0.25, 0.08]) # type: ignore
            area2 = fig.add_axes([0.55, 0.05, 0.25, 0.08]) # type: ignore

            button_true = Button(area1, 'Is a Cloesd Ring', color='green', hovercolor='lightgreen')
            button_false = Button(area2, 'Not a Closed Ring', color='red', hovercolor='lightpink')

            button_true.on_clicked(isTrue)
            button_false.on_clicked(isFalse)
            average_time, year, month, day = get_mean_info(ds)
            fig.suptitle(f'{year}-{month}-{day} {average_time.hour} UTC')
            title = os.path.basename(file_path)
            folder_path = os.path.join(PATH_SAVE_IMAGES, f"{title}.png")
            plt.savefig(folder_path, format="png")
            plt.show()
    
    print(f"How many true labels? {results.count(1)}")
    print(f"How many false labels? {results.count(0)}")

    df['annotated'] = results
    save_path = os.path.join(PATH_DATASET, f'new_{os.path.basename(PATH_INFO)}')
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=float, default=3)
    args = parser.parse_args()
    annotate(args.window_size)
