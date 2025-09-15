import argparse
from tqdm import tqdm
import pandas as pd
import os
from config.utils import get_num_points, get_num_points_over_ocean, calc_percent_over_ocean, mean_std_wind_dir, mean_std_wind_speed
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from typing import List
from scipy.stats import gaussian_kde
from scipy.stats import skew

# Path where ASCAT files are accessed
PATH_DATASET = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\dataset'
# Path where images are saved
PATH_SAVE = r'images\figures'


def get_stats(path_info: str, radius: float) -> None:
    """
    Args: 
        path_info: String of the path to any version of annotations.txt
        radius: distance km away from the center written in the txt with path 'path_info'

    Returns:
        Prints out the number of files, the range of years the files cover, the number of unique cyclone_id, what filters are
        needed to reduce the amount of skewness in the dataset. Optionally, the user can print out the desired plot
        (box plot, scatter plot, histogram, and density plot).

    """
    columns = ['year', 'cyclone_id', 'num_of_points',
               'num_of_points_over_ocean', 'percent_over_ocean', 'mean_ws', 'std_ws', 'mean_wd', 'std_wd', 'label']
    info = pd.DataFrame(columns=columns)
    df = pd.read_csv(path_info, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting statistics of the dataset"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            mean_ws, std_ws = mean_std_wind_speed(
                ds, row['lat'], row['lon'], radius)
            mean_wd, std_wd = mean_std_wind_dir(
                ds, row['lat'], row['lon'], radius)
            info.loc[len(info)] = {'year': row['year'], 'cyclone_id': row['cyclone_id'], 'num_of_points': get_num_points(ds, row['lat'], row['lon'], radius), 'num_of_points_over_ocean': get_num_points_over_ocean(
                ds, row['lat'], row['lon'], radius), 'percent_over_ocean': calc_percent_over_ocean(ds, row['lat'], row['lon'], radius), 'mean_ws': mean_ws, 'std_ws': std_ws, 'mean_wd': mean_wd, 'std_wd': std_wd, 'label': row['label']}

    print(f'How many files? {len(info)}')
    print(f'What years? {info["year"].unique()}')
    print(f'How many unique cyclone_ids? {len(info["cyclone_id"].unique())}')
    print(info.head(26))

    minimize_skew('Number of points over ocean',
                  info['num_of_points_over_ocean'])
    minimize_skew('Percentage over ocean', info['percent_over_ocean'])

    scatter(info["num_of_points_over_ocean"], info['percent_over_ocean'], 'scatterplot.png', 'Number of points over the ocean',
            'Percentage of points over ocean over total number of points (including land)', 'Scatterplot of types of points within 100 km')
    boxplot(info["num_of_points_over_ocean"], 'boxplot.png', '',
            'Number of points over the ocean', 'Boxplot of points over ocean')
    boxplot(info["percent_over_ocean"], 'boxplot_percent.png',
            '', 'Percentages', 'Boxplot of percentages')

    scatter(info['mean_ws'], info['std_ws'], 'scatterplot_ws.png', '', '', '')
    scatter(info['mean_wd'], info['std_wd'], 'scatterplot_wd.png', '', '', '')
    scatter(info['std_wd'], info['mean_ws'],
            'scatterplot_ws_wd.png', '', '', '')


def minimize_skew(label: str, array: pd.Series) -> None:
    min_skew = skew(array)
    min_point = array.min()
    for _ in range(250):
        array = array.drop(array.idxmin())
        if abs(skew(array)) < abs(min_skew):
            min_skew = skew(array)
            min_point = array.min()

    print(f'{label}: {min_point}')
    print(f'{label}: {min_skew}')


def histogram(data: pd.Series, file_name: str, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(12, 10))
    plt.hist(data, edgecolor='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    folder_path = os.path.join(PATH_SAVE, file_name)
    plt.savefig(folder_path, format="png")
    plt.close()


def scatter(x: pd.Series, y: pd.Series, file_name: str, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(12, 10))
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    folder_path = os.path.join(PATH_SAVE, file_name)
    plt.savefig(folder_path, format="png")
    plt.close()


def boxplot(data: pd.Series, file_name: str, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(12, 10))
    bp = plt.boxplot(data)
    # If you want to print out which points are outliers
    # outlier_vals = bp['fliers'][0].get_ydata()
    # print("Outlier values:", outlier_vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    folder_path = os.path.join(PATH_SAVE, file_name)
    plt.savefig(folder_path, format="png")
    plt.close()


def densityplot(data: pd.Series, file_name: str, xlabel: str, ylabel: str, title: str) -> None:
    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 1000)
    y_grid = kde(x_grid)
    plt.figure(figsize=(12, 10))
    plt.plot(x_grid, y_grid)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    folder_path = os.path.join(PATH_SAVE, file_name)
    plt.savefig(folder_path, format='png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_info")
    parser.add_argument("--radius", type=float, default=100)
    args = parser.parse_args()
    get_stats(args.path_info, args.radius)
