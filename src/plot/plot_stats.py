import argparse
from tqdm import tqdm
import pandas as pd
import os
from config.utils import get_boundary_box, calc_percent_valid, dist_bwt_two_points, get_num_points
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from typing import List
from scipy.stats import gaussian_kde
from scipy.stats import skew

PATH_INFO = r'data\processed\annotations_SS.txt'
PATH_DATASET = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\dataset'
PATH_SAVE = r'images\figures'


def get_stats(radius: float, isBBox: bool) -> None:
    columns = ['percent', 'lat', 'lon', 'label',
               'width(km)', 'height(km)', 'num_of_points']
    info = pd.DataFrame(columns=columns)
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting stats of the dataset"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            percentage = 0
            dist_height = 0
            dist_width = 0
            num = 0
            if isBBox:
                mask = ds['wvc_index'].notnull()
                ds = ds.where(mask, drop=True)
                min_lat, min_lon, max_lat, max_lon = get_boundary_box(
                    row['lat'], row['lon'], radius)
                dist_width = dist_bwt_two_points(
                    min_lat, min_lon, min_lat, max_lon)
                dist_height = dist_bwt_two_points(
                    min_lat, min_lon, max_lat, min_lon)
                num = get_num_points(ds, row['lat'], row['lon'], radius, False)
                if num != 0:
                    percentage = calc_percent_valid(
                        ds, row['lat'], row['lon'], radius, True)
            else:
                num = get_num_points(ds, row['lat'], row['lon'], radius, False)
                if num != 0:
                    percentage = calc_percent_valid(
                        ds, row['lat'], row['lon'], radius, False)

            info.loc[len(info)] = {'percent': percentage, 'lat': row['lat'], 'lon': row['lon'], 'label': row['label'],  # type: ignore
                                   'width(km)': dist_width, 'height(km)': dist_height, 'num_of_points': num}

    title = os.path.basename(PATH_INFO)
    percent_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                    50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    percent_counts = np.arange(0, 500, 25)
    num_bins = np.arange(0, 250, 25)
    num_counts = np.arange(0, 550, 25)
    histogram(info['percent'], f'percentages_{title}.png', 'Percent of points in boundary box over ocean (%)',
              'Percentage of wind-speed points over ocean', 'Percentage of wind-speed points over ocean', bins=percent_bins, counts=percent_counts)  # type: ignore
    histogram(info['num_of_points'], f'num_of_points_{title}.png', 'Number of points',
              'Frequency', 'Number of points within 100 km radius', bins=num_bins, counts=num_counts)  # type: ignore
    histogram(info["percent"] / 100 * info["num_of_points"], f'num_of_points_on_ocean_{title}.png', 'Number of points',
              'Frequency', 'Number of points within 100 km radius over the ocean', bins=num_bins, counts=num_counts)  # type: ignore
    scatter(info["percent"] / 100 * info["num_of_points"], info['percent'],
            f'scatterplot_{title}.png', 'Number of points over the ocean', 'Percentage', 'Scatterplot of points vs percentage within 100 km')
    scatter(info["percent"] / 100 * info["num_of_points"], info['num_of_points'],
            f'scatterplot_points_{title}.png', 'Number of points over the ocean', 'Total number of points', 'Scatterplot of types of points within 100 km')
    boxplot(info["percent"] / 100 * info["num_of_points"],
            f'boxplot_{title}.png', '1', 'Number of points over the ocean', 'Boxplot of points over ocean')
    boxplot(info["percent"],
            f'boxplot_percent_{title}.png', '1', 'Percentages', 'Boxplot of percentages')
    densityplot(info["percent"] / 100 * info["num_of_points"],
                f'densityplot_{title}.png', 'Number of points over the ocean', 'Probability Density Estimation', 'Density plot of points over ocean')

    point = info["percent"] / 100 * info["num_of_points"]
    min_skew = skew(point)
    print(min_skew)
    min_point = point.min()
    for i in range(500):
        point = point.drop(point.idxmin())
        if abs(skew(point)) < abs(min_skew):
            min_skew = skew(point)
            min_point = point.min()
    
    print(min_point)    
    print(min_skew)

    point = info["percent"]
    min_skew = skew(point)
    print(min_skew)
    min_point = point.min()
    for i in range(500):
        point = point.drop(point.idxmin())
        if abs(skew(point)) < abs(min_skew):
            min_skew = skew(point)
            min_point = point.min()
    
    print(min_point)    
    print(min_skew)

def histogram(data: pd.Series, file_name: str, xlabel: str, ylabel: str, title: str, bins: Optional[List[float]] = None, counts: Optional[List[float]] = None) -> None:
    plt.figure(figsize=(12, 10))
    if bins is not None:
        plt.hist(data, bins=bins, edgecolor='k')
        plt.xticks(bins)
    else:
        plt.hist(data, edgecolor='k')

    if counts is not None:
        plt.yticks(counts)

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
    parser.add_argument("--radius", type=float, default=100)
    parser.add_argument("--isBBox", type=float, default=False)
    args = parser.parse_args()
    get_stats(args.radius, args.isBBox)
