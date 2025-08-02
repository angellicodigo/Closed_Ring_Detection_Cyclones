
from typing import Union
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
from pyproj import Geod

PATH_CENTERS = r'data\external\TRACKS_CL7.dat'


def get_center(cyclone_id: int, year: int, month: int, day: int, time: pd.Timestamp) -> tuple[float, float]:
    columns = ['cyclone_id', 'lon', 'lat',
               'year', 'month', 'day', 'hour', 'MSLP']
    centers = pd.read_csv(PATH_CENTERS, sep=r'\s+', names=columns)
    round_hour = time.round('h').hour
    row = centers.loc[
        (centers['cyclone_id'] == np.int64(cyclone_id)) &
        (centers['year'] == np.int64(year)) &
        (centers['month'] == np.int64(month)) &
        (centers['day'] == np.int64(day)) &
        (centers['hour'] == np.int64(round_hour))
    ]

    if not row.empty:
        return row['lat'].values[0], row['lon'].values[0]

    return -np.inf, -np.inf


def nearest_neighbors(ds: xr.Dataset, query_lat: float, query_lon: float) -> xr.Dataset:
    row_indices, col_indices = nearest_neighbors_indices(
        ds, query_lat, query_lon)
    dim = list(ds['lon'].sizes)
    row_dim = dim[0]
    col_dim = dim[1]
    points = []
    for i in range(len(row_indices)):
        points.append(
            ds.isel({row_dim: row_indices[i], col_dim: col_indices[i]}))

    return xr.concat(points, dim='neighbors')


def dist_bwt_two_points(lat1: float, lon1: float, lat2:  Union[float, np.ndarray], lon2:  Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    R = 6371  # Earth radius in km
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def haversine(ds: xr.Dataset, query_lon: float, query_lat: float) -> np.ndarray:
    mask = ds['lon'].notnull().values  # Both lon and lat can use the same mask
    lons = np.deg2rad(ds['lon'].values[mask])
    lats = np.deg2rad(ds['lat'].values[mask])
    points = np.stack([lats, lons], axis=1)

    query_point = np.deg2rad([[query_lat, query_lon]])

    return haversine_distances(points, query_point).ravel()


def nearest_neighbors_indices(ds: xr.Dataset, query_lat: float, query_lon: float) -> tuple[np.ndarray, np.ndarray]:
    distances = haversine(ds, query_lon, query_lat)
    sorted_indices = np.argsort(distances)
    mask = ds['lon'].notnull().values
    valid_indices = np.where(mask)
    original_index = (valid_indices[0][sorted_indices],
                      valid_indices[1][sorted_indices])
    return original_index


def get_mean_info(ds: xr.Dataset) -> tuple[pd.Timestamp, int, int, int]:
    average_time = pd.to_datetime(ds.time.mean().values)
    year = int(average_time.year)
    month = int(average_time.month)
    day = int(average_time.day)

    return average_time, year, month, day


def get_boundary_box(query_lat: float, query_lon: float, radius: float) -> tuple[float, float, float, float]:
    radius = radius * 1000
    g = Geod(ellps="WGS84")
    _, latN, _ = g.fwd(query_lon, query_lat, 0,   radius)
    lonE, _, _ = g.fwd(query_lon, query_lat, 90,  radius)
    _, latS, _ = g.fwd(query_lon, query_lat, 180, radius)
    lonW, _, _ = g.fwd(query_lon, query_lat, 270, radius)

    min_lat = latS
    min_lon = lonW
    max_lat = latN
    max_lon = lonE

    return min_lat, min_lon, max_lat, max_lon


def get_num_points(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float, isBBox: bool) -> float:
    if isBBox:
        min_lon, max_lon, min_lat, max_lat = get_boundary_box(
            query_lat, query_lon, radius)
        mask = (min_lon <= ds.lon) & (ds.lon <= max_lon) & (
            min_lat <= ds.lat) & (ds.lat <= max_lat)
        return len(ds['wind_speed'].values[mask])
    else:
        lats = ds['lat'].values
        lons = ds['lon'].values
        distances = dist_bwt_two_points(query_lat, query_lon, lats, lons)
        distance_mask = distances <= radius  # Not 1D
        return np.count_nonzero(distance_mask)


def calc_percent_valid(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float, isBBox: bool) -> float:
    if isBBox:
        min_lon, max_lon, min_lat, max_lat = get_boundary_box(
            query_lat, query_lon, radius)
        mask = (min_lon <= ds.lon) & (ds.lon <= max_lon) & (
            min_lat <= ds.lat) & (ds.lat <= max_lat)
        wind_speed = ds['wind_speed'].values[mask]
        non_nan = ~np.isnan(wind_speed)

        return (np.count_nonzero(non_nan) / len(wind_speed)) * 100
    else:
        mask = get_segmentation_map(ds, query_lat, query_lon, radius)
        return (np.count_nonzero(mask) / get_num_points(ds, query_lat, query_lon, radius, False)) * 100


def get_num_of_points_ocean(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float, isBBox: bool):
    return (calc_percent_valid(ds, query_lat, query_lon, radius, isBBox) / 100) * get_num_points(ds, query_lat, query_lon, radius, isBBox)


def get_segmentation_map(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> xr.DataArray:
    """
        Returns an xarray that includes 1 (Object) and 0 (Background)
    """
    lats = ds['lat'].values
    lons = ds['lon'].values
    distances = dist_bwt_two_points(query_lat, query_lon, lats, lons)

    distance_mask = distances <= radius  # Not 1D
    mask = ~np.isnan(ds['wind_speed'].values)  # Also not 1D
    combined_mask = np.logical_and(distance_mask, mask)

    return xr.DataArray(combined_mask, dims=tuple(ds['lon'].sizes), coords=ds['lon'].coords)
