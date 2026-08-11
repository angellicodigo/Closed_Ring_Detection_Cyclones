import pandas as pd
import numpy as np
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
from pyproj import Geod
from scipy.stats import circmean, circstd
from typing import Tuple
import os
import dotenv

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

def get_center(cyclone_id: int, year: int, month: int, day: int, time: pd.Timestamp) -> tuple[float, float]:
    """Returns the center based on what is written in TRACKS_CL7.dat. The idea is to get the center
    of the ASCAT file in raw/Tracks

    Args:
        cyclone_id: id of the cyclone like 1702
        year: year
        month: month
        day: day 
        time: is a pd.Timestamp to easily round to the nearest whole hour

    Returns: 
        Latitude and longitude of the center, else returns a tuple of negative infinities if such center
        cannot be found

    """
    columns = ['cyclone_id', 'lon', 'lat',
               'year', 'month', 'day', 'hour', 'MSLP']
    centers = pd.read_csv(os.getenv("CENTER_CYCLONES_PATH"), sep=r'\s+', names=columns)
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
    """Given any point (latitude and longitude) and xarray dataset from the ASCAT file, get the nearest points to the center

    Args:
        ds: xarray dataset from the ASCAT file
        query_lat: latitude of any given point in ds
        query_lon: longitude of any given point in ds

    Returns:
        A linear xarray dataset with length N, where N is the number of points in ds. The first index corresponds to the
        nearest neighbor to the query point.

    """
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


def dist_bwt_two_points(lat1: float, lon1: float, lat2:  float | np.ndarray, lon2:  float | np.ndarray) -> float | np.ndarray:
    """Given two points (or a point and numpy of points), calculate the haversine distance between them

    Args: 
        lat1: latitude of the first point
        lon1: longitude of the first point
        lat2: latitude of the second point or a set of points
        lon2: longitude of the second point or a set of points

    Returns:
        Either a single float of distance in km, or a numpy array with same size as lat2 and lon2 of distances in km

    """
    R = 6371  # Earth radius in km
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def haversine(ds: xr.Dataset, query_lat: float, query_lon: float) -> np.ndarray:
    """
    Args:
        ds: xarray dataset in ASCAT files
        query_lat: latitude of the query point in ds
        query_lon: longitude of the query point in ds

    Returns:
        Returns a numpy array that calculates the haversine distance from the query point to each point in ds

    """

    mask = ds['lon'].notnull().values  # Both lon and lat can use the same mask
    lons = np.deg2rad(ds['lon'].values[mask])
    lats = np.deg2rad(ds['lat'].values[mask])
    points = np.stack([lats, lons], axis=1)

    query_point = np.deg2rad([[query_lat, query_lon]])

    return haversine_distances(points, query_point).ravel()


def nearest_neighbors_indices(ds: xr.Dataset, query_lat: float, query_lon: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        ds: xarray dataset in ASCAT file
        query_lat: latitude of point in ds
        query_lon: longitude of point in ds

    Returns:
        Similar to nearest_neighbors(), but instead of returning the point, returns the indices of the nearest points
        in ds.

    """
    distances = haversine(ds, query_lat, query_lon)
    sorted_indices = np.argsort(distances)
    mask = ds['lon'].notnull().values
    valid_indices = np.where(mask)
    original_index = (valid_indices[0][sorted_indices],
                      valid_indices[1][sorted_indices])
    return original_index


def get_mean_info(ds: xr.Dataset) -> tuple[pd.Timestamp, int, int, int]:
    """
    Args:
        ds: xarray dataset of the ASCAT file 

    Returns:
        Average time each point in ds is recorded, the average year, average month, and average day.

    """
    average_time = pd.to_datetime(ds.time.mean().values)
    year = int(average_time.year)
    month = int(average_time.month)
    day = int(average_time.day)

    return average_time, year, month, day

# TODO: May remove because no longer doing object detection
def get_boundary_box(query_lat: float, query_lon: float, radius: float) -> tuple[float, float, float, float]: 
    """Get the boundary box around the latitude and longitude point at a distance radius (km) away

    Args:
        query_lat: latitude of a point
        query_lon: longitude of a point
        radius: distance km away from the point

    Returns:
        Returns the bottom left point and top right point of the boundary box.

    """

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


def get_segmentation_map(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> xr.DataArray:
    """Returns a segmentation map of the ASCAT file

    Args:
        ds: xarray dataset of the ASCAT file
        query_lat: latitude of a point in ds
        query_lon: longitude of a point in ds

    Returns:
        Returns an xarray with the same size as ds (assuming 2D) where 1 is where a pixel is within radius distance away
        from the query point, and 0 else where.

    """
    lats = ds['lat'].values
    lons = ds['lon'].values
    distances = dist_bwt_two_points(query_lat, query_lon, lats, lons)

    distance_mask = distances <= radius  # Distances_mask is not 1D
    mask = ~np.isnan(ds['wind_speed'].values)  # Also mask not 1D
    combined_mask = np.logical_and(distance_mask, mask)

    return xr.DataArray(combined_mask, dims=tuple(ds['lon'].sizes), coords=ds['lon'].coords)


def get_num_points(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> float:
    """
    Args:
        ds: xarray dataset in the ASCAT file
        query_lat: latitude point in ds
        query_lon: longitude point in ds
        radius: distance km away from query point

    Returns:
        Calculates the number of points within radius km away from the query point in ds

    """
    lats = ds['lat'].values
    lons = ds['lon'].values
    distances = dist_bwt_two_points(query_lat, query_lon, lats, lons)
    distance_mask = distances <= radius  # distance_mask is not 1D
    return np.count_nonzero(distance_mask)


def get_num_points_over_ocean(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> float:
    """
    Args:
        ds: xarray dataset in the ASCAT file
        query_lat: latitude point in ds
        query_lon: longitude point in ds
        radius: distance km away from query point

    Returns:
        Number of points in ds that are over the ocean within radius km away from the query point 

    """
    lats = ds['lat'].values
    lons = ds['lon'].values
    distances = dist_bwt_two_points(query_lat, query_lon, lats, lons)

    distance_mask = distances <= radius  # Distances_mask is not 1D. A binary mask
    # Also mask not 1D. Also a binary mask
    mask = ~np.isnan(ds['wind_speed'].values)
    combined_mask = np.logical_and(distance_mask, mask)
    return combined_mask.sum()


def calc_percent_over_ocean(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> float:
    """
    Args:
        ds: xarray dataset in the ASCAT file
        query_lat: latitude point in ds
        query_lon: longitude point in ds
        radius: distance km away from query point

    Returns:
        The percentage of points over the ocean over the total number of points (including land) within radius km away
        from the query point in ds

    """
    return (get_num_points_over_ocean(ds, query_lat, query_lon, radius) / get_num_points(ds, query_lat, query_lon, radius)) * 100

# TODO: may delete the function because it's not being used
def mean_std_wind_dir(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> Tuple[float, float]:
    """
    Args:
        ds: xarray dataset in the ASCAT file
        query_lat: latitude point in ds
        query_lon: longitude point in ds
        radius: distance km away from query point

    Returns:
        Returns the mean and standard deviation of the wind direction of points radius km away of query point in ds

    """
    lats = ds['lat'].values
    lons = ds['lon'].values
    distances = dist_bwt_two_points(query_lat, query_lon, lats, lons)
    distance_mask = distances <= radius  # distance_mask is not 1D
    # Also mask not 1D. Also a binary mask
    mask = ~np.isnan(ds['wind_speed'].values)
    # A binary mask, where 1 is True and 0 is False
    combined_mask = np.logical_and(distance_mask, mask)

    wind_dir = ds['wind_dir'].values[combined_mask]

    return circmean(wind_dir, high=360, low=0), circstd(wind_dir, high=360, low=0)

# TODO: may remove because it's not being used
def mean_std_wind_speed(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> Tuple[float, float]:
    """
    Args:
        ds: xarray dataset in the ASCAT file
        query_lat: latitude point in ds
        query_lon: longitude point in ds
        radius: distance km away from query point

    Returns:
        Returns the mean and standard deviation of the wind speed of points radius km away of query point in ds

    """
    lats = ds['lat'].values
    lons = ds['lon'].values
    distances = dist_bwt_two_points(query_lat, query_lon, lats, lons)
    distance_mask = distances <= radius  # distance_mask is not 1D
    # Also mask not 1D. Also a binary mask
    mask = ~np.isnan(ds['wind_speed'].values)
    # A binary mask, where 1 is True and 0 is False
    combined_mask = np.logical_and(distance_mask, mask)

    wind_speed = ds['wind_speed'].values[combined_mask]

    return float(np.mean(wind_speed)), float(np.std(wind_speed))

if __name__ == "__main__":
    # 1. Create a dummy Synthetic ASCAT xarray Dataset (3x3 grid)
    lats = np.array([[36.0, 36.0, 36.0], [36.5, 36.5, 36.5], [37.0, 37.0, 37.0]])
    lons = np.array([[12.0, 12.5, 13.0], [12.0, 12.5, 13.0], [12.0, 12.5, 13.0]])
    
    # Top-right cell is NaN to mimic land
    wind_speed = np.array([[10.0, 12.0, 15.0], [8.0, 14.0, 11.0], [9.0, 13.0, np.nan]])
    wind_dir = np.array([[180.0, 190.0, 200.0], [175.0, 185.0, 195.0], [170.0, 180.0, np.nan]])
    times = np.full((3, 3), pd.Timestamp('2023-10-15T12:00:00'))

    test_ds = xr.Dataset(
        data_vars={
            'lat': (('x', 'y'), lats),
            'lon': (('x', 'y'), lons),
            'wind_speed': (('x', 'y'), wind_speed),
            'wind_dir': (('x', 'y'), wind_dir),
            'time': (('x', 'y'), times),
        }
    )

    query_lat, query_lon = 36.5, 12.5
    radius = 100.0  # km

    # Test get_center
    lat_c, lon_c = get_center(1702, 2023, 10, 15, pd.Timestamp('2023-10-15T12:00:00'))
    print(f"1. get_center(): lat={lat_c}, lon={lon_c}")

    # Test dist_bwt_two_points
    dist = dist_bwt_two_points(36.5, 12.5, 36.5, 13.0)
    print(f"2. dist_bwt_two_points(): {dist:.2f} km")

    # Test get_boundary_box
    bbox = get_boundary_box(query_lat, query_lon, radius)
    print(f"3. get_boundary_box(): {bbox}")

    # Test get_mean_info
    avg_t, y, m, d = get_mean_info(test_ds)
    print(f"4. get_mean_info(): Date={y}-{m:02d}-{d:02d}, Time={avg_t}")

    # Test nearest_neighbors
    nn_ds = nearest_neighbors(test_ds, query_lat, query_lon)
    print(f"5. nearest_neighbors(): Found {nn_ds.sizes['neighbors']} points")

    # Test ocean counts & percentages
    total_pts = get_num_points(test_ds, query_lat, query_lon, radius)
    ocean_pts = get_num_points_over_ocean(test_ds, query_lat, query_lon, radius)
    pct_ocean = calc_percent_over_ocean(test_ds, query_lat, query_lon, radius)
    print(f"6. Points inside radius: Total={total_pts}, Ocean={ocean_pts}, Ocean%={pct_ocean:.1f}%")

    # Test wind stats
    mean_sp, std_sp = mean_std_wind_speed(test_ds, query_lat, query_lon, radius)
    mean_dir, std_dir = mean_std_wind_dir(test_ds, query_lat, query_lon, radius)
    print(f"7. Wind Speed: Mean={mean_sp:.2f} m/s, Std={std_sp:.2f}")
    print(f"8. Wind Direction: Mean={mean_dir:.2f} deg, Std={std_dir:.2f}")

    # Test segmentation map
    seg_map = get_segmentation_map(test_ds, query_lat, query_lon, radius)
    print(f"9. Segmentation Map shape: {seg_map.shape}, True count: {seg_map.values.sum()}")
          