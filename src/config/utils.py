from typing import Tuple
from typing import Union
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
from scipy.stats import circstd
from pyproj import Geod

PATH_CENTERS = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info\TRACKS_CL7.dat'


def get_center(cyclone_id: int, year: int, month: int, day: int, time: pd.Timestamp) -> Tuple[float, float]:
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


def nearest_neighbors_indices(ds: xr.Dataset, query_lat: float, query_lon: float) -> Tuple[np.ndarray, np.ndarray]:
    distances = haversine(ds, query_lon, query_lat)
    sorted_indices = np.argsort(distances)
    mask = ds['lon'].notnull().values
    valid_indices = np.where(mask)
    original_index = (valid_indices[0][sorted_indices],
                      valid_indices[1][sorted_indices])
    return original_index


def get_mean_info(ds: xr.Dataset) -> Tuple[pd.Timestamp, int, int, int]:
    average_time = pd.to_datetime(ds.time.mean().values)
    year = int(average_time.year)
    month = int(average_time.month)
    day = int(average_time.day)

    return average_time, year, month, day


def get_boundary_box(ds: xr.Dataset, query_lat: float, query_lon: float, radius: float) -> Tuple[float, float, float, float]:
    # row_indices, col_indices = nearest_neighbors_indices(
    #     ds, query_lat, query_lon)
    # nearest_row = row_indices[0]
    # nearest_col = col_indices[0]
    # dim = list(ds['lon'].sizes)
    # row_dim = dim[0]
    # col_dim = dim[1]
    # # Use the nearest point as the center to calculate the boundary box
    # nearest_point = ds.isel({row_dim: nearest_row, col_dim: nearest_col})
    # near_lat = nearest_point["lat"].values
    # near_lon = nearest_point["lon"].values
    # radius = radius * 1000
    # g = Geod(ellps="WGS84")
    # _, latN, _ = g.fwd(near_lon, near_lat, 0,   radius)
    # lonE, _, _ = g.fwd(near_lon, near_lat, 90,  radius)
    # _, latS, _ = g.fwd(near_lon, near_lat, 180, radius)
    # lonW, _, _ = g.fwd(near_lon, near_lat, 270, radius)
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


def get_num_points_bbox(ds: xr.Dataset, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> float:
    mask = (min_lon <= ds.lon) & (ds.lon <= max_lon) & (
        min_lat <= ds.lat) & (ds.lat <= max_lat)
    return len(ds['wind_speed'].values[mask])


def calc_percent_valid(ds: xr.Dataset, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> float:
    mask = (min_lon <= ds.lon) & (ds.lon <= max_lon) & (
        min_lat <= ds.lat) & (ds.lat <= max_lat)
    wind_speed = ds['wind_speed'].values[mask]
    non_nan = ~np.isnan(wind_speed)

    return (np.count_nonzero(non_nan) / len(wind_speed)) * 100


def calc_std_wind_direction(ds: xr.Dataset, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> float:
    mask = (min_lon <= ds['lon']) & (ds['lon'] <= max_lon) & (
        min_lat <= ds['lat']) & (ds['lat'] <= max_lat)
    wind_direction = np.deg2rad(ds['wind_dir'].values[mask])
    return circstd(wind_direction)


def calc_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
        The angle (bearing) from (lon1, lat1) to (lon2, lat2)
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dLon = lon2 - lon1

    y = np.sin(dLon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * \
        np.cos(lat2) * np.cos(dLon)

    bearing = np.atan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing  # Returns clockwise degrees


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


def calc_bearing_given_indices(row1: Union[float, np.ndarray], col1: Union[float, np.ndarray], row2: float, col2: float, lat) -> Union[float, np.ndarray]:
    """
        The bearing from (x1, y1) to (x2, y2) going clockwise from North line pointing up
    """

    # Converting indices to km (https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance)
    km_per_lat = 110.574
    km_per_lon = 111.320 * np.cos(np.deg2rad(lat))

    dx = (row2 - row1) * km_per_lat  # By lat because rows = height
    dy = (col2 - col1) * km_per_lon  # By lon because cols = width

    # Perform a 90 degree clockwise rotation with (dx, dy) -> (-dy, dx) and then flip across origin (dy, -dx)
    flip_x = dy
    flip_y = -dx

    # Convert to a bearing by begin in normal coordinate system
    bearing = (90 - np.atan2(flip_y, flip_x) * 180 / np.pi) % 360
    return bearing


def get_bearing_indices(ds: xr.Dataset, query_lat: float, query_lon: float) -> Union[float, np.ndarray]:
    near_row_indices, near_col_indices = nearest_neighbors_indices(
        ds, query_lat, query_lon)
    nearest_row_index = near_row_indices[0]
    nearest_col_index = near_col_indices[0]
    length = ds['lon'].shape[0]  # Length and width are the same for most cases
    width = ds['lon'].shape[1]
    row_indices, col_indices = np.indices((length, width))
    return calc_bearing_given_indices(row_indices, col_indices, nearest_row_index, nearest_col_index, query_lat)


def calc_distances_given_indices(row1: Union[float, np.ndarray], col1: Union[float, np.ndarray], row2: float, col2: float, lat) -> Union[float, np.ndarray]:
    # Note: I think euclidean distance is more accurate than chebyshev distance in this case
    dx = (row2 - row1)**2
    dy = ((col2 - col1) * np.cos(np.deg2rad(lat)))**2  # Creates a oval shape
    dist = np.sqrt(dx + dy)
    return dist * 4.15  # A value determined by visual inspection of images


def get_distances_indices(ds: xr.Dataset, query_lat: float, query_lon: float) -> Union[float, np.ndarray]:
    near_row_indices, near_col_indices = nearest_neighbors_indices(
        ds, query_lat, query_lon)
    nearest_row_index = near_row_indices[0]
    nearest_col_index = near_col_indices[0]
    length = ds['lon'].shape[0]  # Length and width are the same for most cases
    width = ds['lon'].shape[1]
    row_indices, col_indices = np.indices((length, width))
    return calc_distances_given_indices(row_indices, col_indices, nearest_row_index, nearest_col_index, query_lat)


def coords_to_pixels(ds: xr.Dataset, query_lat: float, query_lon: float, center_lat: float, center_lon: float) -> Tuple[int, int]:
    near_row_indices, near_col_indices = nearest_neighbors_indices(
        ds, center_lat, center_lon)
    nearest_row_index = near_row_indices[0]
    nearest_col_index = near_col_indices[0]
    lat_dim, lon_dim = list(ds.sizes.keys()) 
    nearest_point = ds.isel(
        {lat_dim: nearest_row_index, lon_dim: nearest_col_index})
    near_lat = float(nearest_point['lat'].values)
    near_lon = float(nearest_point['lon'].values)

    index_bearing = get_bearing_indices(ds, center_lat, center_lon)
    geo_bearing = calc_bearing(query_lat, query_lon, near_lat, near_lon) # Calculated by near_lat near_lon
    round_to = 0  # Best value to choose by looking at images
    mask = np.round(geo_bearing, round_to) == np.round(index_bearing, round_to)
    rows, cols = np.where(mask)

    dist = calc_distances_given_indices(
        rows, cols, nearest_row_index, nearest_col_index, query_lat)
    geo_dist = dist_bwt_two_points(query_lat, query_lon, near_lat, near_lon)
    index = np.argmin(abs(dist - geo_dist))
    return rows[index], cols[index]
