import argparse
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr
from config.utils import dist_bwt_two_points, get_boundary_box, calc_percent_valid, nearest_neighbors_indices, get_num_points
import numpy as np
import scipy.io

PATH_SAVE = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed'
PATH_DATASET = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\dataset'
PATH_INFO = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\interim\annotations_interm.txt'
PATH_EXTERNAL = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\external\ASCAT_ModifiedData'

MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]
OVER_LAND = [848, 849, 860, 864, 865, 868, 871, 873, 889, 900, 907, 926, 940, 943, 950, 951, 969, 981, 985, 988, 992, 1001, 1018, 1023, 1025, 1030, 1033, 1034, 1041, 1043, 1060, 1064, 1072, 1089, 1094, 1107, 1112, 1126, 1147, 1154, 1166, 1180, 1185, 1199, 1206, 1214, 1215, 1223, 1225, 1232, 1257, 1262, 1264, 1273,
             1274, 1276, 1306, 1308, 1317, 1318, 1322, 1327, 1341, 1365, 1392, 1404, 1405, 1408, 1414, 1420, 1441, 1450, 1452, 1453, 1456, 1484, 1491, 1492, 1496, 1497, 1533, 1535, 1562, 1563, 1568, 1569, 1570, 1583, 1595, 1596, 1603, 1605, 1608, 1612, 1614, 1615, 1616, 1625, 1648, 1649, 1650, 1651, 1664, 1666, 1686, 1700]
EXCLUDE = ['20120705171501_track00001366_ASCATA-L2-ICM.nc']


def preprocess(radius: float, threshold: float, checkBBox: bool):
    columns = ['cyclone_id', 'year', 'file_name', 'lat', 'lon', 'label']
    result = pd.DataFrame(columns=columns)
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing each file"):
        input = {}
        if row['cyclone_id'] in MEDICANES:
            input = {'cyclone_id': row['cyclone_id'], 'year': row['year'],
                     'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}
        elif (row['cyclone_id'] not in OVER_LAND) and (row['file_name'] not in EXCLUDE):
            if (row['lon'] != -np.inf) and (row['lat'] != -np.inf):
                file_path = os.path.join(PATH_DATASET, row['file_name'])
                with xr.open_dataset(file_path) as ds:
                    if (np.count_nonzero(~np.isnan(ds['wind_speed'].values)) != 0) and (check_boundary(ds, row['lat'], row['lon'])) and (check_within_swaths(ds, row['file_name'], row['lat'], row['lon'])):
                        if checkBBox and (get_num_points(ds, row['lat'], row['lon'], radius, True) != 0):
                            input = get_BBox_input(ds, row, radius, threshold)
                        elif get_num_points(ds, row['lat'], row['lon'], radius, False) != 0:
                            input = get_segmentation_input(ds, row, radius, threshold)
                    ds.close()

        if len(input) != 0:
            result.loc[len(result)] = input  # type: ignore

    title = ''
    if checkBBox:
        title = 'annotations_OD.txt'
    else:
        title = 'annotations_SS.txt'

    folder_path = os.path.join(PATH_SAVE, title)
    result.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(result)}')
    print(f'What years? {result["year"].unique()}')
    print(f'How many unique cyclone_ids? {len(result["cyclone_id"].unique())}')


def check_boundary(ds: xr.Dataset, lat: float, lon: float) -> bool:
    non_nan = ~np.isnan(ds['wind_speed'].values)
    min_lat = np.min(ds['lat'].values[non_nan])
    min_lon = np.min(ds['lon'].values[non_nan])
    max_lat = np.max(ds['lat'].values[non_nan])
    max_lon = np.max(ds['lon'].values[non_nan])

    # Excludes points on the boundary of the swaths, though this is not a perfect check
    if ((lon > min_lon) and (lon < max_lon) and (lat > min_lat) and (lat < max_lat)):
        return True
    return False


def check_within_swaths(ds: xr.Dataset, file_name: str, query_lat: float, query_lon: float) -> bool:
    for root, _, files in os.walk(PATH_EXTERNAL):
        for mat_name in files:
            if os.path.splitext(mat_name)[0] == os.path.splitext(file_name)[0]:
                mat_path = os.path.join(root, mat_name)
                mat = scipy.io.loadmat(
                    mat_path, struct_as_record=False, squeeze_me=True)
                data = mat['ASCATnew']
                lat = data.lat
                lon = data.lon
                distances = np.nan_to_num(dist_bwt_two_points(
                    query_lat, query_lon, lat, lon), nan=np.inf)
                # Index has both row and col index
                index = np.unravel_index(
                    np.argmin(distances.flatten()), lon.shape)
                return not np.isnan(data.wind_speed[index])
    return False


def get_BBox_input(ds: xr.Dataset, row: pd.Series, radius: float, threshold: float) -> dict:
    input = {}
    percent = calc_percent_valid(ds, row['lat'], row['lon'], radius, True)
    if percent >= threshold:
        input = {'cyclone_id': row['cyclone_id'], 'year': row['year'],
                 'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}

    return input


def get_segmentation_input(ds: xr.Dataset, row: pd.Series, radius: float, threshold: float) -> dict:
    input = {}
    percent = calc_percent_valid(ds, row['lat'], row['lon'], radius, False)
    if percent >= threshold:
        mask = ds['wvc_index'].notnull()
        temp = ds.where(mask, drop=True)
        height, width = temp['lon'].shape[0], temp['lat'].shape[1]
        i, j = nearest_neighbors_indices(
            temp, row['lat'], row['lon'])
        i = i[0]
        j = j[0]
        if (i > 0) and (i < height - 1):
            if (width == 81) and (j > 0) and (j < width - 1):
                input = {'cyclone_id': row['cyclone_id'], 'year': row['year'],
                        'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}
            elif (width == 82) and (j > 1) and (j < width - 2) and (j != 42):
                input = {'cyclone_id': row['cyclone_id'], 'year': row['year'],
                        'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}
    return input


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--checkBBox", type=bool, default=False)
    args = parser.parse_args()
    preprocess(args.radius, args.threshold, args.checkBBox)
