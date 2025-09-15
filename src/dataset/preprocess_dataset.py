import pandas as pd
from tqdm import tqdm
import os
import xarray as xr
from config.utils import dist_bwt_two_points
import numpy as np
import scipy.io

# Path to save the preprocessed annotations_raw.txt
PATH_SAVE = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\interim'
# Path to the dataset where the ASCAT files are located
PATH_DATASET = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\dataset'
# Path to annotations_raw.txt
PATH_INFO = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\raw\annotations_raw.txt'
# Path to external data used in check_swath()
PATH_EXTERNAL = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\external\ASCAT_ModifiedData'

# Pre-labed medicanes that are skipped and copied over
MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]
# List of cyclone ids of cyclones that are always over land
OVER_LAND = [848, 849, 860, 864, 865, 868, 871, 873, 889, 900, 907, 926, 940, 943, 950, 951, 969, 981, 985, 988, 992, 1001, 1018, 1023, 1025, 1030, 1033, 1034, 1041, 1043, 1060, 1064, 1072, 1089, 1094, 1107, 1112, 1126, 1147, 1154, 1166, 1180, 1185, 1199, 1206, 1214, 1215, 1223, 1225, 1232, 1257, 1262, 1264, 1273,
             1274, 1276, 1306, 1308, 1317, 1318, 1322, 1327, 1341, 1365, 1392, 1404, 1405, 1408, 1414, 1420, 1441, 1450, 1452, 1453, 1456, 1484, 1491, 1492, 1496, 1497, 1533, 1535, 1562, 1563, 1568, 1569, 1570, 1583, 1595, 1596, 1603, 1605, 1608, 1612, 1614, 1615, 1616, 1625, 1648, 1649, 1650, 1651, 1664, 1666, 1686, 1700]
# Set of cyclones that are excluded. These files were manually checked
EXCLUDE = ['20101010190801_track00001282_ASCATA-L2-ICM.nc', '20110129093048_track00001297_ASCATA-L2-ICM.nc', '20161104091411_track00001543_ASCATB-L2-ICM.nc', '20161121082237_track00001544_ASCATB-L2-ICM.nc', '20161221210152_track00001548_ASCATB-L2-ICM.nc', '20161221191946_track00001549_ASCATB-L2-ICM.nc', '20170226081530_track00001560_ASCATB-L2-ICM.nc', '20170226193400_track00001560_ASCATB-L2-ICM.nc', '20170308080733_track00001561_ASCATB-L2-ICM.nc', '20180111095218_track00001581_ASCATB-L2-ICM.nc', '20180112093226_track00001581_ASCATB-L2-ICM.nc', '20180126192100_track00001585_ASCATB-L2-ICM.nc', '20180214094856_track00001590_ASCATB-L2-ICM.nc', '20180305202020_track00001594_ASCATA-L2-ICM.nc', '20180322080620_track00001599_ASCATA-L2-ICM.nc', '20180409191148_track00001602_ASCATB-L2-ICM.nc', '20181023081418_track00001626_ASCATB-L2-ICM.nc', '20181023193239_track00001626_ASCATB-L2-ICM.nc', '20181118091600_track00001629_ASCATB-L2-ICM.nc', '20181120083403_track00001632_ASCATB-L2-ICM.nc', '20181206080352_track00001634_ASCATB-L2-ICM.nc', '20181206192246_track00001634_ASCATB-L2-ICM.nc', '20190104080356_track00001638_ASCATB-L2-ICM.nc', '20190114191724_track00001640_ASCATB-L2-ICM.nc', '20190126084833_track00001642_ASCATB-L2-ICM.nc', '20190205200224_track00001644_ASCATB-L2-ICM.nc', '20190214074743_track00001645_ASCATA-L2-ICM.nc', '20190217064503_track00001645_ASCATA-L2-ICM.nc',
           '20190217075409_track00001645_ASCATB-L2-ICM.nc', '20190327192724_track00001652_ASCATB-L2-ICM.nc', '20190408184043_track00001655_ASCATB-L2-ICM.nc', '20190504082345_track00001660_ASCATB-L2-ICM.nc', '20191003091913_track00001670_ASCATB-L2-ICM.nc', '20191024072933_track00001672_ASCATA-L2-ICM.nc', '20191024081037_track00001672_ASCATC-L2-ICM.nc', '20191027071003_track00001672_ASCATC-L2-ICM.nc', '20191107201413_track00001673_ASCATB-L2-ICM.nc', '20191107082101_track00001673_ASCATC-L2-ICM.nc', '20191120092335_track00001676_ASCATB-L2-ICM.nc', '20191118205431_track00001676_ASCATC-L2-ICM.nc', '20200109071101_track00001681_ASCATB-L2-ICM.nc', '20200109080946_track00001681_ASCATC-L2-ICM.nc', '20200117102150_track00001682_ASCATC-L2-ICM.nc', '20200331080800_track00001689_ASCATC-L2-ICM.nc', '20200415075705_track00001692_ASCATC-L2-ICM.nc', '20200420090637_track00001694_ASCATA-L2-ICM.nc', '20200420205445_track00001694_ASCATC-L2-ICM.nc', '20201012084437_track00001706_ASCATA-L2-ICM.nc', '20201030073241_track00001707_ASCATA-L2-ICM.nc', '20201027192939_track00001707_ASCATB-L2-ICM.nc', '20201120083746_track00001710_ASCATA-L2-ICM.nc', '20201119204852_track00001710_ASCATC-L2-ICM.nc', '20201204082246_track00001713_ASCATB-L2-ICM.nc', '20201202193945_track00001713_ASCATC-L2-ICM.nc', '20201213195639_track00001715_ASCATB-L2-ICM.nc', '20201212193120_track00001715_ASCATC-L2-ICM.nc', '20201215070918_track00001715_ASCATC-L2-ICM.nc', '20201226202913_track00001716_ASCATB-L2-ICM.nc', '20180321092435_track00001598_ASCATB-L2-ICM.nc', '20180916063422_track00001620_ASCATA-L2-ICM.nc', '20181217081016_track00001637_ASCATA-L2-ICM.nc', '20190406080213_track00001654_ASCATB-L2-ICM.nc', '20201225193630_track00001716_ASCATA-L2-ICM.nc', '20190214190615_track00001645_ASCATA-L2-ICM.nc', '20191114192724_track00001675_ASCATB-L2-ICM.nc', '20191024192815_track00001672_ASCATC-L2-ICM.nc', '20191211201020_track00001678_ASCATB-L2-ICM.nc']


def preprocess() -> None:
    """
    Returns:
        Creates annotations_interm.txt that preprocessed certain files in annotations_raw.txt

    """
    columns = ['cyclone_id', 'year', 'file_name',
               'lat', 'lon', 'label']
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
                    if (np.count_nonzero(~np.isnan(ds['wind_speed'].values)) != 0) and (check_edges(ds, row['lat'], row['lon'])) and (check_within_swaths(row['file_name'], row['lat'], row['lon'])):
                        input = {'cyclone_id': row['cyclone_id'], 'year': row['year'],
                                 'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}
                    ds.close()

        if len(input) != 0:
            result.loc[len(result)] = input  # type: ignore

    folder_path = os.path.join(PATH_SAVE, 'annotations_interm.txt')
    result.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(result)}')


def check_edges(ds: xr.Dataset, lat: float, lon: float) -> bool:
    non_nan = ~np.isnan(ds['wind_speed'].values)
    min_lat = np.min(ds['lat'].values[non_nan])
    min_lon = np.min(ds['lon'].values[non_nan])
    max_lat = np.max(ds['lat'].values[non_nan])
    max_lon = np.max(ds['lon'].values[non_nan])

    # Excludes points on the boundary of the swaths, though this is not a perfect check
    if ((lon > min_lon) and (lon < max_lon) and (lat > min_lat) and (lat < max_lat)):
        return True
    return False


def check_within_swaths(file_name: str, query_lat: float, query_lon: float) -> bool:
    """Uses special MatLab files to check if the query point is within the swaths. 

    Args:
        file_name: name of the ASCAT file
        query_lat: latitude of the query point in the ASCAT file
        query_lon: longitude of the query point in the ASCAT file

    Returns:
        True if the query point is within the swath, else returns False

    """
    for root, _, files in os.walk(PATH_EXTERNAL):
        for mat_name in files:
            # Once you find the corresponding mat file to the netcdf file, stop looping through
            if os.path.splitext(mat_name)[0] == os.path.splitext(file_name)[0]:
                mat_path = os.path.join(root, mat_name)
                mat = scipy.io.loadmat(
                    mat_path, struct_as_record=False, squeeze_me=True)
                data = mat['ASCATnew']
                lat = data.lat
                lon = data.lon
                distances = np.nan_to_num(dist_bwt_two_points(
                    query_lat, query_lon, lat, lon), nan=np.inf)
                # Index is a tuple with both row and col index. Index correspond to the nearest point
                index = np.unravel_index(
                    np.argmin(distances.flatten()), lon.shape)
                wind_speed_is_nan = np.isnan(data.wind_speed[index])
                min_lat = data.lat[index]
                min_lon = data.lon[index]

                file_path = os.path.join(PATH_DATASET, file_name)
                with xr.open_dataset(file_path) as ds:
                    ori_lat = ds['lat'].values
                    ori_lon = ds['lon'].values

                    if (wind_speed_is_nan == False) and (min_lat in ori_lat) and (min_lon in ori_lon):
                        return True
                    else:
                        return False
    return False


if __name__ == '__main__':
    preprocess()
