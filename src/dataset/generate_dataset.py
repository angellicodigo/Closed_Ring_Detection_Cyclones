import os
import xarray as xr
from config.utils import get_mean_info, get_center
import pandas as pd
import shutil

PATH_TRACKS = r'data\raw\Tracks'
PATH_DATASET = r'data\processed\dataset'
PATH_INFO = r'data\raw\annotations_template.txt'
PATH_WHERE_SAVE = r'data\processed'
PATH_INTERM = r'data\interim'

MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]
OVER_LAND = [848, 849, 860, 864, 865, 868, 871, 873, 889, 900, 907, 926, 940, 943, 950, 951, 969, 981, 985, 988, 992, 1001, 1018, 1023, 1025, 1030, 1033, 1034, 1041, 1043, 1060, 1064, 1072, 1089, 1094, 1107, 1112, 1126, 1147, 1154, 1166, 1180, 1185, 1199, 1206, 1214, 1215, 1223, 1225, 1232, 1257, 1262, 1264, 1273,
             1274, 1276, 1306, 1308, 1317, 1318, 1322, 1327, 1341, 1365, 1392, 1404, 1405, 1408, 1414, 1420, 1441, 1450, 1452, 1453, 1456, 1484, 1491, 1492, 1496, 1497, 1533, 1535, 1562, 1563, 1568, 1569, 1570, 1583, 1595, 1596, 1603, 1605, 1608, 1612, 1614, 1615, 1616, 1625, 1648, 1649, 1650, 1651, 1664, 1666, 1686, 1700]


def add_files():
    save_path = os.path.join(PATH_WHERE_SAVE, 'dataset')
    os.makedirs(save_path, exist_ok=True)
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for root, _, files in os.walk(PATH_TRACKS):
        for file_name in files:
            cyclone_id = int(file_name.split('_')[1][5:])
            if ('ASCAT' in file_name) and (cyclone_id not in MEDICANES) and (cyclone_id not in OVER_LAND):
                path = os.path.join(root, file_name)
                with xr.open_dataset(path) as ds:
                    average_time, year, month, day = get_mean_info(ds)
                    center_lat, center_lon = get_center(
                        cyclone_id, year, month, day, average_time)
                    input = {'cyclone_id': cyclone_id, 'year': year, 'file_name': file_name,
                             'lat': center_lat, 'lon': center_lon, 'label': 'N/A'}
                    df.loc[len(df)] = input  # type: ignore
                    PATH_DST = os.path.join(save_path, file_name)
                    shutil.copyfile(path, PATH_DST)

    folder_path = os.path.join(PATH_INTERM, "annotations_all.txt")
    df.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(ds)}')  # type: ignore


if __name__ == '__main__':
    add_files()
