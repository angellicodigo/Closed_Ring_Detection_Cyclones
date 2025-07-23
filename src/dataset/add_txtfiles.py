import os
import numpy as np
import pandas as pd
from utils import get_center
import xarray as xr
from tqdm import tqdm
import shutil


PATH_TXTFILES = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\reformatted_txtfiles'
PATH_FOLDER = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info'
PATH_DATASET = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info\dataset_template.txt'
PATH_TRACKS = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\Tracks'
NUM_OF_TXT = 435
PATH_STORE_DATA = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\dataset'


def add_files() -> None:
    df = pd.read_csv(PATH_DATASET, sep=r'\t', engine='python')
    i = len(df)

    for root, _, files in os.walk(PATH_TXTFILES):
        medicanes = [1328, 1461, 1542, 1575, 1622, 1702]
        over_land = [848, 849, 860, 864, 865, 868, 871, 873, 889, 900, 907, 926, 940, 943, 950, 951, 969, 981, 985, 988, 992, 1001, 1018, 1023, 1025, 1030, 1033, 1034, 1041, 1043, 1060, 1064, 1072, 1089, 1094, 1107, 1112, 1126, 1147, 1154, 1166, 1180, 1185, 1199, 1206, 1214, 1215, 1223, 1225, 1232, 1257, 1262, 1264, 1273,
                     1274, 1276, 1306, 1308, 1317, 1318, 1322, 1327, 1341, 1365, 1392, 1404, 1405, 1408, 1414, 1420, 1441, 1450, 1452, 1453, 1456, 1484, 1491, 1492, 1496, 1497, 1533, 1535, 1562, 1563, 1568, 1569, 1570, 1583, 1595, 1596, 1603, 1605, 1608, 1612, 1614, 1615, 1616, 1625, 1648, 1649, 1650, 1651, 1664, 1666, 1686, 1700]
        for name in tqdm(files, total=NUM_OF_TXT, desc='Searching reformatted_txtfiles'):
            file_path = os.path.join(root, name)
            track = pd.read_csv(file_path, sep=r'\s+',
                                engine='python', na_values=['NaN'], header=0).dropna()
            if not track.empty:
                for _, row in track.iterrows():
                    full_cyclone_id = name[:13]
                    cyclone_id = int(name[9:13])
                    date = str(row['date(AAAAMMGG)'])
                    time = pd.to_datetime(row['time(hh:mm)'], format='%H:%M')
                    satellite = 'ASCAT' + row['satellite'][6:]

                    if (cyclone_id not in medicanes) and (cyclone_id not in over_land):
                        for track_root, _, track_files in os.walk(PATH_TRACKS):
                            for track_name in track_files:
                                if (satellite in track_name) and (full_cyclone_id in track_name) and (date in track_name):
                                    file_track = os.path.join(
                                        track_root, track_name)
                                    with xr.open_dataset(file_track) as ds:
                                        average_time = pd.to_datetime(
                                            ds.time.mean().values).replace(year=1900, month=1, day=1)
                                        if abs(average_time - time) < pd.Timedelta(minutes=30):
                                            year = int(date[:4])
                                            month = int(date[4:6])
                                            day = int(date[6:])
                                            lat, lon = get_center(
                                                cyclone_id, year, month, day, time)

                                            input = {'file_name': track_name, 'lat': lat,
                                                     'lon': lon, 'label': row['closed_ring']}
                                            df.loc[i] = input  # type: ignore
                                            i += 1

                                            PATH_DST = os.path.join(
                                                PATH_STORE_DATA, track_name)
                                            shutil.copyfile(
                                                file_track, PATH_DST)

    folder_path = os.path.join(PATH_FOLDER, "dataset_preprocessed.txt")
    df.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(ds)}') # type: ignore


if __name__ == '__main__':
    add_files()
