import os
import xarray as xr
from config.utils import get_mean_info, get_center
import pandas as pd
import shutil
import numpy as np
from tqdm import tqdm

PATH_TRACKS = r'data\raw\Tracks'
PATH_DATASET = r'data\processed\dataset'
PATH_INFO = r'data\raw\annotations_template.txt'
PATH_WHERE_SAVE = r'data\processed'
PATH_INTERM = r'data\interim'
NUM_OF_FOLDERS = 5939

MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]


def add_files():
    save_path = os.path.join(PATH_WHERE_SAVE, 'dataset')
    os.makedirs(save_path, exist_ok=True)
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for root, _, files in tqdm(os.walk(PATH_TRACKS), total=NUM_OF_FOLDERS, desc='Searching through each folder in Tracks', unit='folder'):
        for file_name in files:
            cyclone_id = int(file_name.split('_')[1][5:])
            if ('ASCAT' in file_name) and (cyclone_id not in MEDICANES):
                path = os.path.join(root, file_name)
                with xr.open_dataset(path) as ds:
                    average_time, year, month, day = get_mean_info(ds)
                    center_lat, center_lon = get_center(
                        cyclone_id, year, month, day, average_time)
                    input = {'cyclone_id': cyclone_id, 'year': year, 'file_name': file_name,
                             'lat': center_lat, 'lon': center_lon, 'label': np.nan}
                    df.loc[len(df)] = input  # type: ignore
                    # PATH_DST = os.path.join(save_path, file_name)
                    # shutil.copyfile(path, PATH_DST)

    folder_path = os.path.join(PATH_INTERM, "annotations_interm.txt")
    df.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(df)}')  # type: ignore


if __name__ == '__main__':
    add_files()
