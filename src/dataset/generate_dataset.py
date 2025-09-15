import os
import xarray as xr
from config.utils import get_mean_info, get_center
import pandas as pd
import shutil
import numpy as np
from tqdm import tqdm

# Path to the folder with ASCAT files (161 x 161)
PATH_TRACKS = r'data\raw\Tracks'
# Path where those ASCAT files are saved in
PATH_DATASET = r'data\processed\dataset'
# Path to a txt files with pre-annotated Medicanes
PATH_INFO = r'data\raw\annotations_template.txt'
# Path where annotations of all cyclones are saved
PATH_SAVE = r'data\raw'
NUM_OF_FOLDERS = 5939  # Number of files in raw/Tracks

# Pre-annotated Medicanes that are already
MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]


def add_files() -> None:
    """
    Returns:
        Creates dataset folder in data/processed and creates annotations_interm.txt in data/interim
        
    """
    os.makedirs(PATH_DATASET, exist_ok=True)
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
                    # Copies the files from raw/Tracks to processed/dataset
                    # PATH_DST = os.path.join(PATH_DATASET, file_name)
                    # shutil.copyfile(path, PATH_DST)

    folder_path = os.path.join(PATH_SAVE, "annotations_raw.txt")
    df.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(df)}') 


if __name__ == '__main__':
    add_files()
