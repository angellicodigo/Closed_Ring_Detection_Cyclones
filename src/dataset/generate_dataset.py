import argparse
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr
from utils import get_boundary_box, calc_percent_valid, get_mean_info
import numpy as np

PATH_SAVE =r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info'
PATH_DATASET = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\dataset'
PATH_INFO = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\medicanes_info\dataset_preprocessed.txt'



def generate(radius: float, threshold: float):
    columns = ['cyclone_id', 'year', 'file_name', 'lat', 'lon', 'label']
    result = pd.DataFrame(columns=columns)
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating boundary box for all files"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
                if within_swath(ds, row['lat'], row['lon']):
                    min_lat, min_lon, max_lat, max_lon = get_boundary_box(ds, row['lat'], row['lon'], radius)
                    percent = calc_percent_valid(ds, min_lat, min_lon, max_lat, max_lon)
                    file_name = os.path.basename(file_path)
                    cyclone_id = file_name.split('_')[1]
                    _, year, _, _ = get_mean_info(ds)
                    if percent >= threshold:
                        input = {'cyclone_id': cyclone_id, 'year': year, 'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}
                        result.loc[len(result)] = input # type: ignore

    folder_path = os.path.join(PATH_SAVE, "annotations.txt")
    result.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(result)}')

def within_swath(ds: xr.Dataset, center_lat: float, center_lon: float) -> bool:
    non_nan = ~np.isnan(ds['wind_speed'].values)
    min_lat = np.min(ds['lat'].values[non_nan])
    min_lon = np.min(ds['lon'].values[non_nan])
    max_lat = np.max(ds['lat'].values[non_nan])
    max_lon = np.max(ds['lon'].values[non_nan])

    # Excludes points on the boundary of the swaths. This is a soft check
    if ((center_lon > min_lon) and (center_lon < max_lon) and (center_lat > min_lat) and (center_lat < max_lat)):
        return True
    return False
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("radius", type=int, default=100)
    parser.add_argument("threshold", type=float, default=95)
    args = parser.parse_args()
    generate(args.radius, args.threshold)
