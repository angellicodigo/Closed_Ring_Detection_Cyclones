import argparse
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr
from config.utils import get_boundary_box, calc_percent_valid, get_mean_info, nearest_neighbors_indices
import numpy as np

PATH_SAVE =r'/home/angel/ML_for_Medicane_Wind_Rings/data/processed'
PATH_DATASET = r'/home/angel/ML_for_Medicane_Wind_Rings/data/processed/dataset'
PATH_INFO = r'/home/angel/ML_for_Medicane_Wind_Rings/data/interim/dataset_preprocessed.txt'



def generate(radius: float, threshold: float, isSS: bool):
    columns = ['cyclone_id', 'year', 'file_name', 'lat', 'lon', 'label']
    result = pd.DataFrame(columns=columns)
    df = pd.read_csv(PATH_INFO, sep=r'\t', engine='python')
    title = ''
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating boundary box for all files"):
        file_path = os.path.join(PATH_DATASET, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            if within_swath(ds, row['lat'], row['lon']):
                file_name = os.path.basename(file_path)
                cyclone_id = file_name.split('_')[1]
                _, year, _, _ = get_mean_info(ds)
                if isSS:
                    title = 'annotations_SS.txt'
                    mask = ds['wvc_index'].notnull()
                    temp = ds.where(mask, drop=True)
                    height, width = temp['lon'].shape[0], temp['lat'].shape[1]
                    i, j = nearest_neighbors_indices(temp, row['lat'], row['lon'])
                    i = i[0]
                    j = j[0]
                    if (i > 0) and (i < height - 1):
                        if (width == 81) and (j > 0) and (j < width - 1):
                            input = {'cyclone_id': cyclone_id, 'year': year, 'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon']}
                            result.loc[len(result)] = input # type: ignore
                        elif (width == 82) and (j > 1) and (j < width - 2):
                            input = {'cyclone_id': cyclone_id, 'year': year, 'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon']}
                            result.loc[len(result)] = input # type: ignore
                else:
                    title = 'annotations_OD.txt'
                    min_lat, min_lon, max_lat, max_lon = get_boundary_box(ds, row['lat'], row['lon'], radius)
                    percent = calc_percent_valid(ds, min_lat, min_lon, max_lat, max_lon)
                    if percent >= threshold:
                        input = {'cyclone_id': cyclone_id, 'year': year, 'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}
                        result.loc[len(result)] = input # type: ignore
    
    folder_path = os.path.join(PATH_SAVE, title)
    result.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(result)}')
    print(f'What years? {result["year"].unique()}')
    print(f'How many years? {len(result["year"].unique())}')
    print(f'What cyclone_id? {result["cyclone_id"].unique()}')
    print(f'How many cyclone_id? {len(result["cyclone_id"].unique())}')

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
    parser.add_argument("--radius", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=95)
    parser.add_argument("--isSemanticSegmentation", type=bool, default=True)
    args = parser.parse_args()
    generate(args.radius, args.threshold, args.isSemanticSegmentation)
