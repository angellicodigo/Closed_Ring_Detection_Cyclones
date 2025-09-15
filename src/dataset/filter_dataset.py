import argparse
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr
from config.utils import get_num_points_over_ocean, calc_percent_over_ocean

# Path to save annotations.txt
PATH_SAVE = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed'
# Path to the dataset where the ASCAT files are located
PATH_DATASET = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\dataset'
# Path to annotations_interm.txt
PATH_INFO = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\interim\annotations_interm.txt'

MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]

def filter(radius: float, threshold: float, num: int) -> None:
    """
    Args:
        radius: distance km away from the center of the cyclone in each ASCAT file
        threshold: the desired percentage of points that are over the ocean
        num: the desired number of points are over the ocean

    Returns:
        Creates annotations.txt that filters through each row in annotations_interm.txt based on radius, threshold, and num 
        
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
        else:
            file_path = os.path.join(PATH_DATASET, row['file_name'])
            with xr.open_dataset(file_path) as ds:
                if (get_num_points_over_ocean(ds, row['lat'], row['lon'], radius) >= num) and (calc_percent_over_ocean(ds, row['lat'], row['lon'], radius) >= threshold):
                    input = {'cyclone_id': row['cyclone_id'], 'year': row['year'], 'file_name': row['file_name'], 'lat': row['lat'], 'lon': row['lon'], 'label': row['label']}

        if len(input) != 0:
            result.loc[len(result)] = input  # type: ignore

    folder_path = os.path.join(PATH_SAVE, 'annotations.txt')
    result.to_csv(folder_path, index=False, sep='\t')
    print(f'How many files? {len(result)}')
    print(f'How many cyclones? {len(result["cyclone_id"].unique())}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=67)
    parser.add_argument("--n", type=int, default=109)
    args = parser.parse_args()
    filter(args.radius, args.threshold, args.n)