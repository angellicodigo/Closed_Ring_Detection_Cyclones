from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import xarray as xr
import numpy as np
from config.utils import get_boundary_box, coords_to_pixels, get_segmentation_map, nearest_neighbors_indices


class CycloneDatasetOD(Dataset):  # For object detection
    def __init__(self, path_txt: str, root_dir: str, radius=100, transform=None):
        self.annotations = pd.read_csv(path_txt, sep=r'\t', engine='python')
        self.root_dir = root_dir
        self.radius = radius
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        file_path = os.path.join(self.root_dir, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
            V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))

            data = torch.from_numpy(
                xr.concat([U, V], dim='channel').values).float()

            min_lat, min_lon, max_lat, max_lon = get_boundary_box(
                ds, row['lat'], row['lon'], self.radius)
            top_left = (min_lon, max_lat)
            bot_right = (max_lon, min_lat)

            x1, y1 = coords_to_pixels(
                ds, top_left[1], top_left[0], row['lat'], row['lon'])
            x2, y2 = coords_to_pixels(
                ds, bot_right[1], bot_right[0], row['lat'], row['lon'])

            target = {}
            target['boxes'] = torch.tensor(
                [[x1, y1, x2, y2]], dtype=torch.float32)
            target['labels'] = torch.tensor([row['label']], dtype=torch.int64)

            if self.transform is not None:
                data, target = self.transform(data, target)

            # Replace nan with -1
            data = torch.nan_to_num(data, nan=-1)
            ds.close()
            return data, target


class CycloneDatasetSS(Dataset):  # For semantic segmentation
    def __init__(self, path_txt: str, root_dir: str, radius=100, transform=None):
        self.annotations = pd.read_csv(path_txt, sep=r'\t', engine='python')
        self.root_dir = root_dir
        self.radius = radius
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        file_path = os.path.join(self.root_dir, row['file_name'])
        with xr.open_dataset(file_path) as ds:
            i, j = nearest_neighbors_indices(ds, row['lat'], row['lon'])
            non_nan = ds['wvc_index'].notnull()
            ds = ds.where(non_nan, drop=True)

            row_dim = list(ds['lon'].sizes)[0]
            col_dim = list(ds['lon'].sizes)[1]
            # Resize dataset from (161 x 82) or (161 x 81) to (160 x 80)
            if i[0] >= int(ds['lon'].shape[1] / 2):
                ds = ds.drop_isel({col_dim: 0})
                if ds['lon'].shape[1] == 81:
                    ds = ds.drop_isel({col_dim: 1})
            else:
                ds = ds.drop_isel({col_dim: -1})
                if ds['lon'].shape[1] == 81:
                    ds = ds.drop_isel({col_dim: -2})

            if j[0] >= int(ds['lon'].shape[0] / 2):
                ds = ds.drop_isel({row_dim: 0})
            else:
                ds = ds.drop_isel({row_dim: -1})

            U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
            V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))

            land_sea_mask = ds['wind_speed'].notnull().astype(int)

            # data = torch.from_numpy(
            #     xr.concat([U, V, land_sea_mask, ds['lon'], ds['lat']], dim='channel').values).float()
            data = torch.from_numpy(
                xr.concat([U, V, land_sea_mask], dim='channel').values).float()

            mask = get_segmentation_map(
                ds, row['lat'], row['lon'], self.radius)
            mask = xr.where(mask, row['label'] + 1, 0)
            mask = torch.from_numpy(mask.values).long()

            if self.transform is not None:
                data, mask = self.transform(data, mask)

            # Replace nan with -1
            data = torch.nan_to_num(data, nan=-1)
            ds.close()
            return data, mask
