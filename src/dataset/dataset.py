from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import xarray as xr
import numpy as np
from src.config.utils import get_segmentation_map, nearest_neighbors_indices
from typing import List


class CycloneDataset(Dataset):
    def __init__(self, path_txt: str, root_dir: str, radius=100, num_classes = 2, transform=None, metadata=False, augment=False, reduction_ratio=None):
        self.radius = radius
        self.transform = transform
        self.data = []
        self.metadata = metadata
        self.epsilon = 1e-07

        org_annotations = pd.read_csv(path_txt, sep=r'\t', engine='python')
        annotations = pd.DataFrame(columns=org_annotations.columns)
        for _, row in org_annotations.iterrows():
            file_path = os.path.join(root_dir, row['file_name'])
            with xr.open_dataset(file_path) as ds:
                i, j = nearest_neighbors_indices(ds, row['lat'], row['lon'])
                i, j = i[0], j[0]
                non_nan = ds['wvc_index'].notnull()
                ds = ds.where(non_nan, drop=True)

                row_dim = list(ds['lon'].sizes)[0]
                col_dim = list(ds['lon'].sizes)[1]
                # Resize dataset from (161 x 82) or (161 x 81) to (160 x 80) by dropping columns
                if i >= int(ds['lon'].shape[1] / 2):
                    ds = ds.drop_isel({col_dim: 0})
                    if ds['lon'].shape[1] == 81:
                        ds = ds.drop_isel({col_dim: 1})
                else:
                    ds = ds.drop_isel({col_dim: -1})
                    if ds['lon'].shape[1] == 81:
                        ds = ds.drop_isel({col_dim: -2})

                if j >= int(ds['lon'].shape[0] / 2):
                    ds = ds.drop_isel({row_dim: 0})
                else:
                    ds = ds.drop_isel({row_dim: -1})

                upper_limit = 0.85
                lower_limit = 0.15

                indices = []
                if (j > 39) and (j < 119):
                    dist_top = j
                    dist_bot = 159 - j
                    start = 0
                    end = 0
                    if dist_top <= dist_bot:
                        start = int(dist_top * lower_limit)
                        end = int(dist_top * upper_limit)
                        index = np.random.randint(start, end)
                        indices = np.arange(index, index + 80)
                    else:
                        start = j + int(dist_bot * lower_limit)
                        end = j + int(dist_bot * upper_limit)
                        index = np.random.randint(start, end)
                        indices = np.arange(index - 80, index)
                elif j <= 39:
                    indices = np.arange(80)
                elif j >= 119:
                    indices = np.arange(80, 160)

                # Now 80 x 80 by keeping certain columns
                ds = ds.isel({row_dim: indices})
                ds['U'] = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
                ds['V'] = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))

                self.data.append(ds.copy())
                annotations.loc[len(annotations)] = row.copy() 

                if augment and (row['label'] == 1):
                    ds['U'] = -ds['U']
                    ds['V'] = -ds['V']
                    self.data.append(ds)
                    annotations.loc[len(annotations)] = row.copy()  # So data and annotations are same length


        if reduction_ratio != None:
            indices = annotations.index[annotations['label'] == 0].to_numpy()
            n = int(len(indices) * reduction_ratio)
            remove_indices = np.random.choice(indices, size=n, replace=False)
            self.data = [data for i, data in enumerate(self.data) if i not in remove_indices]
            annotations = annotations.drop(remove_indices).reset_index(drop=True) # type: ignore
        
        self.annotations = annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        ds = self.data[idx]

        data = torch.from_numpy(
            xr.concat([ds['U'], ds['V']], dim='channel').values).float()

        if row['label'] == 1:
            mask = get_segmentation_map(
                ds, row['lat'], row['lon'], self.radius)
            mask = xr.where(mask, row['label'], 0)
            mask = torch.from_numpy(mask.values).long()
        else:
            mask = torch.zeros(
                (data.shape[1], data.shape[2]), dtype=torch.long)

        binary_mask = torch.from_numpy(
            ~np.isnan(ds['wind_speed'].values)).float().unsqueeze(0)

        # mask = get_segmentation_map(
        #     ds, row['lat'], row['lon'], self.radius)
        # mask = xr.where(mask, row['label'], 0)
        # mask = torch.from_numpy(mask.values).long()

        if self.transform is not None:
            data, mask = self.transform(data, mask)

        # Replace nan with 0 because U and V have negative numbers, but none of it is 0
        data = torch.nan_to_num(data, nan=0)

        # Data has shape (NUM_CLASSES, 80, 80)
        if self.metadata:
            metadata = {
                'data': data,
                'mask': mask,
                'file_name': row['file_name'],
                'lat': row['lat'],
                'lon': row['lon'],
                'label': row['label'],
                'idx': idx
            }
            return metadata

        return data, mask, binary_mask

    def get_weights_pixels(self, num_classes: int) -> torch.Tensor:
        counts = torch.zeros(num_classes, dtype=torch.float32)

        for batch in self:
            counts += torch.bincount(batch[1].flatten(), minlength=num_classes).float() # type: ignore

        weights = 1.0 / (torch.sqrt(counts) + self.epsilon)
        return weights

    def get_weights_class(self, num_classes: int, indices=None) -> List[float]:
        counts = torch.zeros(num_classes, dtype=torch.float32)

        if indices == None:
            indices = range(len(self))

        class_weights = 1.0 / (counts + self.epsilon)
        sample_weights = []
        for idx in indices:
            label = int(self.annotations.iloc[idx]['label'])
            sample_weights.append(class_weights[label])
        return sample_weights
