from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import torch
import pandas as pd
import xarray as xr
import numpy as np
import sys

file_path = Path(__file__).resolve()
src_dir = file_path.parent.parent  
sys.path.append(str(src_dir))
from utils.utils import get_segmentation_map, nearest_neighbors_indices

class CycloneDataset(Dataset):
    def __init__(
        self,
        annotations_file: Path,
        data_dir: Path,
        radius=100,
        transform=None,
        augment=False,
        isClassification=False
    ):
        self.radius = radius
        self.transform = transform
        self.data = []
        self.epsilon = 1e-07
        self.isClassification = isClassification

        org_annotations = pd.read_csv(
            annotations_file,
            sep=r"\t",
            engine="python",
        )

        annotations = pd.DataFrame(columns=org_annotations.columns)

        for _, row in org_annotations.iterrows():
            file_path = data_dir / row["file_name"]

            with xr.open_dataset(file_path) as ds:
                i, j = nearest_neighbors_indices(ds, row["lat"], row["lon"])
                i, j = i[0], j[0]

                non_nan = ds["wvc_index"].notnull()
                ds = ds.where(non_nan, drop=True)

                row_dim = list(ds["lon"].sizes)[0]
                col_dim = list(ds["lon"].sizes)[1]

                # Resize dataset from (161 x 82) or (161 x 81) to (160 x 80)
                if i >= int(ds["lon"].shape[1] / 2):
                    ds = ds.drop_isel({col_dim: 0})
                    if ds["lon"].shape[1] == 81:
                        ds = ds.drop_isel({col_dim: 1})
                else:
                    ds = ds.drop_isel({col_dim: -1})
                    if ds["lon"].shape[1] == 81:
                        ds = ds.drop_isel({col_dim: -2})

                if j >= int(ds["lon"].shape[0] / 2):
                    ds = ds.drop_isel({row_dim: 0})
                else:
                    ds = ds.drop_isel({row_dim: -1})

                upper_limit = 0.85
                lower_limit = 0.15

                if (j > 39) and (j < 119):
                    dist_top = j
                    dist_bot = 159 - j

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

                else:  # j >= 119
                    indices = np.arange(80, 160)

                # Now 80 x 80 by keeping certain rows
                ds = ds.isel({row_dim: indices})

                ds["U"] = ds["wind_speed"] * np.sin(np.radians(ds["wind_dir"]))
                ds["V"] = ds["wind_speed"] * np.cos(np.radians(ds["wind_dir"]))

                self.data.append(ds.copy())
                annotations.loc[len(annotations)] = row.copy()

                if augment and (row["label"] == 1):
                    ds["U"] = -ds["U"]
                    ds["V"] = -ds["V"]
                    self.data.append(ds)
                    annotations.loc[len(annotations)] = row.copy()

        self.annotations = annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        ds = self.data[idx]

        data = torch.from_numpy(
            xr.concat([ds["U"], ds["V"]], dim="channel").values
        ).float()

        if row["label"] == 1:
            mask = get_segmentation_map(
                ds,
                row["lat"],
                row["lon"],
                self.radius,
            )
            mask = xr.where(mask, row["label"], 0)
            mask = torch.from_numpy(mask.values).long()
        else:
            mask = torch.zeros(
                (data.shape[1], data.shape[2]),
                dtype=torch.long,
            )

        binary_mask = torch.from_numpy(
            ~np.isnan(ds["wind_speed"].values)
        ).float().unsqueeze(0)

        if self.transform is not None:
            data, mask = self.transform(data, mask)

        # Replace NaNs with 0
        data = torch.nan_to_num(data, nan=0)

        if self.isClassification:
            return (
                data,
                torch.tensor(row["label"], dtype=torch.float).unsqueeze(0),
                binary_mask,
            )

        return data, mask, binary_mask

    # TODO: may remove
    def get_weights_pixels(self, num_classes: int, indices=None):
        counts = torch.zeros(num_classes, dtype=torch.float32)

        for batch in self:
            counts += torch.bincount(
                batch[1].flatten(),
                minlength=num_classes,
            ).float()

        weights = 1.0 / (torch.sqrt(counts) + self.epsilon)

        if indices is not None:
            sample_weights = []

            for idx in indices:
                label = int(self.annotations.iloc[idx]["label"])
                sample_weights.append(weights[label])

            return sample_weights

        return weights

    # TODO: May remove
    def get_weights_class(self, num_classes: int, indices=None) -> List[float]:
        counts = torch.zeros(num_classes, dtype=torch.float32)

        if indices is None:
            indices = range(len(self))

        for idx in indices:
            label = int(self.annotations.iloc[idx]["label"])
            counts[label] += 1

        class_weights = 1.0 / (counts + self.epsilon)

        sample_weights = []
        for idx in indices:
            label = int(self.annotations.iloc[idx]["label"])
            sample_weights.append(class_weights[label])

        return sample_weights

if __name__ == '__main__':
    import dotenv
    import os
    from pathlib import Path
    import time

    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    dotenv.load_dotenv(dotenv_path)

    annotation_file_str = os.getenv("ANNOTATION_FILE_PATH")
    dataset_path_str = os.getenv("DATASET_PATH")

    if not annotation_file_str or not dataset_path_str:
        raise ValueError("ANNOTATION_FILE_PATH or DATASET_PATH are not properly defined in the .env file.")

    ann_path = Path(annotation_file_str)
    data_dir = Path(dataset_path_str)

    # ==========================================
    # Test Dataset in Segmentation Mode (Default)
    # ==========================================
    print("\n" + "="*50)
    print("TESTING DATASET: Segmentation Mode")
    print("="*50)

    start_time = time.perf_counter()
    seg_dataset = CycloneDataset(
        annotations_file=ann_path,
        data_dir=data_dir,
        radius=100,
        isClassification=False
    )
    elapsed  = time.perf_counter() - start_time
    print(f"Dataset load time: {elapsed:.4f} seconds")
    print(f"Total samples loaded (Segmentation): {len(seg_dataset)}")

    if len(seg_dataset) > 0:
        data, mask, binary_mask = seg_dataset[0]
        print(f"  -> Data tensor shape (U, V channels): {data.shape}")
        print(f"  -> Segmentation mask shape: {mask.shape}")
        print(f"  -> Binary valid-data mask shape: {binary_mask.shape}")
        print(f"  -> Unique values in segmentation mask: {torch.unique(mask)}")

    # ==========================================
    # Test Dataset in Classification Mode
    # ==========================================
    print("\n" + "="*50)
    print("TESTING DATASET: Classification Mode")
    print("="*50)
    
    class_dataset = CycloneDataset(
        annotations_file=ann_path,
        data_dir=data_dir,
        radius=100,
        isClassification=True
    )
    
    print(f"Total samples loaded (Classification): {len(class_dataset)}")

    if len(class_dataset) > 0:
        data, label, binary_mask = class_dataset[0]
        print(f"  -> Data tensor shape (U, V channels): {data.shape}")
        print(f"  -> Classification label tensor: {label} (type: {label.dtype})")
        print(f"  -> Binary valid-data mask shape: {binary_mask.shape}")

    # ==========================================
    # Test Weight Calculation Utilities
    # ==========================================
    print("\n" + "="*50)
    print("TESTING UTILITY: Class Weight Calculation")
    print("="*50)
    
    # Assuming binary classification/segmentation (0 and 1)
    weights = class_dataset.get_weights_class(num_classes=2)
    print(f"Sample-wise weights computed for first 5 items: {weights[:5]}")