import torch
from torch.utils.data import DataLoader
from src.dataset.dataset import CycloneDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import os
from tqdm import tqdm
# import cartopy.crs as ccrs

PATH_TRUE = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\images\dataset\true'
PATH_FALSE = r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\images\dataset\false'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_dataset() -> None:
    dataset = CycloneDataset(r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\annotations_SS.txt', r'C:\Users\angel\VSCode\ML_for_Medicane_Wind_Rings\data\processed\dataset', metadata=True)
    training_samples = len(dataset)
    train_set, _ = torch.utils.data.random_split(dataset, [training_samples, len(dataset) - training_samples])
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=False)

    for batch in tqdm(train_loader, total=len(train_loader)):
        datas = batch['data'].to(device)
        masks = batch['mask'].to(device)
        datas_copy = datas.detach().cpu().numpy()[0]
        masks_copy = masks.detach().cpu().numpy()[0]

        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        U = datas_copy[0]
        V = datas_copy[1]
        magnitude = np.sqrt(U**2 + V**2)
        U[U == 0] = np.nan
        V[V == 0] = np.nan
        magnitude[magnitude == 0] = np.nan

        height, width = U.shape
        y, x = np.mgrid[0:height, 0:width]

        boundaries = np.arange(0, 32.6, 2.5)
        cmap = plt.get_cmap("turbo")
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)

        quiver = ax1.quiver(x, y, U, V, magnitude, angles='xy', cmap='turbo',
                            pivot='mid', scale=500, norm=norm)
        ax1.set_title("Wind Vectors of Dataloader")
        ax1.set_aspect('equal')
        cbar1 = fig.colorbar(quiver, ax=ax1)
        cbar1.set_ticks(boundaries) # type: ignore

        ds = dataset.data[batch['idx'][0]]
        U_true = ds['wind_speed'] * \
            np.sin(np.radians(ds['wind_dir']))
        V_true = ds['wind_speed'] * \
            np.cos(np.radians(ds['wind_dir']))

        # quiver = ax2.quiver(ds[index]['lon'], ds[index]['lat'], U_true, V_true, ds[index]
        #                     ['wind_speed'], transform=ccrs.PlateCarree(), cmap='turbo', pivot='mid', norm=norm)
        # ax2.plot(row['lon'], row['lat'], 'x', markersize=10,
        #          color="black", transform=ccrs.PlateCarree())
        quiver = ax2.quiver(
            x, y, U_true, V_true, ds['wind_speed'].values, angles='xy', cmap='turbo', pivot='mid', scale=500, norm=norm)
        ax2.set_title("Wind Vectors of original dataset")
        ax2.set_aspect('equal')
        cbar2 = fig.colorbar(quiver, ax=ax2)
        cbar2.set_ticks(boundaries) # type: ignore

        ax3.imshow(masks_copy, aspect='equal', origin='lower')
        ax3.set_aspect('equal')
        label = batch['label'][0]
        ax3.set_title(f"Segmentation Mask with label {label}")

        plt.tight_layout()
        file_name = batch['file_name'][0]
        if label == 1:
            folder_path = os.path.join(PATH_TRUE, f"{file_name}.png")
        else:
            folder_path = os.path.join(PATH_FALSE, f"{file_name}.png")

        plt.savefig(folder_path, dpi=300)
        plt.close()


if __name__ == '__main__':
    plot_dataset()
