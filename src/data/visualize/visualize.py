import argparse
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
import sys

file_path = Path(__file__).resolve()
src_dir = file_path.parents[2]
sys.path.append(str(src_dir))

from utils.utils import (
    calc_percent_over_ocean,
    dist_bwt_two_points,
    get_center,
    get_mean_info,
    get_num_points,
    get_segmentation_map,
    nearest_neighbors_indices,
)

import os
import dotenv

src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
dotenv_path = os.path.join(src_dir, ".env")
dotenv.load_dotenv(dotenv_path)


def find_file(file_path: Path) -> Path:
    if file_path.is_file():
        return file_path.resolve()

    tracks_path = os.getenv("TRACKS_PATH")

    if tracks_path is None:
        raise ValueError(
            "TRACKS_PATH is not set in the .env file."
        )

    tracks_path = Path(tracks_path)

    if not tracks_path.is_dir():
        raise FileNotFoundError(
            f"TRACKS_PATH does not exist:\n"
            f"{tracks_path}"
        )

    matches = list(
        tracks_path.rglob(file_path.name)
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find file:\n"
            f"{file_path.name}\n"
            f"under TRACKS_PATH:\n"
            f"{tracks_path}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple files with the name "
            f"{file_path.name}:\n"
            + "\n".join(
                str(match)
                for match in matches
            )
        )

    return matches[0].resolve()


def open_file(
    file_path: Path,
    query_lon: float | None,
    query_lat: float | None,
    radius: float,
    window_size: float,
) -> None:
    with xr.open_dataset(file_path) as ds:
        average_time, year, month, day = get_mean_info(ds)
        hour = average_time.hour

        if query_lon is None or query_lat is None:
            cyclone_id = int(
                file_path.name.split("_")[1][5:]
            )

            query_lat, query_lon = get_center(
                cyclone_id,
                year,
                month,
                day,
                average_time,
            )

        row_indices, col_indices = nearest_neighbors_indices(
            ds,
            query_lat,
            query_lon,
        )

        nearest_row = row_indices[0]
        nearest_col = col_indices[0]

        dimensions = list(ds.sizes)
        row_dim = dimensions[0]
        col_dim = dimensions[1]

        nearest_point = ds.isel(
            {
                row_dim: nearest_row,
                col_dim: nearest_col,
            }
        )

        plot(
            ds,
            query_lat,
            query_lon,
            year,
            month,
            day,
            hour,
            radius,
            window_size,
        )

        print(f"Name: {file_path.name}")
        print(
            f"Center (lat, lon): "
            f"{query_lat} {query_lon}"
        )
        print(
            f"Nearest Neighbor (lat, lon): "
            f"{nearest_point['lat'].values} "
            f"{nearest_point['lon'].values}"
        )

        distance = dist_bwt_two_points(
            query_lat,
            query_lon,
            nearest_point["lat"].values,
            nearest_point["lon"].values,
        )

        print(
            f"Distance from Nearest Neighbor: "
            f"{distance}"
        )
        print(
            f"Does nearest neighbor have wind speed? "
            f"{nearest_point['wind_speed'].values}"
        )
        print(
            f"Average time: {average_time}"
        )

        output_path = (
            Path(__file__).resolve().parent
            / f"{file_path.stem}.png"
        )

        plt.savefig(
            output_path,
            format="png",
            dpi=1200,
            bbox_inches="tight",
        )

        print(
            f"Saved image: {output_path}"
        )

        plt.close()


def plot(
    ds: xr.Dataset,
    query_lat: float,
    query_lon: float,
    year: int,
    month: int,
    day: int,
    hour: int,
    radius: float,
    window_size: float,
) -> None:
    plt.figure(figsize=(12, 12))

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    boundaries = np.arange(
        0,
        32.6,
        2.5,
    )

    cmap = plt.get_cmap("turbo")

    norm = plt.Normalize(
        vmin=boundaries.min(),
        vmax=boundaries.max(),
    )

    U = (
        ds["wind_speed"]
        * np.sin(
            np.radians(ds["wind_dir"])
        )
    )

    V = (
        ds["wind_speed"]
        * np.cos(
            np.radians(ds["wind_dir"])
        )
    )

    quiver = ax.quiver(
        ds["lon"],
        ds["lat"],
        U,
        V,
        ds["wind_speed"],
        cmap="turbo",
        transform=ccrs.PlateCarree(),
        scale=500,
        pivot="mid",
        norm=norm,
    )

    cbar = plt.colorbar(quiver)

    cbar.set_label(
        "Wind Speed (m/s)"
    )

    cbar.set_ticks(boundaries)

    ax.coastlines()

    ax.set_xticks(
        np.arange(
            round(query_lon - window_size),
            round(query_lon + window_size) + 1,
            1,
        ),
        crs=ccrs.PlateCarree(),
    )

    ax.set_yticks(
        np.arange(
            round(query_lat - window_size),
            round(query_lat + window_size) + 1,
            1,
        ),
        crs=ccrs.PlateCarree(),
    )

    plt.tight_layout()

    plt.xlim(
        round(query_lon - window_size),
        round(query_lon + window_size),
    )

    plt.ylim(
        round(query_lat - window_size),
        round(query_lat + window_size),
    )

    plt.plot(
        query_lon,
        query_lat,
        "x",
        markersize=12,
        color="purple",
        transform=ccrs.PlateCarree(),
    )

    if radius != 0:
        plot_semantic_segmentation(
            ds,
            query_lat,
            query_lon,
            radius,
            "grey",
        )

        num = get_num_points(
            ds,
            query_lat,
            query_lon,
            radius,
        )

        percent = calc_percent_over_ocean(
            ds,
            query_lat,
            query_lon,
            radius,
        )

        plt.title(
            f"{year}-{month}-{day} "
            f"{hour} UTC "
            f"(N={num}; "
            f"{round(percent, 1)}% over ocean)"
        )

    else:
        plt.title(
            f"{year}-{month}-{day} "
            f"{hour} UTC"
        )


def plot_semantic_segmentation(
    ds: xr.Dataset,
    query_lat: float,
    query_lon: float,
    radius: float,
    color: str,
) -> None:
    mask = get_segmentation_map(
        ds,
        query_lat,
        query_lon,
        radius,
    )

    mask_lats = (
        ds["lat"]
        .where(mask)
        .values
    )

    mask_lons = (
        ds["lon"]
        .where(mask)
        .values
    )

    plt.scatter(
        mask_lons,
        mask_lats,
        c=color,
        s=100,
        marker="o",
        alpha=0.75,
        transform=ccrs.PlateCarree(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query_lat",
        type=float,
    )

    parser.add_argument(
        "--query_lon",
        type=float,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--window_size",
        type=float,
        default=5,
    )

    args = parser.parse_args()

    file_path = find_file(
        args.file_path
    )

    open_file(
        file_path,
        args.query_lon,
        args.query_lat,
        args.radius,
        args.window_size,
    )