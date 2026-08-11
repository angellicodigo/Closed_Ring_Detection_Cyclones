import argparse
from pathlib import Path
import pandas as pd
import xarray as xr
from pathlib import Path
import os
import sys

file_path = Path(__file__).resolve()
src_dir = file_path.parents[2]
sys.path.append(str(src_dir))

from utils.utils import calc_percent_over_ocean, get_num_points_over_ocean
import dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, ".env")
dotenv.load_dotenv(dotenv_path)

MEDICANES = [1328, 1461, 1542, 1575, 1622, 1702]


def filter(radius: float, threshold: float, num: int) -> None:
    """Filters cyclone tracks based on ocean point coverage surrounding their center.

    Args:
        radius (float): Distance (in km) around cyclone center to evaluate.
        threshold (float): Minimum required percentage of grid points over
          ocean.
        num (int): Minimum required absolute number of grid points over ocean.
    """
    columns = ["cyclone_id", "year", "file_name", "lat", "lon", "label"]
    result = pd.DataFrame(columns=columns)

    df = pd.read_csv(
        os.getenv("ANNOTATIONS_INTERIM_PATH"), sep=r"\t", engine="python"
    )

    tracks_dir = Path(os.getenv("TRACKS_PATH"))
    file_map = {f.name: str(f) for f in tracks_dir.rglob("*.nc")}

    for _, row in df.iterrows():
        input_data = {}
        if row["cyclone_id"] in MEDICANES:
            input_data = {
                "cyclone_id": row["cyclone_id"],
                "year": row["year"],
                "file_name": row["file_name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "label": row["label"],
            }
        else:
            file_name = row["file_name"]

            if file_name in file_map:
                file_path = file_map[file_name]
            elif "file_path" in row and pd.notna(row["file_path"]):
                tracks_parent = os.path.dirname(os.path.normpath(tracks_dir))
                file_path = os.path.normpath(
                    os.path.join(tracks_parent, row["file_path"])
                )
            else:                
                continue

            # Open NetCDF file and evaluate ocean point criteria within the target radius
            with xr.open_dataset(file_path) as ds:
                points_over_ocean = get_num_points_over_ocean(
                    ds, row["lat"], row["lon"], radius
                )
                percent_over_ocean = calc_percent_over_ocean(
                    ds, row["lat"], row["lon"], radius
                )

                # Keep row if both absolute count and percentage thresholds are met
                if (points_over_ocean >= num) and (
                    percent_over_ocean >= threshold
                ):
                    input_data = {
                        "cyclone_id": row["cyclone_id"],
                        "year": row["year"],
                        "file_name": row["file_name"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "label": row["label"],
                    }

        # Append valid candidate row to final results DataFrame
        if len(input_data) != 0:
            result.loc[len(result)] = input_data

    folder_path = os.path.join(os.getenv("PROCESSED_PATH"), "annotations.txt")
    result.to_csv(folder_path, index=False, sep="\t")

    print(f"How many files? {len(result)}")
    print(f'How many cyclones? {len(result["cyclone_id"].unique())}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radius",
        type=int,
        default=100,
        help="Search radius around cyclone center (in km).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=67,
        help="Minimum ocean percentage required within radius.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=109,
        help="Minimum count of ocean points required within radius.",
    )
    args = parser.parse_args()

    import time
    start_time = time.perf_counter()
    filter(args.radius, args.threshold, args.n)
    elapsed = time.perf_counter() - start_time
    print(f"Execution time: {elapsed:.4f} seconds")