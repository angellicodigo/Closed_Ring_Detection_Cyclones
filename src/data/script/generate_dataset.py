import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

file_path = Path(__file__).resolve()
src_dir = file_path.parents[2]
sys.path.append(str(src_dir))

from utils.utils import get_center, get_mean_info
import os
import dotenv

src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
dotenv_path = os.path.join(src_dir, ".env")
dotenv.load_dotenv(dotenv_path)

COLUMNS = [
    "cyclone_id",
    "year",
    "file_name",
    "lat",
    "lon",
    "label",
    "file_path",
]


def generate_dataset(num_workers: int = 64) -> None:
    """
    Generate annotations_raw.txt.

    The function:

    1. Loads the existing annotation template.
    2. Walks through TRACKS_PATH and finds ASCAT files.
    3. Ignores files already present in the template.
    4. Processes all newly discovered ASCAT files.
    5. Uses multiple threads to process files concurrently.
    6. Combines the template rows with the newly discovered rows.
    7. Saves the result as annotations_raw.txt.
    """

    def __process_file__(abs_path: str, cyclone_id: int, file_name: str, file_path: str) -> dict:
        """
        Process one ASCAT NetCDF file.

        The cyclone center is calculated using get_center().
        Newly discovered files receive label=np.nan because
        they have not been manually labeled yet.

        If the file cannot be processed, the error is raised
        with information about the file that caused the error.
        """

        try:
            # cache=False prevents xarray from keeping the file
            # in its cache, which is useful when processing many
            # files concurrently.
            with xr.open_dataset(abs_path, cache=False) as ds:
                average_time, year, month, day = get_mean_info(ds)
                lat, lon = get_center(
                    cyclone_id,
                    year,
                    month,
                    day,
                    average_time,
                )

        except Exception as e:
            raise RuntimeError(f"""
                Error processing file:\n
                file_path: {file_path}\n
                cyclone_id: {cyclone_id}\n
                file_name: {file_name}\n
                error: {e}
                """
            ) from e

        return {
            "cyclone_id": cyclone_id,
            "year": year,
            "file_name": file_name,
            "lat": lat,
            "lon": lon,
            "label": np.nan,
            "file_path": file_path,
        }

    template_df = pd.read_csv(os.getenv("ANNOTATIONS_TEMPLATE_PATH"), sep="\t")

    # Create a set of filenames already present in the template.
    # A set provides fast O(1) lookup when checking thousands
    # of files under Tracks.
    template_file_names = set(
        template_df["file_name"].astype(str)
    )

    # Get the parent directory of Tracks.
    # This is used to create relative file paths such as:
    # Tracks/track00001291/ASCATA-L2-ICM/2010/...
    tracks_parent = os.path.dirname(
        os.path.normpath(os.getenv("TRACKS_PATH"))
    )

    # Store information about every newly discovered ASCAT file
    # before sending the files to the worker threads.
    candidate_files = []

    # Recursively walk through the entire Tracks directory.
    for root, _, files in os.walk(os.getenv("TRACKS_PATH")):
        for file_name in files:
            if "ASCAT" not in file_name:
                continue

            if not file_name.endswith(".nc"):
                continue

            if file_name in template_file_names:
                continue

            abs_path = os.path.join(root, file_name)
            rel_file_path = os.path.relpath(
                abs_path,
                start=tracks_parent,
            ).replace(
                os.sep,
                "/",
            )

            # Extract the cyclone ID from the filename.
            #
            # Example:
            # 20101212190242_track00001291_ASCATA-L2-ICM.nc
            #
            # file_name.split("_")[1] -> track00001291
            # [5:] -> 00001291
            # int(...) -> 1291
            try:
                cyclone_id = int(
                    file_name.split("_")[1][5:]
                )
            except Exception as e:
                raise ValueError(f"""
                    Could not extract cyclone_id from file:\n
                    file_path: {rel_file_path}\n
                    error: {e}
                    """
                ) from e

            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"""
                    Discovered file does not exist:\n
                    file_path: {rel_file_path}\n
                    absolute_path: {abs_path}
                    """
                )

            # Save everything needed to process this file.
            candidate_files.append(
                (
                    abs_path,
                    cyclone_id,
                    file_name,
                    rel_file_path,
                )
            )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        new_results = list(
            executor.map(
                lambda candidate: __process_file__(
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3],
                ),
                candidate_files,
            )
        )

    # Make sure every discovered file produced a result.
    if len(new_results) != len(candidate_files):
        raise RuntimeError(f"""
            The number of processed files does not match the number of discovered files.
            Discovered: {len(candidate_files)}\n
            Processed: {len(new_results)}
            """
        )

    new_df = pd.DataFrame(
        new_results,
        columns=COLUMNS,
    )

    result = pd.concat(
        [
            template_df,
            new_df,
        ],
        ignore_index=True,
    )

    output_path = os.path.join(os.getenv("RAW_PATH"), "annotations_raw.txt")

    result.to_csv(
        output_path,
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    import time
    start_time = time.perf_counter()
    generate_dataset()
    elapsed = time.perf_counter() - start_time
    print(f"Execution time: {elapsed:.4f} seconds")