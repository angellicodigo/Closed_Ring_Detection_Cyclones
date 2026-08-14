"""
The main function:

    ASCATnew = Create_AdditionalAscatSwaths(nc_file)

reads one ASCAT NetCDF-4 file and returns an ASCATnew object with:
    .lat
    .lon
    .wind_speed
    .wind_dir
    .time
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import h5py
import numpy as np
from geographiclib.geodesic import Geodesic

# Spatial conversion constants
DIST_KM = 12.5
EARTH_RADIUS_KM = 6371.0
NL = 13


@dataclass
class ASCATnew:
    lat: np.ndarray
    lon: np.ndarray
    wind_speed: np.ndarray
    wind_dir: np.ndarray
    time: np.ndarray


def Create_AdditionalAscatSwaths(nc_file: str | Path) -> ASCATnew:
    nc_path = Path(nc_file)

    # Helper function to extract, transpose, mask, and scale dataset variables
    def __read_netcdf_variable__(h5_file: h5py.File, name: str) -> np.ndarray:
        if name not in h5_file:
            available_keys = list(h5_file.keys())
            raise KeyError(
                f"Variable {name!r} not found in NetCDF file '{nc_path}'. "
                f"Available root keys: {available_keys}"
            )

        try:
            variable = h5_file[name]
            raw_data = variable[...]
            data = np.asarray(raw_data, dtype=np.float64)

            if data.ndim != 2:
                raise ValueError(
                    f"Variable {name!r} in '{nc_path}' expected 2-D array, "
                    f"got shape {data.shape} (raw shape: {raw_data.shape})."
                )

            # Reorient array axes to standard spatial dimension ordering
            data = data.T

            # Mask invalid fill values as NaNs
            if "_FillValue" in variable.attrs:
                fill_value = float(np.asarray(variable.attrs["_FillValue"]).item())
                data[np.isclose(data, fill_value) | (data == fill_value)] = np.nan

            # Apply linear scaling and offset transformations if specified
            if "scale_factor" in variable.attrs or "add_offset" in variable.attrs:
                scale = float(variable.attrs.get("scale_factor", 1.0))
                offset = float(variable.attrs.get("add_offset", 0.0))
                data = data * scale + offset

            return data

        except Exception as e:
            if isinstance(e, (KeyError, ValueError)):
                raise
            raise RuntimeError(
                f"Failed to read/transform variable {name!r} from file '{nc_path}'. "
                f"Error: {e}"
            ) from e

    # Helper function to load all target fields from the file
    def __load__(nc_file_path: Path) -> Dict[str, np.ndarray]:
        names = ("lat", "lon", "wind_speed", "wind_dir", "time")
        if not nc_file_path.exists():
            raise FileNotFoundError(f"NetCDF file does not exist at path: '{nc_file_path}'")

        try:
            with h5py.File(nc_file_path, "r") as h5_file:
                return {name: __read_netcdf_variable__(h5_file, name) for name in names}
        except Exception as e:
            if isinstance(e, (FileNotFoundError, KeyError, ValueError)):
                raise
            raise RuntimeError(
                f"Failed to open or parse HDF5/NetCDF-4 file at '{nc_file_path}'. "
                f"File size: {nc_file_path.stat().st_size if nc_file_path.exists() else 'N/A'} bytes. "
                f"Error: {e}"
            ) from e

    # Helper function to compute forward azimuths between points on the WGS84 ellipsoid
    def __azimuth_wgs84__(
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        lat1 = np.asarray(lat1, dtype=np.float64)
        lon1 = np.asarray(lon1, dtype=np.float64)
        lat2 = np.asarray(lat2, dtype=np.float64)
        lon2 = np.asarray(lon2, dtype=np.float64)

        if not (lat1.shape == lon1.shape == lat2.shape == lon2.shape):
            raise ValueError(
                f"All azimuth input arrays must have matching shapes. Got shapes: "
                f"lat1={lat1.shape}, lon1={lon1.shape}, lat2={lat2.shape}, lon2={lon2.shape} "
                f"when processing file '{nc_path}'."
            )

        result = np.full(lat1.shape, np.nan, dtype=np.float64)
        valid = (
            np.isfinite(lat1)
            & np.isfinite(lon1)
            & np.isfinite(lat2)
            & np.isfinite(lon2)
        )

        geodesic = Geodesic.WGS84
        valid_indices = np.flatnonzero(valid)

        for index in valid_indices:
            try:
                result[index] = (
                    geodesic.Inverse(
                        float(lat1[index]),
                        float(lon1[index]),
                        float(lat2[index]),
                        float(lon2[index]),
                    )["azi1"]
                    % 360.0
                )
            except Exception as e:
                raise ValueError(
                    f"Geodesic inverse azimuth computation failed at index {index} in file '{nc_path}'. "
                    f"Input coordinates: lat1={lat1[index]}, lon1={lon1[index]}, "
                    f"lat2={lat2[index]}, lon2={lon2[index]}. Error: {e}"
                ) from e

        return result

    # Helper function to project destination coordinates given starting points, distance, and bearing
    def __reckon__(
        lat1: np.ndarray,
        lon1: np.ndarray,
        arclen_deg: float,
        azimuth_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            lat1 = np.asarray(lat1, dtype=np.float64)
            lon1 = np.asarray(lon1, dtype=np.float64)

            phi1 = np.deg2rad(lat1)
            lambda1 = np.deg2rad(lon1)
            azimuth = np.deg2rad(azimuth_deg)
            arc = np.deg2rad(arclen_deg)

            phi2 = np.arcsin(
                np.sin(phi1) * np.cos(arc)
                + np.cos(phi1) * np.sin(arc) * np.cos(azimuth)
            )

            lambda2 = lambda1 + np.arctan2(
                np.sin(azimuth) * np.sin(arc) * np.cos(phi1),
                np.cos(arc) - np.sin(phi1) * np.sin(phi2),
            )

            lat2 = np.rad2deg(phi2)
            lon2 = ((np.rad2deg(lambda2) + 180.0) % 360.0) - 180.0

            return lat2, lon2
        except Exception as e:
            raise ValueError(
                f"Reckon projection calculation failed for file '{nc_path}'. "
                f"Inputs -> arclen_deg: {arclen_deg}, azimuth_deg: {azimuth_deg}, "
                f"lat1 shape: {lat1.shape}, lon1 shape: {lon1.shape}. Error: {e}"
            ) from e

    # Helper function to calculate average valid azimuth across points
    def __mean_azimuth__(
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> float:
        try:
            azimuth = __azimuth_wgs84__(lat1, lon1, lat2, lon2)
            valid_count = np.count_nonzero(np.isfinite(azimuth))
            if valid_count == 0:
                raise ValueError(
                    f"Could not calculate a valid mean azimuth (all azimuth entries were NaN or infinite). "
                    f"Input vector sizes -> lat1 finite count: {np.count_nonzero(np.isfinite(lat1))}/{lat1.size}, "
                    f"lat2 finite count: {np.count_nonzero(np.isfinite(lat2))}/{lat2.size}."
                )
            return float(np.nanmean(azimuth))
        except Exception as e:
            raise ValueError(
                f"Failed to compute mean azimuth for file '{nc_path}'. Details: {e}"
            ) from e

    def __add_top_padding__(array: np.ndarray) -> np.ndarray:
        try:
            return np.vstack(
                (
                    np.full((gap, array.shape[1]), np.nan),
                    array[:-gap, :],
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed adding top padding in file '{nc_path}'. "
                f"Array shape: {array.shape}, gap size: {gap}. Error: {e}"
            ) from e

    def __add_bottom_padding__(array: np.ndarray) -> np.ndarray:
        try:
            return np.vstack(
                (
                    array[gap:, :],
                    np.full((gap, array.shape[1]), np.nan),
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed adding bottom padding in file '{nc_path}'. "
                f"Array shape: {array.shape}, gap size: {gap}. Error: {e}"
            ) from e

    # Crop array edges and introduce a 26-row blank section at the gap location
    def __reshape_swath__(array: np.ndarray) -> np.ndarray:
        try:
            return np.vstack(
                (
                    array[13 : ind_g + 1, :],
                    np.full((26, array.shape[1]), np.nan),
                    array[ind_g + 1 : -13, :],
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed reshaping swath in file '{nc_path}'. "
                f"Array shape: {array.shape}, ind_g index: {ind_g}. Error: {e}"
            ) from e

    # Read data from file into working arrays
    variables = __load__(nc_path)

    lat = variables["lat"].copy()
    lon = variables["lon"].copy()
    wind_speed = variables["wind_speed"].copy()
    wind_dir = variables["wind_dir"].copy()
    time = variables["time"].copy()

    if lat.ndim != 2:
        raise ValueError(
            f"File '{nc_path}': Expected 2-D lat array, got shape {lat.shape}."
        )

    # Identify valid non-NaN columns across the grid
    valid_cols = np.flatnonzero(~np.isnan(lat).all(axis=0))

    if valid_cols.size == 0:
        raise ValueError(
            f"File '{nc_path}': Entire latitude grid contains only NaNs. Shape: {lat.shape}."
        )

    # Dynamically select the first column containing valid data
    ref_col = int(valid_cols[0])

    # Identify valid non-NaN indices along the reference column
    idx = np.flatnonzero(~np.isnan(lat[:, ref_col]))

    if idx.size == 0:
        total_valid = np.count_nonzero(~np.isnan(lat))
        raise ValueError(
            f"File '{nc_path}': No valid latitude values found in reference column lat[:, {ref_col}]. "
            f"Lat array shape: {lat.shape}. Total valid non-NaN values across whole grid: {total_valid}."
        )

    # Apply row padding at the top or bottom depending on data alignment
    try:
        if idx[0] < 26:
            gap = 26 - int(idx[0])
            lat = __add_top_padding__(lat)
            lon = __add_top_padding__(lon)
            wind_speed = __add_top_padding__(wind_speed)
            wind_dir = __add_top_padding__(wind_dir)
            time = __add_top_padding__(time)

        elif idx[-1] > 134:
            gap = int(idx[-1]) - 134
            lat = __add_bottom_padding__(lat)
            lon = __add_bottom_padding__(lon)
            wind_speed = __add_bottom_padding__(wind_speed)
            wind_dir = __add_bottom_padding__(wind_dir)
            time = __add_bottom_padding__(time)
    except Exception as e:
        raise RuntimeError(
            f"File '{nc_path}': Failed during initial swath padding operation. "
            f"idx[0]={idx[0]}, idx[-1]={idx[-1]}, lat shape={lat.shape}. Error: {e}"
        ) from e

    # Re-evaluate valid row indices after initial padding using reference column
    idx = np.flatnonzero(~np.isnan(lat[:, ref_col]))

    # Locate spatial discontinuity boundary in the latitude array
    lat_diffs = np.abs(np.diff(lat[:, ref_col]))
    ind_gaps = np.flatnonzero(lat_diffs > 1)

    if ind_gaps.size != 1:
        max_diff = float(np.nanmax(lat_diffs)) if lat_diffs.size > 0 else "N/A"
        raise ValueError(
            f"File '{nc_path}': Expected exactly one latitude gap where abs(diff(lat[:, {ref_col}])) > 1. "
            f"Found {ind_gaps.size} gap(s) at indices: {ind_gaps.tolist()}. "
            f"Max absolute lat diff found was {max_diff}. Valid index range: [{idx[0]}, {idx[-1]}]."
        )

    ind_g = int(ind_gaps[0])

    try:
        lat_new = __reshape_swath__(lat)
        lon_new = __reshape_swath__(lon)
        wind_speed_new = __reshape_swath__(wind_speed)
        wind_dir_new = __reshape_swath__(wind_dir)
        time_new = __reshape_swath__(time)
    except Exception as e:
        raise RuntimeError(
            f"File '{nc_path}': Swath reshaping failed with ind_g={ind_g}. Error: {e}"
        ) from e

    # Recompute valid row indices and gap position on reshaped arrays
    idx_new = np.flatnonzero(~np.isnan(lat_new[:, ref_col]))
    idx_diffs = np.abs(np.diff(idx_new))
    ind_gaps_new = np.flatnonzero(idx_diffs > 1)

    if ind_gaps_new.size != 1:
        raise ValueError(
            f"File '{nc_path}': Expected exactly 1 gap in idx_new after reshaping, found {ind_gaps_new.size}: {ind_gaps_new.tolist()}. "
            f"idx_new length: {len(idx_new)}, lat_new shape: {lat_new.shape}."
        )

    ind_g_new = int(ind_gaps_new[0])

    # Compute step distance arc length in degrees
    arclen_deg = float(np.rad2deg(DIST_KM / EARTH_RADIUS_KM))

    # Extrapolate geographic coordinates preceding the start of the first swath segment
    try:
        az = __mean_azimuth__(
            lat[idx[1], :],
            lon[idx[1], :],
            lat[idx[0], :],
            lon[idx[0], :],
        )

        lat_in = lat[idx[1], :].copy()
        lon_in = lon[idx[1], :].copy()

        for i in range(1, NL + 1):
            target_row = idx_new[0] - i
            lat_out, lon_out = __reckon__(lat_in, lon_in, arclen_deg, az)
            lat_new[target_row, :] = lat_out
            lon_new[target_row, :] = lon_out
            lat_in, lon_in = lat_out, lon_out
    except Exception as e:
        raise RuntimeError(
            f"File '{nc_path}': Extrapolation 1 (preceding first segment start) failed. "
            f"idx[0]={idx[0]}, idx[1]={idx[1]}, idx_new[0]={idx_new[0]}. Error: {e}"
        ) from e

    # Extrapolate geographic coordinates extending after the first segment into the gap
    try:
        az = __mean_azimuth__(
            lat[ind_g - 1, :],
            lon[ind_g - 1, :],
            lat[ind_g, :],
            lon[ind_g, :],
        )

        lat_in = lat[ind_g - 1, :].copy()
        lon_in = lon[ind_g - 1, :].copy()

        for i in range(1, NL + 1):
            target_row = idx_new[ind_g_new] + i
            lat_out, lon_out = __reckon__(lat_in, lon_in, arclen_deg, az)
            lat_new[target_row, :] = lat_out
            lon_new[target_row, :] = lon_out
            lat_in, lon_in = lat_out, lon_out
    except Exception as e:
        raise RuntimeError(
            f"File '{nc_path}': Extrapolation 2 (extending into swath gap) failed. "
            f"ind_g={ind_g}, ind_g_new={ind_g_new}, base_idx={idx_new[ind_g_new]}. Error: {e}"
        ) from e

    # Extrapolate geographic coordinates preceding the start of the second swath segment
    try:
        az = __mean_azimuth__(
            lat[ind_g + 2, :],
            lon[ind_g + 2, :],
            lat[ind_g + 1, :],
            lon[ind_g + 1, :],
        )

        lat_in = lat[ind_g + 2, :].copy()
        lon_in = lon[ind_g + 2, :].copy()

        for i in range(1, NL + 1):
            target_row = idx_new[ind_g_new + 1] - i
            lat_out, lon_out = __reckon__(lat_in, lon_in, arclen_deg, az)
            lat_new[target_row, :] = lat_out
            lon_new[target_row, :] = lon_out
            lat_in, lon_in = lat_out, lon_out
    except Exception as e:
        raise RuntimeError(
            f"File '{nc_path}': Extrapolation 3 (preceding second segment start) failed. "
            f"ind_g={ind_g}, ind_g_new={ind_g_new}, base_idx={idx_new[ind_g_new + 1]}. Error: {e}"
        ) from e

    # Extrapolate geographic coordinates extending past the end of the final swath segment
    try:
        az = __mean_azimuth__(
            lat[idx[-2], :],
            lon[idx[-2], :],
            lat[idx[-1], :],
            lon[idx[-1], :],
        )

        lat_in = lat[idx[-2], :].copy()
        lon_in = lon[idx[-2], :].copy()

        for i in range(1, NL + 1):
            target_row = idx_new[-1] + i
            lat_out, lon_out = __reckon__(lat_in, lon_in, arclen_deg, az)
            lat_new[target_row, :] = lat_out
            lon_new[target_row, :] = lon_out
            lat_in, lon_in = lat_out, lon_out
    except Exception as e:
        raise RuntimeError(
            f"File '{nc_path}': Extrapolation 4 (past end of final segment) failed. "
            f"idx[-2]={idx[-2]}, idx[-1]={idx[-1]}, idx_new[-1]={idx_new[-1]}. Error: {e}"
        ) from e

    # Package output into container object
    return ASCATnew(
        lat=lat_new,
        lon=lon_new,
        wind_speed=wind_speed_new,
        wind_dir=wind_dir_new,
        time=time_new,
    )