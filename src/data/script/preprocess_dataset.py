import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
from Create_AdditionalAscatSwaths import Create_AdditionalAscatSwaths

file_path = Path(__file__).resolve()
src_dir = file_path.parents[2]
sys.path.append(str(src_dir))

from utils.utils import dist_bwt_two_points
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

MEDICANES = [
    1328,
    1461,
    1542,
    1575,
    1622,
    1702,
]

OVER_LAND = [
    848, 849, 860, 864, 865, 868, 871, 873, 889, 900, 907, 926, 940, 943, 950,
    951, 969, 981, 985, 988, 992, 1001, 1018, 1023, 1025, 1030, 1033, 1034,
    1041, 1043, 1060, 1064, 1072, 1089, 1094, 1107, 1112, 1126, 1147, 1154,
    1166, 1180, 1185, 1199, 1206, 1214, 1215, 1223, 1225, 1232, 1257, 1262,
    1264, 1273, 1274, 1276, 1306, 1308, 1317, 1318, 1322, 1327, 1341, 1365,
    1392, 1404, 1405, 1408, 1414, 1420, 1441, 1450, 1452, 1453, 1456, 1484,
    1491, 1492, 1496, 1497, 1533, 1535, 1562, 1563, 1568, 1569, 1570, 1583,
    1595, 1596, 1603, 1605, 1608, 1612, 1614, 1615, 1616, 1625, 1648, 1649,
    1650, 1651, 1664, 1666, 1686, 1700,
]

EXCLUDE = [
    "20101010190801_track00001282_ASCATA-L2-ICM.nc",
    "20110129093048_track00001297_ASCATA-L2-ICM.nc",
    "20161104091411_track00001543_ASCATB-L2-ICM.nc",
    "20161121082237_track00001544_ASCATB-L2-ICM.nc",
    "20161221210152_track00001548_ASCATB-L2-ICM.nc",
    "20161221191946_track00001549_ASCATB-L2-ICM.nc",
    "20170226081530_track00001560_ASCATB-L2-ICM.nc",
    "20170226193400_track00001560_ASCATB-L2-ICM.nc",
    "20170308080733_track00001561_ASCATB-L2-ICM.nc",
    "20180111095218_track00001581_ASCATB-L2-ICM.nc",
    "20180112093226_track00001581_ASCATB-L2-ICM.nc",
    "20180126192100_track00001585_ASCATB-L2-ICM.nc",
    "20180214094856_track00001590_ASCATB-L2-ICM.nc",
    "20180305202020_track00001594_ASCATA-L2-ICM.nc",
    "20180322080620_track00001599_ASCATA-L2-ICM.nc",
    "20180409191148_track00001602_ASCATB-L2-ICM.nc",
    "20181023081418_track00001626_ASCATB-L2-ICM.nc",
    "20181023193239_track00001626_ASCATB-L2-ICM.nc",
    "20181118091600_track00001629_ASCATB-L2-ICM.nc",
    "20181120083403_track00001632_ASCATB-L2-ICM.nc",
    "20181206080352_track00001634_ASCATB-L2-ICM.nc",
    "20181206192246_track00001634_ASCATB-L2-ICM.nc",
    "20190104080356_track00001638_ASCATB-L2-ICM.nc",
    "20190114191724_track00001640_ASCATB-L2-ICM.nc",
    "20190126084833_track00001642_ASCATB-L2-ICM.nc",
    "20190205200224_track00001644_ASCATB-L2-ICM.nc",
    "20190214074743_track00001645_ASCATA-L2-ICM.nc",
    "20190217064503_track00001645_ASCATA-L2-ICM.nc",
    "20190217075409_track00001645_ASCATB-L2-ICM.nc",
    "20190327192724_track00001652_ASCATB-L2-ICM.nc",
    "20190408184043_track00001655_ASCATB-L2-ICM.nc",
    "20190504082345_track00001660_ASCATB-L2-ICM.nc",
    "20191003091913_track00001670_ASCATB-L2-ICM.nc",
    "20191024072933_track00001672_ASCATA-L2-ICM.nc",
    "20191024081037_track00001672_ASCATC-L2-ICM.nc",
    "20191027071003_track00001672_ASCATC-L2-ICM.nc",
    "20191107201413_track00001673_ASCATB-L2-ICM.nc",
    "20191107082101_track00001673_ASCATC-L2-ICM.nc",
    "20191120092335_track00001676_ASCATB-L2-ICM.nc",
    "20191118205431_track00001676_ASCATC-L2-ICM.nc",
    "20200109071101_track00001681_ASCATB-L2-ICM.nc",
    "20200109080946_track00001681_ASCATC-L2-ICM.nc",
    "20200117102150_track00001682_ASCATC-L2-ICM.nc",
    "20200331080800_track00001689_ASCATC-L2-ICM.nc",
    "20200415075705_track00001692_ASCATC-L2-ICM.nc",
    "20200420090637_track00001694_ASCATA-L2-ICM.nc",
    "20200420205445_track00001694_ASCATC-L2-ICM.nc",
    "20201012084437_track00001706_ASCATA-L2-ICM.nc",
    "20201030073241_track00001707_ASCATA-L2-ICM.nc",
    "20201027192939_track00001707_ASCATB-L2-ICM.nc",
    "20201120083746_track00001710_ASCATA-L2-ICM.nc",
    "20201119204852_track00001710_ASCATC-L2-ICM.nc",
    "20201204082246_track00001713_ASCATB-L2-ICM.nc",
    "20201202193945_track00001713_ASCATC-L2-ICM.nc",
    "20201213195639_track00001715_ASCATB-L2-ICM.nc",
    "20201212193120_track00001715_ASCATC-L2-ICM.nc",
    "20201215070918_track00001715_ASCATC-L2-ICM.nc",
    "20201226202913_track00001716_ASCATB-L2-ICM.nc",
    "20180321092435_track00001598_ASCATB-L2-ICM.nc",
    "20180916063422_track00001620_ASCATA-L2-ICM.nc",
    "20181217081016_track00001637_ASCATA-L2-ICM.nc",
    "20190406080213_track00001654_ASCATB-L2-ICM.nc",
    "20201225193630_track00001716_ASCATA-L2-ICM.nc",
    "20190214190615_track00001645_ASCATA-L2-ICM.nc",
    "20191114192724_track00001675_ASCATB-L2-ICM.nc",
    "20191024192815_track00001672_ASCATC-L2-ICM.nc",
    "20191211201020_track00001678_ASCATB-L2-ICM.nc",
]


def check_within_swaths(
    ds: xr.Dataset,
    abs_path: str,
    query_lat: float,
    query_lon: float,
) -> bool:
    ascat_new = Create_AdditionalAscatSwaths(abs_path)

    lat = ascat_new.lat
    lon = ascat_new.lon

    distances = np.nan_to_num(
        dist_bwt_two_points(
            query_lat,
            query_lon,
            lat,
            lon,
        ),
        nan=np.inf,
    )

    index = np.unravel_index(
        np.argmin(distances),
        lon.shape,
    )

    if np.isnan(ascat_new.wind_speed[index]):
        return False

    min_lat = ascat_new.lat[index]
    min_lon = ascat_new.lon[index]

    ori_lat = ds["lat"].values
    ori_lon = ds["lon"].values

    return bool((min_lat in ori_lat) and (min_lon in ori_lon))


def process_file(
    row,
    tracks_parent: str,
):
    cyclone_id = row["cyclone_id"]
    file_name = row["file_name"]
    file_path = row["file_path"]
    lat = row["lat"]
    lon = row["lon"]

    if pd.isna(lat) or pd.isna(lon):
        return None

    if not np.isfinite(lat) or not np.isfinite(lon):
        return None

    abs_path = os.path.normpath(
        os.path.join(
            tracks_parent,
            file_path,
        )
    )

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"ASCAT file does not exist:\n"
            f"file_path: {file_path}\n"
            f"absolute_path: {abs_path}\n"
            f"file_name: {file_name}\n"
            f"cyclone_id: {cyclone_id}"
        )

    try:
        with xr.open_dataset(
            abs_path,
            cache=False,
        ) as ds:
            if np.count_nonzero(~np.isnan(ds["wind_speed"].values)) == 0:
                return None

            if not check_within_swaths(
                ds,
                abs_path,
                lat,
                lon,
            ):
                return None

    except Exception as e:
        raise RuntimeError(
            f"Error processing file:\n"
            f"file_path: {file_path}\n"
            f"absolute_path: {abs_path}\n"
            f"file_name: {file_name}\n"
            f"cyclone_id: {cyclone_id}\n"
            f"error: {e}"
        ) from e

    return row.to_dict()


def preprocess(num_workers: int = 32) -> None:
    df = pd.read_csv(os.getenv("ANNOTATIONS_RAW_PATH"), sep="\t")

    tracks_parent = os.path.dirname(os.path.normpath(os.getenv("TRACKS_PATH")))

    rows_to_process = []
    result = []

    for _, row in df.iterrows():
        lat = row["lat"]
        lon = row["lon"]

        try:
            lat_val, lon_val = float(lat), float(lon)
            if not np.isfinite(lat_val) or not np.isfinite(lon_val):
                continue
        except (ValueError, TypeError):
            continue

        cyclone_id = row["cyclone_id"]
        file_name = row["file_name"]

        # Medicanes skip land and exclusion filters, but STILL undergo swath checks
        if cyclone_id not in MEDICANES:
            if cyclone_id in OVER_LAND:
                continue

            if file_name in EXCLUDE:
                continue

        rows_to_process.append(row)

    if rows_to_process:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            processed_rows = executor.map(
                lambda row: process_file(
                    row,
                    tracks_parent,
                ),
                rows_to_process,
            )

            result.extend(row for row in processed_rows if row is not None)

    result = pd.DataFrame(
        result,
        columns=COLUMNS,
    )

    result.to_csv(
        os.path.join(
            os.getenv("INTERIM_PATH"),
            "annotations_interim.txt",
        ),
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    import time

    start_time = time.perf_counter()
    preprocess()
    elapsed = time.perf_counter() - start_time
    print(f"Execution time: {elapsed:.4f} seconds")