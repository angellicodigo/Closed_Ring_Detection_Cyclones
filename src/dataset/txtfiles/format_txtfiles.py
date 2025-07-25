import pandas as pd
import os
from functools import reduce

PATH_TXTFILES = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\txtFiles'
PATH_TEMP = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\temp.txt'
PATH_NEW_TXTFILES = r'C:\Users\angel\VSCode\ML-Detect-Closed-Ring-Medicanes\reformatted_txtfiles'


def check() -> None:
    for root, _, files in os.walk(PATH_TXTFILES):
        for name in files:
            file_path = os.path.join(root, name)
            with open(file_path, 'r') as infile:
                content = infile.read()
                repls = [
                    ('date (AAAAMMGG)',  'date(AAAAMMGG)'),
                    ('time (hh:mm)',     'time(hh:mm)'),
                    ('TRACKS_CL7 lat',   'TRACKS_CL7_lat'),
                    ('TRACKS_CL7 lon',   'TRACKS_CL7_lon'),
                    ('MeRCAD lat',       'MeRCAD_lat'),
                    ('MeRCAD lon',       'MeRCAD_lon'),
                    ('geo dist (km)',    'geo_dist(km)'),
                    ('RMW (km)',         'RMW(km)'),
                    ('Vmax (m/s)',       'Vmax(m/s)'),
                    ('closed ring',       'closed_ring'),
                    ('closed eye',       'closed_ring'),
                    ('out of bounds',    'NaN'),
                    ('--',               'NaN'),
                    ('9999.999',         'NaN'),
                    ('All-NaN slice',    'NaN'),
                    ('\tNaN\tNaN\t',     '\tNaN\t'),
                    ('\t',               ' '),
                ]
                result = reduce(lambda a, kv: a.replace(*kv), repls, content)

            with open(PATH_TEMP, 'w') as outfile:
                outfile.write(result)

            write(name)


def write(name: str) -> None:
    df = pd.read_csv(
        PATH_TEMP,
        sep=r'\s+',
        engine='python',
        na_values=['NaN']
    )
    if not df.empty:
        string = df.to_string(index=False)
    else:
        string = '  '.join(df.columns)  # Only the header should be inserted
    file_path = os.path.join(PATH_NEW_TXTFILES, name)
    with open(file_path, 'w') as f:
        f.write(string)


if __name__ == '__main__':
    check()
