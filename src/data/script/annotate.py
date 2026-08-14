import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from Create_AdditionalAscatSwaths import Create_AdditionalAscatSwaths
import os
import dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, ".env")
dotenv.load_dotenv(dotenv_path)


def resolve_nc_path(row: pd.Series, base_dir: Path) -> Path | None:
    p_rel = Path(str(row.get('file_path', '')))
    p_name = Path(str(row.get('file_name', '')))
    
    if p_rel.is_file():
        return p_rel.resolve()
    
    cand1 = base_dir / p_rel
    if cand1.is_file():
        return cand1.resolve()
        
    cand2 = base_dir / p_name
    if cand2.is_file():
        return cand2.resolve()
        
    tracks_env = os.getenv("TRACKS_PATH")
    if tracks_env:
        t_path = Path(tracks_env)
        if t_path.is_dir():
            matches = list(t_path.rglob(p_name.name))
            if matches:
                return matches[0].resolve()
    return None


def annotate(args):
    input_path = args.file_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_path = input_path.parent / "annotations_label.txt"

    # Load existing output file if it exists, otherwise load original input
    load_path = out_path if out_path.is_file() else input_path
    df = pd.read_csv(load_path, sep='\t', dtype={'label': str}, keep_default_na=True)

    # Clean existing labels and handle empty/missing values
    if 'label' not in df.columns:
        df['label'] = np.nan
    else:
        df['label'] = df['label'].astype(str).str.strip()
        unlabeled_mask = df['label'].isin(['nan', 'NaN', 'None', '', '<NA>'])
        df.loc[unlabeled_mask, 'label'] = np.nan

    # Target indices: cycle through unlabeled rows
    target_indices = df[df['label'].isna()].index.tolist()
    if not target_indices:
        target_indices = df.index.tolist()

    curr_pos = [0]

    fig = plt.figure(figsize=(13, 8))
    
    # Pre-allocated fixed axes for map and colorbar to prevent layout shifts
    ax = fig.add_axes([0.08, 0.18, 0.76, 0.74], projection=ccrs.PlateCarree())
    cax = fig.add_axes([0.86, 0.18, 0.025, 0.74])

    # 5 Action Buttons
    btn_ax_0    = plt.axes([0.05, 0.04, 0.16, 0.08])
    btn_ax_1    = plt.axes([0.23, 0.04, 0.16, 0.08])
    btn_ax_x    = plt.axes([0.41, 0.04, 0.16, 0.08])
    btn_ax_prev = plt.axes([0.59, 0.04, 0.16, 0.08])
    btn_ax_next = plt.axes([0.77, 0.04, 0.16, 0.08])

    btn_0    = Button(btn_ax_0, 'Not Closed (0)', color='lightcoral', hovercolor='red')
    btn_1    = Button(btn_ax_1, 'Closed Ring (1)', color='lightgreen', hovercolor='green')
    btn_x    = Button(btn_ax_x, 'Discard (x)', color='khaki', hovercolor='gold')
    btn_prev = Button(btn_ax_prev, 'Previous', color='lightblue', hovercolor='blue')
    btn_next = Button(btn_ax_next, 'Next', color='lightblue', hovercolor='blue')

    def save_df():
        df.to_csv(out_path, sep='\t', index=False)

    def render():
        ax.clear()
        cax.clear()

        if not target_indices:
            ax.text(0.5, 0.5, "No dataset items available.", transform=ax.transAxes, ha='center')
            fig.canvas.draw()
            return

        row_idx = target_indices[curr_pos[0]]
        row = df.loc[row_idx]

        nc_file = resolve_nc_path(row, input_path.parent)
        
        lbl_val = str(row['label'])
        if lbl_val in ['0', '0.0']:
            lbl_str = "0"
        elif lbl_val in ['1', '1.0']:
            lbl_str = "1"
        elif lbl_val == 'x':
            lbl_str = "x"
        else:
            lbl_str = "Unlabeled"

        title_text = f"[{curr_pos[0] + 1}/{len(target_indices)}] Row Index: {row_idx} | File: {row['file_name']} | Current Label: {lbl_str}"

        if nc_file is None or not nc_file.is_file():
            ax.text(0.5, 0.5, f"NetCDF file missing:\n{row['file_name']}", transform=ax.transAxes, ha='center', va='center', color='red', fontsize=12)
            fig.suptitle(title_text)
            fig.canvas.draw()
            return

        with xr.open_dataset(nc_file) as ds:
            boundaries = np.arange(0, 32.6, 2.5)
            cmap = plt.get_cmap("turbo")
            norm = BoundaryNorm(boundaries, ncolors=cmap.N)

            U = ds['wind_speed'] * np.sin(np.radians(ds['wind_dir']))
            V = ds['wind_speed'] * np.cos(np.radians(ds['wind_dir']))

            quiver = ax.quiver(
                ds['lon'], ds['lat'], U, V, ds['wind_speed'],
                cmap=cmap, transform=ccrs.PlateCarree(), scale=500, pivot='mid', norm=norm
            )
            
            cbar = fig.colorbar(quiver, cax=cax, orientation='vertical')
            cbar.set_label("Wind Speed (m/s)")
            cbar.set_ticks(boundaries)

            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False

            q_lat = float(row['lat'])
            q_lon = float(row['lon'])

            ax.set_xlim(q_lon - args.window_size, q_lon + args.window_size)
            ax.set_ylim(q_lat - args.window_size, q_lat + args.window_size)

            if args.extend:
                try:
                    ascat_ext = Create_AdditionalAscatSwaths(nc_file)
                    ax.plot(
                        ascat_ext.lon, ascat_ext.lat,
                        '.', color='black', markersize=3, alpha=0.6,
                        transform=ccrs.PlateCarree(), zorder=2
                    )
                except Exception as e:
                    print(f"Warning: Failed to compute extended swaths for {nc_file.name}: {e}")

            ax.plot(q_lon, q_lat, 'x', markersize=10, color='purple', markeredgewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)

        fig.suptitle(title_text, fontsize=11, fontweight='bold')
        fig.canvas.draw()

    def set_label(val: str):
        row_idx = target_indices[curr_pos[0]]
        df.at[row_idx, 'label'] = val
        save_df()
        
        if curr_pos[0] < len(target_indices) - 1:
            curr_pos[0] += 1

        render()

    def go_prev(event):
        if curr_pos[0] > 0:
            curr_pos[0] -= 1
            render()

    def go_next(event):
        if curr_pos[0] < len(target_indices) - 1:
            curr_pos[0] += 1
            render()

    btn_0.on_clicked(lambda e: set_label("0"))
    btn_1.on_clicked(lambda e: set_label("1"))
    btn_x.on_clicked(lambda e: set_label("x"))
    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)

    render()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=Path, required=True)
    parser.add_argument("--window_size", type=float, default=5.0)
    parser.add_argument("--extend", action="store_true", default=False)
    args = parser.parse_args()
    annotate(args)