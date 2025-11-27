# -*- coding: utf-8 -*-
"""
Created at University of Houston 

Observation extraction code for extraction of AIRKOREA__.txt files and saving
the values in .csv according to station name

@author - Deveshwar Singh
"""
import numpy as np
import pandas as pd
import sys
import os
import datetime
import torch
import calendar
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def main(num_workers=None):
    """
    Produce tensors with obs values patched to the O3 reanalysis ext data using multiprocessing.
    """
    if num_workers is None:
        num_workers = cpu_count()  # or a fixed value like 8

    misc_path = "../Misc"
    station_grid_file = os.path.join(misc_path, "station_with_nearest_grid.csv")
    station_grid = pd.read_csv(station_grid_file)

    
    storage_path = "/storage/dataset/ozone"
    # obs_txt_root = os.path.join(storage_path, "ObsData/korea/air")
    input_path_obs = os.path.join(storage_path , "NIER_AI_v8/Data/InputData/Fusion_Obs_Patched_ext/")
    input_path_Fusion = os.path.join(storage_path, "NIER_AI_v8/Data/InputData/Fusion_ext")
    obs_csv_root = os.path.join(storage_path, "NIER_AI_v8/Data/InputData/Obs_ext")

    task_args = []
    for year in range(2019, 2024):
        print(f"Preparing tasks for year {year}")
        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                task_args.append((year, month, day, input_path_obs, input_path_Fusion, obs_csv_root, station_grid_file))

    print(f"Processing {len(task_args)} days with {num_workers} workers")
    with Pool(processes=num_workers) as pool:
        pool.map(process_one_day, task_args)

    draw_weekly_o3_comparison_gif(
        start_date_str="20220720",
        reanalysis_root=input_path_Fusion,
        patched_root=input_path_obs,
        save_gif_path="New_Comparison_20220720_to_0727.gif"
    )
    return

def process_one_day(args):
    year, month, day, input_path_obs, input_path_reanalysis_CMAQ, obs_csv_root, station_grid_file = args
    date_str = f"{year}{month:02d}{day:02d}"
    reanalysis_tensor_path = os.path.join(input_path_reanalysis_CMAQ, f"{year}/{month:02d}/{day:02d}/{date_str}.pt")
    output_dir = os.path.join(input_path_obs, f"{year}/{month:02d}/{day:02d}")
    os.makedirs(output_dir, exist_ok=True)
    output_tensor_path = os.path.join(output_dir, f"{date_str}.pt")

    if not os.path.exists(reanalysis_tensor_path):
        print(f"Missing CMAQ tensor for {date_str}")
        return

    patch_cmaq_tensor_with_obs_new(
        date_str=date_str,
        tensor_path=reanalysis_tensor_path,
        obs_root=obs_csv_root,
        station_grid_csv=station_grid_file,
        save_path=output_tensor_path
    )

def patch_cmaq_tensor_with_obs_new(date_str, tensor_path, obs_root, station_grid_csv, save_path):
    """
    Replace CMAQ grid O3 values with observed O3 data from hourly obs CSVs.

    This version corrects for a 4-hour offset:
    - Assumes tensor file 'YYYYMMDD.pt' starts at 'YYYY-MM-DD 04:00:00 UTC'.
    - Assumes obs files 'obs_YYYYMMDDHH.csv' are named with UTC time.

    Parameters:
        date_str (str): Date in yyyymmdd format (e.g., '20160502').
        tensor_path (str): Path to full .pt CMAQ reanalysis data.
        obs_root (str): Root path to obs data (contains yearly/monthly/daily folders).
        station_grid_csv (str): Path to station-to-grid mapping CSV.
        save_path (str): Output path for patched O3 tensor.
    """
    tensor = torch.load(tensor_path)  # shape: [COL, ROW, TSTEP, VAR]
    o3_var_index = 0  # O3 variable index
    o3_tensor = tensor[:, :, :, o3_var_index].clone()

    # Load mapping
    station_grid = pd.read_csv(station_grid_csv)
    station_to_grid = {int(row["Station_ID"]): (int(row["COL"]), int(row["ROW"]))
                       for _, row in station_grid.iterrows()}

    # Get the UTC datetime corresponding to the *start* of the tensor (Hour 0)
    # This is 03:00 UTC on the given date_str
    try:
        tensor_start_utc = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(hours=3)
    except ValueError:
        print(f"Invalid date_str: {date_str}")
        return

    # Dict to accumulate per grid cell across multiple stations
    grid_hour_values = {h: {} for h in range(24)}  # h -> {(c,r): [values...]}

    # Loop over the 24 tensor time steps (0-23)
    for tensor_hour in range(24):
        # Calculate the actual UTC datetime for this tensor step
        obs_utc_datetime = tensor_start_utc + datetime.timedelta(hours=tensor_hour)

        # Format this datetime to find the correct obs file
        obs_year = obs_utc_datetime.strftime("%Y")
        obs_month = obs_utc_datetime.strftime("%m")
        obs_day = obs_utc_datetime.strftime("%d")
        obs_hour_str = obs_utc_datetime.strftime("%H") # "00" to "23"
        obs_date_str = obs_utc_datetime.strftime("%Y%m%d")

        # Obs files location: yearly/monthly/daily/hourly
        daily_dir = os.path.join(obs_root, obs_year, obs_month, obs_day)

        # Define file name and path
        fname = f"obs_{obs_date_str}{obs_hour_str}.csv"
        fpath = os.path.join(daily_dir, fname)

        if not os.path.exists(fpath):
            # This is an expected warning if obs data is missing
            # print(f"Missing hourly obs file {fname} (for tensor {date_str} hour {tensor_hour})")
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        if "O3" not in df.columns or "STNID" not in df.columns:
            print(f"Invalid obs file format: {fname}")
            continue

        for _, row in df.iterrows():
            try:
                station_id = int(row["STNID"])
                o3_value = float(row["O3"])  # Already in ppb, not ppm
            except:
                continue # Skip if conversion fails

            if np.isnan(o3_value) or np.isinf(o3_value) or o3_value < 0:
                continue

            if station_id not in station_to_grid:
                continue

            c, r = station_to_grid[station_id]
            # Use tensor_hour as the key, as this is the index we will patch
            grid_hour_values[tensor_hour].setdefault((c, r), []).append(o3_value)

    # Apply averages to tensor
    for tensor_hour in range(24):
        for (c, r), vals in grid_hour_values[tensor_hour].items():
            if vals: # Ensure list is not empty
                avg_val = np.mean(vals)
                o3_tensor[c, r, tensor_hour] = avg_val

    # Save patched tensor
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(o3_tensor, save_path)


def draw_o3_comparison(before_path, after_path, hour_idx=12):
    """
    Draws 2D images of O₃ concentrations for a given hour before and after observation patching,
    using a shared color scale.

    Parameters:
        before_path (str): Path to original .pt tensor file (with full variables).
        after_path (str): Path to O₃-only patched tensor file.
        hour_idx (int): Which hour to visualize (default is 0 for first hour).
    """
    # Load tensors
    tensor = torch.load(before_path)         # [COL, ROW, TSTEP, VAR]
    o3_tensor_patched = torch.load(after_path)  # [COL, ROW, TSTEP]

    # Extract original O₃ data
    o3_var_index = 0
    o3_tensor_orig = tensor[:, :, :, o3_var_index]

    # Extract hour slice and convert to numpy [ROW, COL]
    orig = o3_tensor_orig[:, :, hour_idx].T.numpy()
    patched = o3_tensor_patched[:, :, hour_idx].T.numpy()

    # Shared color scale
    vmin = min(orig.min(), patched.min())
    vmax = max(orig.max(), patched.max())

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(orig, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Original CMAQ O₃ (Hour {hour_idx})")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(patched, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Patched O₃ with Observations (Hour {hour_idx})")
    plt.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    plt.savefig("before and after insertion")


def draw_weekly_o3_comparison_gif(start_date_str, reanalysis_root, patched_root, save_gif_path="comparison_week.gif"):
    """
    Creates a GIF comparing original and patched O₃ over a 7-day period (hourly resolution).

    Parameters:
        start_date_str (str): Start date in 'yyyymmdd' format.
        reanalysis_root (str): Root path to reanalysis .pt files.
        patched_root (str): Root path to patched .pt files (O₃-only).
        save_gif_path (str): Output path for the GIF.
    """
    start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d")
    frame_paths = []
    vmin, vmax = float('inf'), float('-inf')

    # First pass to determine global color scale
    for i in range(7):
        date = start_date + datetime.timedelta(days=i)
        y, m, d = date.year, date.month, date.day
        fname = f"{y}{m:02d}{d:02d}.pt"
        re_file = os.path.join(reanalysis_root, f"{y}/{m:02d}/{d:02d}", fname)
        patched_file = os.path.join(patched_root, f"{y}/{m:02d}/{d:02d}", fname)
        if not os.path.exists(re_file) or not os.path.exists(patched_file):
            print(f"Missing file(s) for {fname}, skipping day.")
            continue
        o3_orig = torch.load(re_file)[:, :, :, 0].numpy().transpose(2, 1, 0)
        o3_patch = torch.load(patched_file).numpy().transpose(2, 1, 0)
        vmin = min(vmin, o3_orig.min(), o3_patch.min())
        vmax = max(vmax, o3_orig.max(), o3_patch.max())

    os.makedirs("gif_frames", exist_ok=True)

    # Second pass to generate frames
    for i in range(7):
        date = start_date + datetime.timedelta(days=i)
        y, m, d = date.year, date.month, date.day
        fname = f"{y}{m:02d}{d:02d}.pt"
        re_file = os.path.join(reanalysis_root, f"{y}/{m:02d}/{d:02d}", fname)
        patched_file = os.path.join(patched_root, f"{y}/{m:02d}/{d:02d}", fname)
        if not os.path.exists(re_file) or not os.path.exists(patched_file):
            continue

        o3_orig = torch.load(re_file)[:, :, :, 0].numpy().transpose(2, 1, 0)
        o3_patch = torch.load(patched_file).numpy().transpose(2, 1, 0)

        for hour in range(24):
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            im0 = axs[0].imshow(o3_orig[hour], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axs[0].set_title(f"Original CMAQ O₃\n{y}-{m:02d}-{d:02d} Hour {hour}")
            plt.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(o3_patch[hour], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axs[1].set_title(f"Patched O₃ with Observations\n{y}-{m:02d}-{d:02d} Hour {hour}")
            plt.colorbar(im1, ax=axs[1])

            plt.tight_layout()
            frame_path = f"gif_frames/frame_{y}{m:02d}{d:02d}_{hour:02d}.png"
            plt.savefig(frame_path)
            plt.close(fig)
            frame_paths.append(frame_path)

    # Make GIF
    with imageio.get_writer(save_gif_path, mode='I', duration=0.5, loop=0) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    print(f"Weekly GIF saved to {save_gif_path}")

    # Cleanup
    for path in frame_paths:
        os.remove(path)
    os.rmdir("gif_frames")

if __name__ == "__main__":
    main()