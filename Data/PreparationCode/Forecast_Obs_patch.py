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
from collections import defaultdict
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import warnings

def main(Path):
    global input_path

    misc_path = "../Misc"
    station_grid_file = os.path.join(misc_path, "station_with_nearest_grid.csv")
    station_grid = pd.read_csv(station_grid_file)

    input_path_forecast_CMAQ = os.path.join(Path, "Data/InputData/Forecast_CMAQ_ext")
    obs_csv_root = os.path.join(Path, "Data/InputData/Obs_ext")
    output_path_forecast_CMAQ = os.path.join(Path, "Data/InputData/Forecast_CMAQ_Obs_ext")

    station_row_col_dict = dict(
        zip(
            station_grid["Station_ID"],
            zip(station_grid["ROW"], station_grid["COL"])
        )
    )

    obs_cache = {}

    patch_start_date = '20190101'
    patch_end_date = '20231231'
    date_range = pd.date_range(start=patch_start_date, end=patch_end_date, freq='D')

    # Cache initialization
    start_date = patch_start_date
    start_time = datetime.datetime.strptime(start_date, "%Y%m%d").replace(hour=3) # Adjust to 3AM UTC (2019-01-01 03:00)

    for hour in range(180):
        target_time = start_time + datetime.timedelta(hours=hour)
        target_time_str = target_time.strftime("%Y%m%d%H") # e.g., '2019010103'
        target_year = str(target_time.year)
        target_month = f"{target_time.month:02d}"
        target_day = f"{target_time.day:02d}"
        target_time_obs_str = 'obs_' + target_time_str + '.csv'
        obs_file_path = os.path.join(obs_csv_root, target_year, target_month, target_day, target_time_obs_str)

        if not os.path.exists(obs_file_path):
            continue

        obs_df = pd.read_csv(obs_file_path)
        obs_cache[target_time_str] = build_station_dict(obs_df, station_row_col_dict)

    # current_start_time = start_time



    # for year in tqdm(range(2021, 2024), desc="Year"):
    #     # print(f"\n--- Processing Year: {year} ---")
    #     for month in tqdm(range(1, 13), desc=f"Year {year}", leave=False):
    #         # print(f"  Processing Month: {year}-{month:02d}")
    #         days_in_month = calendar.monthrange(year, month)[1]
    #         for day in range(1, days_in_month + 1):
    #             # print(f"  Processing Day: {year}-{month:02d}-{day:02d}")
    for date in tqdm(date_range, desc="Processing daily data"):
        year = date.year
        month = date.month
        day = date.day
        date_str = f"{year}{month:02d}{day:02d}"
        tensor_path = os.path.join(input_path_forecast_CMAQ, str(year), f"{month:02d}", f"{day:02d}", date_str + '.pt')
        save_path = os.path.join(output_path_forecast_CMAQ, str(year), f"{month:02d}", f"{day:02d}", date_str + '.pt')

        start_time = datetime.datetime(year, month, day, 3) # Adjust to 3AM UTC

        if os.path.exists(save_path):
            continue

        patch_cmaq_tensor_with_obs(date_str, tensor_path, obs_cache, save_path)

        end_time = start_time + datetime.timedelta(hours=180)
        if end_time.year > 2023:
            continue

        old_times = [
            (start_time + datetime.timedelta(hours=h)).strftime("%Y%m%d%H")
            for h in range(24)
        ]

        new_times = [
            (end_time + datetime.timedelta(hours=h)).strftime("%Y%m%d%H")
            for h in range(24)
        ]
        # print('keys:', obs_cache.keys())
        obs_cache = update_obs_cache(obs_cache, obs_csv_root, old_times, new_times, station_row_col_dict)
        start_time += datetime.timedelta(days=1)


def update_obs_cache(obs_cache, obs_path, old_time_list, new_time_list, station_row_col_dict):
    # Remove old obs data from cache
    for t_str in old_time_list:
        if t_str in obs_cache:
            del obs_cache[t_str]
    # Add new obs data to cache
    for t_str in new_time_list:
        if t_str not in obs_cache:
            y, m, d = t_str[:4], t_str[4:6], t_str[6:8]
            fname = f"obs_{t_str}.csv"
            fpath = os.path.join(obs_path, y, m, d, fname)
            if os.path.exists(fpath):
                obs_cache[t_str] = build_station_dict(pd.read_csv(fpath), station_row_col_dict)
            else:
                obs_cache[t_str] = None
    return obs_cache

def build_station_dict(df, station_row_col_dict):
    """Maps (row, col) to lists of data from an OBS DataFrame."""
    point_dict = defaultdict(list)
    for _, row in df.iterrows():
        key = station_row_col_dict.get(row["STNID"])
        if key is None: 
            continue
        values = row[["PM25", "PM10", "SO2", "NO2", "O3", "CO"]].to_numpy(dtype=float)
        point_dict[key].append(values)
    return point_dict

def patch_cmaq_tensor_with_obs(date_str, tensor_path, obs_cache, save_path):
    """
    Replace CMAQ grid values with observed data (PM2P5, PM10, SO2, NO2, O3, CO) where available.

    Parameters:
        date_str (str): Date in yyyymmdd format (e.g., '20160101').
        tensor_path (str): Path to full .pt file (CMAQ forecast data).
        obs_cache (dict): obs cache.
        save_path (str): Where to save the patched tensor.
    """
    tensor = torch.load(tensor_path)  # shape: [COL, ROW, TSTEP, VAR], forecast data
    pm2p5_var_index = 6
    pm10_var_index = 19
    so2_var_index = 7
    no2_var_index = 9
    o3_var_index = 10
    co_var_index = 8
    index_list = [pm2p5_var_index, pm10_var_index, so2_var_index, no2_var_index, o3_var_index, co_var_index]
    date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").replace(hour=3) # Adjust to 3AM UTC
    total_nan_skipped_count = 0
    total_attempted_patches = 0
    for hour in range(180):
        target_time = date_obj + datetime.timedelta(hours=hour)
        target_time_str = target_time.strftime("%Y%m%d%H")
        # target_year = str(target_time.year)
        # target_month = f"{target_time.month:02d}"
        # target_day = f"{target_time.day:02d}"
        # target_time_obs_str = 'obs_' + target_time_str + '.csv'
        # obs_file_path = os.path.join(obs_path, target_year, target_month, target_day, target_time_obs_str)

        # if not os.path.exists(obs_file_path):
        #     continue

        # obs_df = pd.read_csv(obs_file_path)
        station_dict = obs_cache.get(target_time_str) # { (row, col): [[PM25, PM10, SO2, NO2, O3, CO], ...] }
        if station_dict is None:
            continue

        # station_dict = build_station_dict(obs_df, station_row_col_dict)
        mean_dict = {} # { (row, col): mean_values of [PM25, PM10, SO2, NO2, O3, CO] }
        for key, data_list in station_dict.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data_array = np.array(data_list)
                mean_dict[key] = np.nanmean(data_list, axis=0)
    
        frame = tensor[:, :, hour, :]

        for (row, col), mean_values in mean_dict.items():
            for idx, value in zip(index_list, mean_values):
                total_attempted_patches += 1
                if np.isfinite(value):
                    frame[col, row, idx] = value
                else:
                    total_nan_skipped_count += 1

    # Patching Summary
    # if total_attempted_patches > 0:
    #     print(f"\n --- Patching Summary for {date_str} ---")
    #     print(f"Total attempted patches: {total_attempted_patches}")
    #     print(f"Total NaN (missing) values skipped: {total_nan_skipped_count}")
    #     print(f"Percentage of skipped NaN values: {total_nan_skipped_count / total_attempted_patches * 100:.2f}%")

    # Ensure save path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor, save_path)


if __name__ == "__main__":
    Path = sys.argv[1]
    main(Path) # ~/NIER_AI_v8/Data/PreparationCode$ python Forecast_Obs_patch.py ../../