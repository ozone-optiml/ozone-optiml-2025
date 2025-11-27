# -*- coding: utf-8 -*-
"""
Created at KAIST OptiML (Donghwa Kim)
"""

import xarray as xr
import calendar
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from netCDF4 import Dataset
import os, sys
import glob

np.set_printoptions(precision=6, suppress=True)

def show_ds(ds, num=5):
    for i, var in enumerate(ds.data_vars):
        print(f"{var}: shape={ds[var].shape}, dims={ds[var].dims}, units={ds[var].attrs.get('units', 'N/A')}")
        if i > num:
            break


def save_nc_as_tensor(nc_file_path, dir_save, filename):
    """
    Load a METCRO NetCDF file, extract selected variables, convert to tensor, and save to disk.
    Output tensor has shape [COL, ROW, TSTEP, VAR].

    Parameters:
        nc_file_path (str): Path to the input NetCDF file.
        dir_save (str): Directory where the .pt file should be saved.
        filename (str): Output filename.
    """
    try:
        ds = xr.open_dataset(nc_file_path)

        # Exclude non-feature variables
        exclude_vars = ["TFLAG"]
        vars_to_use = [var for var in ds.data_vars if var not in exclude_vars]

        tensors = []

        for var in vars_to_use:
            arr = ds[var].values  # shape: (TSTEP, LAY, ROW, COL)
            if arr.ndim != 4:
                raise ValueError(f"Unexpected shape for variable {var}: {arr.shape}")

            # Use the first layer (LAY)
            arr = arr[:, 0, :, :]

            tensors.append(torch.tensor(arr, dtype=torch.float32))  # [TSTEP, ROW, COL]

        # Stack into (TSTEP, VAR, ROW, COL)
        data_tensor = torch.stack(tensors, dim=1)  # [TSTEP, VAR, ROW, COL]

        # Permute to (COL, ROW, TSTEP, VAR)
        data_tensor_reshaped = data_tensor.permute(3, 2, 0, 1)  # [COL, ROW, TSTEP, VAR]

        # Save
        os.makedirs(dir_save, exist_ok=True)
        save_path = os.path.join(dir_save, filename)
        torch.save(data_tensor_reshaped, save_path)

    except Exception as e:
        print(f"Failed to convert {nc_file_path} to tensor: {e}")



def main(Path):
    storage_path = "/storage/dataset/ozone"
    forecast_path = os.path.join(storage_path, "ForecastData")
    input_path = Path
    
    for year in tqdm(range(2019, 2024), desc="Year", position=0):
        year_str = str(year)
        forecast_year_path = os.path.join(forecast_path, year_str)
        input_year_path = os.path.join(input_path, year_str)
        for month in tqdm(range(1, 13), desc="Month", position=1, leave=False):
            month_str = f"{month:02d}"
            month_dir = os.path.join(forecast_year_path, month_str)
            input_month_path = os.path.join(input_year_path, month_str)
            for date in tqdm(range(1, calendar.monthrange(year, month)[1]+1), desc="Date", position=2, leave=False):
                date_str = f"{date:02d}"
                date_dir = os.path.join(month_dir, date_str)
                input_date_path = os.path.join(input_month_path, date_str)

                netcdf_dir = os.path.join(date_dir, "NIER_09_01")
                files = [f for f in os.listdir(netcdf_dir) if f.startswith("METCRO2D")]

                metcro2d_file_path = os.path.join(netcdf_dir, files[0])
                full_date = year_str + month_str + date_str

                save_nc_as_tensor(metcro2d_file_path, input_date_path, f"{full_date}.pt")


if __name__ == "__main__":
    default_path = "~/NIER_AI_v8/Data/InputData/Forecast_MCIP_ext"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    path = os.path.expanduser(path)
    main(path)