# -*- coding: utf-8 -*-
"""
Created at KAIST OptiML
"""
import numpy as np
import pandas as pd
import sys
import os
import xarray as xr
import calendar
import torch
from tqdm import tqdm

# Only extract CMAQ as we only need O3 ranalysis data
def main(Path):
    global input_path
    storage_path = "/storage/dataset/ozone"
    reanalysis_path = os.path.join(storage_path, "ReanalysisData")
    reanalysis_CMAQ_path = os.path.join(reanalysis_path, "CMAQ")
    input_path = os.path.join(Path , "Data/InputData/Reanalysis_CMAQ_ACONC_ext/")

    example_file = os.path.join(reanalysis_CMAQ_path, "2016/KNU_09_01/ACONC.PM_RQ40i8a.KNU_09_01.20160720.nc")
    # preview_nc_file(example_file)

    extract_reanalysis_CMAQ(reanalysis_CMAQ_path)
    return

def extract_reanalysis_CMAQ(CMAQ_root):
    for year in range(2016, 2024):
        print(f"Extracting Year {year}")
        year_dir = os.path.join(input_path, str(year))
        os.makedirs(year_dir, exist_ok=True)
        extract_reanalysis_CMAQ_year(year, CMAQ_root)
    return

# Process year
def extract_reanalysis_CMAQ_year(year, CMAQ_root):
    for month in tqdm(range(1, 13)):
        month_dir = os.path.join(input_path, str(year), f"{month:02d}")
        os.makedirs(month_dir, exist_ok=True)
        extract_reanalysis_CMAQ_year_month(year, month, CMAQ_root)
    return

# Process month
def extract_reanalysis_CMAQ_year_month(year, month, CMAQ_root):
    days_in_month = calendar.monthrange(year, month)[1]
    for day in range(1, days_in_month + 1):
        day_dir = os.path.join(input_path, str(year), f"{month:02d}", f"{day:02d}") # Directory to save tensor
        os.makedirs(day_dir, exist_ok=True)
        extract_reanalysis_CMAQ_year_month_day(year, month, day, CMAQ_root, day_dir)
    return

# Process day
def extract_reanalysis_CMAQ_year_month_day(year, month, day, CMAQ_root, day_dir):
    month_str = f"{month:02d}"
    date_str = f"{year}{month_str}{day:02d}"

    knu_folder = f"KNU_09_01"
    file_name = f"ACONC.PM_RQ40i8a.{knu_folder}.{date_str}.nc"
    nc_file_path = os.path.join(CMAQ_root, str(year), knu_folder, file_name)

    save_nc_as_tensor(nc_file_path, day_dir, f"{date_str}.pt")
    return

# Process day
def save_nc_as_tensor(nc_file_path, dir_save, filename):
    """
    Load a NetCDF file, extract selected variables, convert to tensor, and save to disk.

    Parameters:
        nc_file_path (str): Path to the input NetCDF file.
        dir_save (str): Directory where the .pt file should be saved.
        filename (str): Output filename.
    """
    try:
        ds = xr.open_dataset(nc_file_path)

        # Exclude non-feature variables if needed
        exclude_vars = ["TFLAG"]
        vars_to_use = [var for var in ds.data_vars if var not in exclude_vars]

        tensors = []

        for var in vars_to_use:
            arr = ds[var].values  # shape: (TSTEP, LAY, ROW, COL)
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0]  # remove LAY dimension if it's singleton
            tensors.append(torch.tensor(arr, dtype=torch.float32))  # shape: (TSTEP, ROW, COL)

        # Stack to shape (TSTEP, VAR, ROW, COL)
        data_tensor = torch.stack(tensors, dim=1)  # [TSTEP, VAR, ROW, COL]
        data_tensor_reshaped = data_tensor.permute(3, 2, 0, 1) # [COL, ROW, TSTEP, VAR]

        # Ensure save directory exists
        os.makedirs(dir_save, exist_ok=True)
        save_path = os.path.join(dir_save, filename)
        torch.save(data_tensor_reshaped, save_path)
    except Exception as e:
        print(f"Failed to convert {nc_file_path} to tensor: {e}")

# For checking
def preview_nc_file(nc_file_path):
    try:
        ds = xr.open_dataset(nc_file_path)
        print(f"\nSuccessfully opened: {nc_file_path}\n")

        # List variables (columns)
        print("Variables (columns) in file:")
        print(list(ds.data_vars))  # or ds.variables.keys() for all, including coordinates

        # Show dimensions
        print("\nDimensions:")
        print(ds.dims)
        print(ds['TFLAG'].isel(VAR=0).values[:5])
        # # Show first few rows of each variable
        # print("\nSample data (first few entries):")
        # for var in list(ds.data_vars)[:3]:  # limit to first 3 vars for brevity
        #     print(f"\n{var}:")
        #     print(ds[var].isel(time=0).values if 'time' in ds[var].dims else ds[var].values[:5])

    except FileNotFoundError:
        print(f"File not found: {nc_file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")


def check_folder_exists(path):
    if not os.path.exists(path):
        print(f"{path} does not exists")
    else:
        print(f"{path} exists")

if __name__ == "__main__":
    Path = sys.argv[1]
    main(Path)