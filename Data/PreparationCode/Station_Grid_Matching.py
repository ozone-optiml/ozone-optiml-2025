# -*- coding: utf-8 -*-
"""
Created at KAIST OptiML
"""
import numpy as np
import pandas as pd
import sys
import os
import xarray as xr
from math import radians, cos, sin, asin, sqrt

def main(Path,DP):
    storage_path = "/storage/dataset/ozone"
    obs_path = os.path.join(storage_path, "ObsData/kor/air")
    misc_path = os.path.join(Path, "Data/Misc")
    grid_info_file = os.path.join(misc_path, "GridInfo", "GRID_INFO_09km.nc")
    station_info_file = os.path.join(misc_path, "Korea_AQS_site_info_Oct_2021_updated.csv")
    input_path = os.path.join(Path , "Data/Input_data/Obs_ext/")

    # Adds nearest grid info to the station info
    grid_info = netcdf_to_df(grid_info_file)
    station_info = pd.read_csv(station_info_file)
    add_grid_info_to_station(grid_info, station_info)
    station_info.to_csv(os.path.join(misc_path, "station_with_nearest_grid.csv"))

    for year in range(2016, 2024):
        continue
    return

def add_grid_info_to_station(grid_info, station_info):
    """
    Creates map from station ID to (row, col) of 9km by 9km grid
    """
    nearest_rows = []
    nearest_cols = []

    for idx, station in station_info.iterrows():
        lon, lat = station["Lon"], station["Lat"]
        
        # Compute distances to all grid centers
        distances = grid_info.apply(lambda row: haversine(lon, lat, row["LON"], row["LAT"]), axis=1)
        nearest_idx = distances.idxmin()
        
        nearest_row = grid_info.loc[nearest_idx, "ROW"]
        nearest_col = grid_info.loc[nearest_idx, "COL"]

        nearest_rows.append(nearest_row)
        nearest_cols.append(nearest_col)

    # Add to station DataFrame
    station_info["ROW"] = nearest_rows
    station_info["COL"] = nearest_cols
    return

def process_obs(year, month, date):
    """
    Create 82(rows) by 67(rows)
    """
    return

def netcdf_to_df(file):
    ds = xr.open_dataset(file)
    df = ds.to_dataframe().reset_index()
    return df

def haversine(lon1, lat1, lon2, lat2):
    # convert degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    
    return c

def parse_airkorea_file(filepath):
    # Skip header lines starting with '=' or text
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith('=') and not line.startswith('Airkorea') and not line.startswith('FILE_NAME') and not line.startswith('TIME')]

    # Read into DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(lines)), header=None)
    df.columns = ["TIME", "STNID", "LAT", "LON", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AREA_NAME"]

    # Drop metadata columns if not needed
    df = df[["STNID", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]]

    # Filter invalid rows (e.g., PM2.5 == -999)
    df = df[df["PM2.5"] != -999]

    # Sort by station ID for consistent order
    df = df.sort_values(by="STNID").reset_index(drop=True)

    return df

if __name__ == "__main__":
    Path = sys.argv[1]
    DP = sys.argv[2]
    main(Path, DP)