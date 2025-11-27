#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created by OptiML Lab, KAIST AI

Generates Input/Output Dataset
"""

import os, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from datetime import datetime, timedelta
from DataPreprocessor import DataPreprocessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_OZONE_ROOT = '/storage/dataset/ozone/NIER_AI_v8'
DATA_ROOT = os.path.join(STORAGE_OZONE_ROOT, "Data")
INPUT_DIR = os.path.join(DATA_ROOT, "InputData")
DATASET_ROOT = os.path.join(STORAGE_OZONE_ROOT, "Dataset")

INPUT_DIR_CMAQ = os.path.join(DATA_ROOT, "InputData/Forecast_CMAQ_ext")
OUTPUT_DIR_CMAQ = os.path.join(DATASET_ROOT, "EncoderInput/CMAQ")
INPUT_DIR_MCIP = os.path.join(DATA_ROOT, "InputData/Forecast_MCIP_ext")
OUTPUT_DIR_MCIP = os.path.join(DATASET_ROOT, "EncoderInput/MCIP")

data_preprocessor = DataPreprocessor(root_dir=INPUT_DIR)

def str_to_date(s):
    return datetime.strptime(s, "%Y%m%d").date()

def main():
    start_date = "20190103"
    end_date = "20231231"

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for target_date_str in tqdm(date_range, desc="Processing CMAQ daily data"):
        target_date = target_date_str.date()
        y, m, d = target_date.year, target_date.month, target_date.day
        out_dir = os.path.join(OUTPUT_DIR_CMAQ, f"{y}", f"{m:02d}", f"{d:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{y}{m:02d}{d:02d}.pt")

        if os.path.exists(out_file):
            continue

        try:
            tensor = data_preprocessor.build_input_tensor(INPUT_DIR_CMAQ, target_date)
            torch.save(tensor, out_file)
        except FileNotFoundError:
            continue

    print("\nCMAQ preprocessing completed.")

    for target_date_str in tqdm(date_range, desc="Processing MCIP daily data"):
        target_date = target_date_str.date()
        y, m, d = target_date.year, target_date.month, target_date.day
        out_dir = os.path.join(OUTPUT_DIR_MCIP, f"{y}", f"{m:02d}", f"{d:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{y}{m:02d}{d:02d}.pt")

        if os.path.exists(out_file):
            continue

        try:
            tensor = data_preprocessor.build_input_tensor(INPUT_DIR_MCIP, target_date)
            torch.save(tensor, out_file)
        except FileNotFoundError:
            continue

    print("\nMCIP preprocessing completed.")

if __name__ == "__main__":
    main()