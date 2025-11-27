#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created by OptiML Lab, KAIST AI

Generates Label Dataset
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from LabelPreprocessor import LabelPreprocessor  # Assuming you have this class

STORAGE_ROOT = "/storage/dataset/ozone"
STORAGE_PROJECT_ROOT = os.path.join(STORAGE_ROOT, "NIER_AI_v8")
INPUT_DIR = os.path.join(STORAGE_PROJECT_ROOT, "Data", "InputData")
PATCHED_SUBDIR = "Patched_Obs_ext_parallel"
OUTPUT_ROOT = os.path.join(STORAGE_PROJECT_ROOT, "Dataset/Output")

label_preprocessor = LabelPreprocessor(root_dir=os.path.join(INPUT_DIR, PATCHED_SUBDIR))

def str_to_date(s):
    return datetime.strptime(s, "%Y%m%d").date()

def main():
    start_date = "20160101"
    end_date = "20231231"

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for target_date_str in tqdm(date_range, desc="Processing daily labels"):
        target_date = target_date_str.date()
        y, m, d = target_date.year, target_date.month, target_date.day
        out_dir = os.path.join(OUTPUT_ROOT, f"{y}", f"{m:02d}", f"{d:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"y_{y}{m:02d}{d:02d}.pt")
        if os.path.exists(out_file):
            continue

        try:
            # This assumes LabelPreprocessor has build_label_tensor() method
            tensor = label_preprocessor.build_label_tensor(
                target_date
            )
            torch.save(tensor, out_file)
        except FileNotFoundError:
            continue

if __name__ == "__main__":
    main()
