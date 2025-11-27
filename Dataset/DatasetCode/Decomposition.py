"""
data_utils, used for data preprocessing
"""
import os, sys
import numpy as np
import pandas as pd
import scipy.signal as signal
from tqdm import tqdm
import torch
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))
OZONE_ROOT = os.path.join(PROJECT_ROOT, "storage", "dataset", "ozone", "NIER_AI_v8")
DATASET_ROOT = os.path.join(OZONE_ROOT, "Dataset")
INPUT_DIR_CMAQ = os.path.join(DATASET_ROOT, "Input/CMAQ")
OUTPUT_ROOT_CMAQ = os.path.join(DATASET_ROOT, "Input_Decomposed/CMAQ")
INPUT_DIR_MCIP = os.path.join(DATASET_ROOT, "Input/MCIP")
OUTPUT_ROOT_MCIP = os.path.join(DATASET_ROOT, "Input_Decomposed/MCIP")


print(f"Input Directory CMAQ: {INPUT_DIR_CMAQ}")
print(f"Output Root CMAQ: {OUTPUT_ROOT_CMAQ}")

def str_to_date(s):
    return datetime.strptime(s, "%Y%m%d").date()

def fir_kaiser_filter_tensor(tensor, beta=5, cutoff_period=21, order=42):
    if tensor.shape[2] == 0:
        return tensor.clone(), torch.zeros_like(tensor)
        
    input_dtype = tensor.dtype
    device = tensor.device
    np_input = tensor.cpu().numpy()
    
    cutoff_frequency = 2 / cutoff_period
    fir_coefficients = signal.firwin(order + 1, cutoff=cutoff_frequency, window=('kaiser', beta))

    x_t0 = np_input[:, :, 0:1, :] 

    tensor_subtracted = np_input - x_t0
    
    lt_subtracted = signal.lfilter(fir_coefficients, [1.0], tensor_subtracted, axis=2)
    
    lt_component = lt_subtracted + x_t0
    
    st_component = np_input - lt_component

    lt_tensor = torch.from_numpy(lt_component).to(device=device, dtype=input_dtype)
    st_tensor = torch.from_numpy(st_component).to(device=device, dtype=input_dtype)

    return lt_tensor, st_tensor


def main():
    start_date = "20190103"
    end_date = "20231231"

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for target_date_str in tqdm(date_range, desc="Processing CMAQ daily data"):
        target_date = target_date_str.date()
        y, m, d = target_date.year, target_date.month, target_date.day
        out_dir = os.path.join(OUTPUT_ROOT_CMAQ, f"{y}", f"{m:02d}", f"{d:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_file_lt = os.path.join(out_dir, f"{y}{m:02d}{d:02d}_LT.pt")
        out_file_st = os.path.join(out_dir, f"{y}{m:02d}{d:02d}_ST.pt")

        if os.path.exists(out_file_lt) and os.path.exists(out_file_st):
            continue

        try:
            tensor = torch.load(os.path.join(INPUT_DIR_CMAQ, f"{y}", f"{m:02d}", f"{d:02d}", f"{y}{m:02d}{d:02d}.pt"))
            lt_tensor, st_tensor = fir_kaiser_filter_tensor(tensor)
            torch.save(lt_tensor, out_file_lt)
            torch.save(st_tensor, out_file_st)
        except FileNotFoundError:
            continue

    print("\nCMAQ decomposition completed.")

    for target_date_str in tqdm(date_range, desc="Processing MCIP daily data"):
        target_date = target_date_str.date()
        y, m, d = target_date.year, target_date.month, target_date.day
        out_dir = os.path.join(OUTPUT_ROOT_MCIP, f"{y}", f"{m:02d}", f"{d:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_file_lt = os.path.join(out_dir, f"{y}{m:02d}{d:02d}_LT.pt")
        out_file_st = os.path.join(out_dir, f"{y}{m:02d}{d:02d}_ST.pt")

        if os.path.exists(out_file_lt) and os.path.exists(out_file_st):
            continue

        try:
            tensor = torch.load(os.path.join(INPUT_DIR_MCIP, f"{y}", f"{m:02d}", f"{d:02d}", f"{y}{m:02d}{d:02d}.pt"))
            lt_tensor, st_tensor = fir_kaiser_filter_tensor(tensor)
            torch.save(lt_tensor, out_file_lt)
            torch.save(st_tensor, out_file_st)
        except FileNotFoundError:
            continue

    print("MCIP decomposition completed.")
    
if __name__ == "__main__":
    main()