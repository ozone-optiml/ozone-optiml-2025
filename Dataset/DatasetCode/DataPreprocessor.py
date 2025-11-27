# Second Step Preprocessing Code
"""
Created by OptiML Lab, KAIST AI
"""
import os, sys
data_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.dirname(data_path)
sys.path.append(code_path)

import numpy as np
import torch
from numpy import genfromtxt
import pandas as pd
from fancyimpute import SoftImpute
from tqdm import tqdm
from datetime import date, timedelta

class DataPreprocessor:
    def __init__(self, root_dir=None):
        default_data_root = os.path.join("Data", "InputData")
        self.data_root = root_dir if root_dir is not None else default_data_root

        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root path not found: {self.data_root}")
        
        self.forecast_cmaq_dir = os.path.join(self.data_root, "Forecast_CMAQ_ext")
        self.forecast_mcip_dir = os.path.join(self.data_root, "Forecast_MCIP_ext")


    def load_tensor(self, subdir, year, month, day):
        """
        Load a tensor (.pt file) from the specified subdirectory and date.

        Args:
            subdir (str): Subdirectory under self.data_root (e.g., "Forecast_CMAQ_Obs_ext")
            year (int): 4-digit year
            month (int): Month (1-12)
            day (int): Day (1-31)

        Returns:
            torch.Tensor: shape (col, row, time, var)
        """
        dir_path = os.path.join(
            self.data_root,
            subdir,
            f"{year}",
            f"{month:02d}",
            f"{day:02d}"
        )
        file_path = os.path.join(dir_path, f'{year}{month:02d}{day:02d}.pt')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tensor file not found: {file_path}")

        return torch.load(file_path)


    def build_input_tensor(self, subdir, target_date):
        """
        Concatenate tensors from 2 days ago (24h), 1 day ago (24h), and today (72h)
        along the time axis.

        Args:
            subdir (str): Subdirectory name (e.g., "Forecast_CMAQ_Obs_ext")
            target_date (datetime.date): The target date

        Returns:
            torch.Tensor: shape (col, row, T=120, var)
        """
        d2 = target_date - timedelta(days=2) # D-2
        d1 = target_date - timedelta(days=1) # D-1
        d0 = target_date # D-day

        T2 = self.load_tensor(subdir, d2.year, d2.month, d2.day)[:, :, :24, :]
        T1 = self.load_tensor(subdir, d1.year, d1.month, d1.day)[:, :, :24, :]
        T0 = self.load_tensor(subdir, d0.year, d0.month, d0.day)[:, :, :72, :]

        return torch.cat([T2, T1, T0], dim=2)