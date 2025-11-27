import os
import torch
from datetime import timedelta

class LabelPreprocessorDifference:
    def __init__(self, root_dir=None, start_hour_kst=13, tz_diff_kst_utc=9):
        """
        root_dir: root directory where patched observation tensors are stored.
                  Expected structure: root_dir/year/month/day/date.pt
        start_hour_kst: starting hour in KST for label extraction (default 13 = 1 PM)
        tz_diff_kst_utc: KST-UTC difference in hours (default 9)
        """
        self.default_data_root = "/storage/dataset/ozone/NIER_AI_v8/Data/InputData"
        self.data_root = root_dir if root_dir is not None else self.default_data_root
        self.patched_obs_dir = os.path.join(self.data_root, "Fusion_Obs_Patched_ext")
        self.forecast_cmaq_dir = os.path.join(self.data_root, "Forecast_CMAQ_ext")

        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root path not found: {self.data_root}")

        self.start_hour_kst = start_hour_kst
        self.tz_diff_kst_utc = tz_diff_kst_utc
        self.start_hour_utc = (self.start_hour_kst - self.tz_diff_kst_utc) % 24

    def load_tensor_obs_patched(self, year, month, day):
        """
        Load O3-only patched tensor for a specific date.

        Returns:
            torch.Tensor: shape (COL, ROW, 24)
        """
        dir_path = os.path.join(self.patched_obs_dir, f"{year}", f"{month:02d}", f"{day:02d}")
        file_path = os.path.join(dir_path, f"{year}{month:02d}{day:02d}.pt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Patched tensor not found: {file_path}")

        return torch.load(file_path)

    def load_tensor_cmaq_o3(self, year, month, day):
        """
        Load O3-only patched tensor for a specific date.

        Returns:
            torch.Tensor: shape (COL, ROW, 24)
        """
        dir_path = os.path.join(self.forecast_cmaq_dir, f"{year}", f"{month:02d}", f"{day:02d}")
        file_path = os.path.join(dir_path, f"{year}{month:02d}{day:02d}.pt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Patched tensor not found: {file_path}")

        return torch.load(file_path)

    def build_patched_obs_tensor(self, target_date):
        """
        Build label tensor consisting of 72 hours starting from start_hour_kst
        on target_date.

        Args:
            target_date (datetime.date)

        Returns:
            torch.Tensor: shape (COL, ROW, 72)
        """
        sh = self.start_hour_utc

        # Days needed: D0, D1, D2, D3
        d0 = target_date
        d1 = target_date + timedelta(days=1)
        d2 = target_date + timedelta(days=2)
        d3 = target_date + timedelta(days=3)

        T0 = self.load_tensor_obs_patched(d0.year, d0.month, d0.day)  # (C, R, 24)
        T1 = self.load_tensor_obs_patched(d1.year, d1.month, d1.day)  # (C, R, 24)
        T2 = self.load_tensor_obs_patched(d2.year, d2.month, d2.day)  # (C, R, 24)
        T3 = self.load_tensor_obs_patched(d3.year, d3.month, d3.day)  # (C, R, 24)

        # Slicing to get exactly 72 hours from sh UTC
        part0 = T0[:, :, sh:]    # from start hour to end of day0
        part1 = T1[:, :, :24]    # full day1
        part2 = T2[:, :, :24]    # full day2
        part3 = T3[:, :, :72 - (part0.shape[2] + 24 + 24)]  # remaining hours from day3
        label_tensor = torch.cat([part0, part1, part2, part3], dim=2)  # (C, R, 72)

        return label_tensor

    def build_cmaq_o3_tensor(self, target_date):
        """
        Concatenate tensors from 2 days ago (24h), 1 day ago (24h), and today (72h)
        along the time axis.

        Args:
            subdir (str): Subdirectory name (e.g., "Forecast_CMAQ_Obs_ext")
            target_date (datetime.date): The target date

        Returns:
            torch.Tensor: shape (col, row, T=120, var)
        """
        sh = self.start_hour_utc
        d0 = target_date
        d1 = target_date + timedelta(days=1)
        d2 = target_date + timedelta(days=2)
        d3 = target_date + timedelta(days=3)

        o3_index = 10
        T0 = self.load_tensor_cmaq_o3(d0.year, d0.month, d0.day)[:, :, :24, o3_index]
        T1 = self.load_tensor_cmaq_o3(d1.year, d1.month, d1.day)[:, :, :24, o3_index]
        T2 = self.load_tensor_cmaq_o3(d2.year, d2.month, d2.day)[:, :, :24, o3_index]
        cmaq_o3_tensor = torch.cat([T0, T1, T2], dim=2)  # (C, R, 72)
        return cmaq_o3_tensor
    
    def build_label_tensor(self, target_date):
        """
        Build label tensor consisting of 72 hours starting from start_hour_kst
        on target_date.

        Args:
            target_date (datetime.date)

        Returns:
            torch.Tensor: shape (COL, ROW, 72)
        """
        patched_obs_tensor = self.build_patched_obs_tensor(target_date)  # (C, R, 72)
        cmaq_o3_tensor = self.build_cmaq_o3_tensor(target_date)  # (C, R, 72)
        diff_label = patched_obs_tensor - cmaq_o3_tensor
        return diff_label, cmaq_o3_tensor, patched_obs_tensor