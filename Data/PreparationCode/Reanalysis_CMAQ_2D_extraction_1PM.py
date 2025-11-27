import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import torch
from datetime import timedelta
import calendar

def parse_tflag_to_datetimes(tflag_raw):
    a = np.array(tflag_raw)
    if a.ndim == 3:
        a = a[:, 0, :]
    if a.ndim == 1 and a.dtype.kind in 'U':
        a = a.reshape(-1, 2)
    a = a.astype(int)
    days = a[:, 0]
    times = a[:, 1]
    datetimes = []
    for d, t in zip(days, times):
        yr = d // 1000
        doy = d % 1000
        base = pd.Timestamp(year=yr, month=1, day=1, tz='UTC') + pd.Timedelta(days=doy - 1)
        hh = t // 10000
        mm = (t % 10000) // 100
        ss = t % 100
        ts = pd.Timestamp(base.year, base.month, base.day, hh, mm, ss, tz='UTC')
        datetimes.append(ts)
    return pd.DatetimeIndex(datetimes)

def file_name_format(year, month):
    # return f"ACONC.1hr.day1.CRI.ME_KQ01_Krig2.KNU_09_01.{year}.{month:02d}.ncf"
    return f"ACONC.1hr.day1.CRI.PM_RQ40i8a_krig3.KNU_09_01.{year}.{month:02d}.ncf"


def load_month_concat(src_root, year, month):
    fn_cur = file_name_format(year, month)
    path_cur = os.path.join(src_root, fn_cur)
    if not os.path.exists(path_cur):
        return None, None, None
    ds_cur = xr.open_dataset(path_cur)
    vars_use = [v for v in ds_cur.data_vars if v != "TFLAG"]
    tflag_cur = ds_cur["TFLAG"].isel(VAR=0).values
    times_cur = parse_tflag_to_datetimes(tflag_cur)
    arrs_cur = []
    for v in vars_use:
        arr = ds_cur[v].values
        if arr.ndim == 4:
            arr = arr[:, 0]
        arrs_cur.append(arr)
    data_cur = np.stack(arrs_cur, axis=1)
    ds_cur.close()
    data_all = data_cur
    times_all = times_cur

    next_year = year + (1 if month == 12 else 0)
    next_month = 1 if month == 12 else month + 1
    fn_next = file_name_format(next_year, next_month)
    path_next = os.path.join(src_root, fn_next)
    if os.path.exists(path_next):
        ds_next = xr.open_dataset(path_next)
        tflag_next = ds_next["TFLAG"].isel(VAR=0).values
        times_next = parse_tflag_to_datetimes(tflag_next)
        start_mask = (times_next >= pd.Timestamp(f"{next_year}-{next_month:02d}-01 00:00:00", tz="UTC")) & \
                     (times_next <= pd.Timestamp(f"{next_year}-{next_month:02d}-01 02:00:00", tz="UTC"))
        if start_mask.any():
            arrs_next = []
            for v in vars_use:
                arr = ds_next[v].values
                if arr.ndim == 4:
                    arr = arr[:, 0]
                arrs_next.append(arr[start_mask])
            data_next = np.stack(arrs_next, axis=1)
            data_all = np.concatenate([data_all, data_next], axis=0)
            times_all = pd.DatetimeIndex(np.concatenate([times_all.values, times_next[start_mask].values]))
        ds_next.close()

    if times_all.tz is None:
        times_all = times_all.tz_localize("UTC")
    else:
        times_all = times_all.tz_convert("UTC")
    return data_all, times_all, vars_use


def extract_month_days(data_all, times_all, vars_use, year, month, dst_root):
    days_in_month = calendar.monthrange(year, month)[1]
    month_start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
    month_end = pd.Timestamp(year=year, month=month, day=days_in_month, tz='UTC')
    date_cursor = month_start
    while date_cursor <= month_end:
        start = date_cursor + pd.Timedelta(hours=3)
        end = start + pd.Timedelta(hours=23)
        idx = np.where((times_all >= start) & (times_all <= end))[0]
        if len(idx) == 24:
            slice_data = data_all[idx]
            tensor = torch.tensor(slice_data, dtype=torch.float32).permute(3, 2, 0, 1)
            ymd = start.strftime("%Y%m%d")
            dir_save = os.path.join(dst_root, f"{year}", f"{month:02d}", f"{start.day:02d}")
            os.makedirs(dir_save, exist_ok=True)
            torch.save(tensor, os.path.join(dir_save, f"{ymd}.pt"))
            print(f"Saved {ymd} ({tensor.shape})")
        else:
            print(f"Skipped {date_cursor.strftime('%Y-%m-%d')} (incomplete)")
        date_cursor += pd.Timedelta(days=1)


def main():
    src_root = "/storage/dataset/ozone/Fusion_251116"
    dst_root = "/storage/dataset/ozone/NIER_AI_v8/Data/InputData/Fusion_ext"
    os.makedirs(dst_root, exist_ok=True)
    for year in range(2019, 2021):
        for month in range(1, 13):
            print(f"Processing {year}-{month:02d}")
            data_all, times_all, vars_use = load_month_concat(src_root, year, month)
            if data_all is None:
                print(f"No file for {year}-{month:02d}")
                continue
            extract_month_days(data_all, times_all, vars_use, year, month, dst_root)

if __name__ == "__main__":
    main()