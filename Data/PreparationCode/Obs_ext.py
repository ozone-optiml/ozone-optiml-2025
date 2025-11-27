import numpy as np
import pandas as pd
import sys
import os

def main(Path):
    obs_txt_root = os.path.join("/storage/dataset/ozone/ObsData/korea/air")
    input_path_obs = os.path.join(Path, "Data/InputData/Obs_ext")
    convert_obs_txt_to_csv(obs_txt_root, input_path_obs)
    return

def convert_obs_txt_to_csv(obs_txt_root, input_path_obs):
    years = sorted(os.listdir(obs_txt_root))
    for year in years:
        year_dir = os.path.join(obs_txt_root, year)
        if not os.path.isdir(year_dir):
            continue
        print(f"Processing year: {year}")
        months = sorted(os.listdir(year_dir))
        for month in months:
            month_dir = os.path.join(year_dir, month)
            if not os.path.isdir(month_dir):
                continue

            days = sorted(os.listdir(month_dir))
            for day in days:
                day_dir = os.path.join(month_dir, day)
                if not os.path.isdir(day_dir):
                    continue

                txt_files = sorted(f for f in os.listdir(day_dir) if f.endswith(".txt"))
                for filename in txt_files:
                    txt_file_path = os.path.join(day_dir, filename)

                    with open(txt_file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    data_lines = [line.strip() for line in lines[6:] if line.strip()]

                    if not data_lines:
                        print(f"⚠️ No data in file: {txt_file_path}")
                        continue

                    columns = [
                        "TIME(UTC)", "STNID", "LAT", "LON",
                        "PM25", "PM10", "SO2", "NO2", "O3", "CO", "AREA_NAME"
                    ]

                    df = pd.DataFrame(
                        [line.split(",") for line in data_lines],
                        columns=columns
                    )

                    for col in ["PM25", "PM10", "SO2", "NO2", "O3", "CO"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].replace(-999, pd.NA)

                    datetime_str = filename[9:19]   # e.g., '2016010100'
                    date = datetime_str[:8]         # '20160101'
                    hour = datetime_str[8:]         # '00'
                    yyyy, mm, dd = date[:4], date[4:6], date[6:8]

                    nested_output_dir = os.path.join(input_path_obs, yyyy, mm, dd)
                    os.makedirs(nested_output_dir, exist_ok=True)

                    output_filename = f"obs_{date}{hour}.csv"
                    output_path = os.path.join(nested_output_dir, output_filename)
                    df.to_csv(output_path, index=False, na_rep="NaN")


if __name__ == "__main__":
    Path = sys.argv[1]
    main(Path)
