import sys
import os
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# --- Add project root to path to allow importing custom modules ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Dataset.ozone_dataset import OzoneTestDataset
from Models.ozone_informer.models.model import Informer

# --- Helper Functions (Copied from your original script) ---

def get_args():
    """Parses command line arguments for station-based evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the Informer model against ground station observations.")
    
    # --- Paths ---
    parser.add_argument("--root_dir", type=str, default="/storage/dataset/ozone/NIER_AI_v8/Dataset", help="Root directory of the dataset")
    parser.add_argument("--model_name", type=str, default="6. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed)/epoch99_loss0.00000_rmse0.00066.pth", help="Name of the trained model file")
    parser.add_argument("--station_info_path", type=str, default="station_with_nearest_grid.csv", help="Path to the CSV file with station grid info.")
    parser.add_argument("--obs_data_root", type=str, default="/storage/dataset/ozone/NIER_AI_v8/Data/InputData/Obs_ext", help="Root directory for the hourly observation CSV files.")

    # --- Model Hyperparameters ---
    parser.add_argument("--e_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--d_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model Dimension")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward layer dimension")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--reduction", type=int, default=4, help="Spatial reduction factor")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.model_name_short = args.model_name
    args.model_name = os.path.join(script_dir, "../../../Train/checkpoints", args.model_name)
    
    return args


def load_model(args, device):
    """Initializes and loads the pre-trained Informer model."""
    print("Initializing Informer model...")
    model = Informer(
        enc_in=57, dec_in=57, c_out=57,
        seq_len=120, label_len=48, out_len=72,
        factor=5, d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=args.d_layers, d_ff=args.d_ff,
        dropout=args.dropout, attn='flash', embed='fixed', freq='h',
        activation='gelu', output_attention=False, distil=False,
        mix=False, device=device
    )

    print(f"Loading model from: {args.model_name}")
    assert os.path.exists(args.model_name), f"Model file not found: {args.model_name}"

    checkpoint = torch.load(args.model_name, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    return model

def load_station_info(station_info_path, reduction_factor):
    """
    Loads station data and returns a dictionary with details for each station.
    """
    print(f"Loading station info from: {station_info_path}")
    try:
        stations_df = pd.read_csv(station_info_path)
    except FileNotFoundError:
        print(f"Error: Station info file not found at {station_info_path}")
        sys.exit(1)

    # Calculate reduced grid coordinates
    stations_df['R_ROW'] = stations_df['ROW'] // reduction_factor
    stations_df['R_COL'] = stations_df['COL'] // reduction_factor
    
    # Create a dictionary mapping station_id to its details
    stations_df.rename(columns={'Station_ID': 'station_id'}, inplace=True)
    station_details = stations_df[['station_id', 'Province', 'R_ROW', 'R_COL']].set_index('station_id').to_dict('index')
    
    all_station_ids = set(stations_df['station_id'])
    
    print(f"Loaded details for {len(all_station_ids)} stations.")
    return station_details, all_station_ids

def get_daily_station_max_obs(target_date, obs_data_root, all_station_ids):
    """
    Loads 24 hourly observation files for a given date and finds the maximum 
    O3 value for each individual station.
    """
    date_str = target_date.strftime('%Y%m%d')
    year_str = target_date.strftime('%Y')
    month_str = target_date.strftime('%m')
    day_str = target_date.strftime('%d')
    
    hourly_files_dir = os.path.join(obs_data_root, year_str, month_str, day_str)
    
    if not os.path.isdir(hourly_files_dir):
        return None

    all_obs_for_day = []
    for hour in range(24):
        hh = f"{hour:02d}"
        obs_file = os.path.join(hourly_files_dir, f"obs_{date_str}{hh}.csv")
        if os.path.exists(obs_file):
            df = pd.read_csv(obs_file)
            all_obs_for_day.append(df)
            
    if not all_obs_for_day:
        return None
        
    full_day_df = pd.concat(all_obs_for_day, ignore_index=True)
    
    relevant_obs = full_day_df[
        full_day_df['STNID'].isin(all_station_ids) & pd.notna(full_day_df['O3'])
    ].copy()
    
    # Calculate daily maximum per station
    daily_max_o3_per_station = (relevant_obs.groupby('STNID')['O3'].max() * 1000).to_dict()
    
    return daily_max_o3_per_station

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    station_details, all_station_ids = load_station_info(
        args.station_info_path, args.reduction
    )

    model = load_model(args, device)
    test_dataset = OzoneTestDataset(args.root_dir, reduction=args.reduction)
    
    daily_results = []

    normalization_path = os.path.join(os.path.dirname(__file__), f'../../../Dataset/normalization_reduction_{args.reduction}.pt')
    normalization_info = torch.load(normalization_path)
    mean = normalization_info['mean'][10].item()
    std = normalization_info['std'][10].item()

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating Test Dataset vs Stations"):
            encoder_input, decoder_input, true_output, paths_dict = test_dataset[i]
            
            # --- Model Prediction & Data Preparation ---
            encoder_input = encoder_input.unsqueeze(0).to(device)
            decoder_input = decoder_input.unsqueeze(0).to(device)
            
            pred = model(encoder_input, decoder_input)
            pred = pred.permute(0, 2, 1, 3).squeeze(0).squeeze(-1).cpu().numpy()
            pred_ppb = pred * 1000

            true_output = true_output.permute(1, 0, 2).cpu().numpy()
            true_ppb = true_output * 1000
            
            forecast_pred = encoder_input[:, :, :, -72:, 10]
            forecast_pred = (forecast_pred.squeeze(0).squeeze(-1).cpu().numpy() * std) + mean
            forecast_pred_ppb = forecast_pred.transpose(1, 0, 2) * 1000

            start_date_str = os.path.basename(paths_dict['output']).split('_')[1].replace('.pt', '')
            start_date = datetime.strptime(start_date_str, '%Y%m%d')

            day_slices = {1: slice(0, 24), 2: slice(24, 48), 3: slice(48, 72)}
            
            for day_idx, time_slice in day_slices.items():
                target_date = start_date + timedelta(days=day_idx - 1)
                
                observed_max_per_station = get_daily_station_max_obs(
                    target_date, args.obs_data_root, all_station_ids
                )
                
                if observed_max_per_station is None:
                    continue

                pred_day = pred_ppb[:, :, time_slice]
                reanalysis_day = true_ppb[:, :, time_slice]
                forecast_day = forecast_pred_ppb[:, :, time_slice]

                grid_rows, grid_cols, _ = pred_day.shape

                # --- Iterate through each station ---
                for station_id, details in station_details.items():
                    province_name = details['Province']
                    r_row, r_col = details['R_ROW'], details['R_COL']

                    # Check if station coordinates are within the model grid bounds
                    if not (r_row < grid_rows and r_col < grid_cols):
                        continue
                        
                    # Extract values from the specific grid cell for the station
                    pred_val = np.max(pred_day[r_row, r_col, :])
                    reanalysis_val = np.max(reanalysis_day[r_row, r_col, :])
                    forecast_val = np.max(forecast_day[r_row, r_col, :])
                    
                    observed_val = observed_max_per_station.get(station_id, np.nan)

                    if np.isnan(observed_val):
                        continue

                    daily_results.append({
                        'day': day_idx,
                        'province': province_name,
                        'station_id': station_id,
                        'Pred': pred_val,
                        'Reanalysis': reanalysis_val,
                        'Forecast': forecast_val,
                        'Observed': observed_val
                    })

    # --- Save results to CSV ---
    results_df = pd.DataFrame(daily_results)
    safe_name_part = args.model_name_short.replace('/', '_').replace('.pth', '')
    results_csv_path = f"daily_station_values_{safe_name_part}.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    print("\n" + "="*80)
    print(f"Station-based evaluation data saved to: {results_csv_path}")
    print("This file can now be used for station-wise metric calculations.")
    print("="*80)

if __name__ == '__main__':
    main()

