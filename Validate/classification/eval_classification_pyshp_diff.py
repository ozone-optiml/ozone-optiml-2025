import sys
import os
import argparse
from collections import OrderedDict, Counter
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import netCDF4
import shapefile  # Using pyshp
from tqdm import tqdm

# --- Add project root to path to allow importing custom modules ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Dataset.ozone_dataset import OzoneTestDataset
from Models.ozone_informer.models.model import Informer


def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the Informer model for ozone classification.")
    parser.add_argument("--root_dir", type=str, default="/storage/dataset/ozone/NIER_AI_v8/Dataset", help="Root directory of the dataset")
    parser.add_argument("--model_name", type=str, default="6. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed)/epoch99_loss0.00000_rmse0.00066.pth", help="Name of the trained model file")
    parser.add_argument("--grid_info_path", type=str, default="GRID_INFO_09km.nc", help="Path to the NetCDF grid information file.")
    parser.add_argument("--shapefile_path", type=str, default="gadm41_KOR_1.dbf", help="Path to the South Korea provinces .shp file.")
    parser.add_argument("--e_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--d_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model Dimension")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward layer dimension")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--reduction", type=int, default=4, help="Spatial reduction factor")
    args = parser.parse_args()
    
    # Construct full model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.model_name_short = args.model_name
    args.model_name = os.path.join(script_dir, "../../Train/checkpoints", args.model_name)
    
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
        # new_k = k.removeprefix("module.")
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    return model


def is_point_in_path(x, y, path_vertices):
    """
    Checks if a point is inside a polygon using the Ray Casting algorithm.
    """
    num_vertices = len(path_vertices)
    if num_vertices == 0:
        return False
    
    is_inside = False
    p1x, p1y = path_vertices[0]
    for i in range(num_vertices + 1):
        p2x, p2y = path_vertices[i % num_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersection:
                        is_inside = not is_inside
        p1x, p1y = p2x, p2y
    return is_inside


def create_or_load_province_map(grid_info_path, shapefile_path, reduction_factor, force_recompute=False):
    """
    Creates a 2D map linking reduced grid cells to provinces using the 'pyshp' library.
    The map is cached to a .npy file for faster loading.
    """
    cache_file = f"province_map_reduction_{reduction_factor}.npy"
    provinces_file = f"provinces_reduction_{reduction_factor}.txt"

    if not force_recompute and os.path.exists(cache_file) and os.path.exists(provinces_file):
        print(f"Loading cached province map from {cache_file}")
        province_grid = np.load(cache_file, allow_pickle=True)
        with open(provinces_file, 'r') as f:
            provinces = [line.strip() for line in f]
        return province_grid, provinces

    print("Creating new province map with 'pyshp' (this may take a few minutes)...")
    
    # 1. Load grid coordinates
    with netCDF4.Dataset(grid_info_path) as grid_nc:
        lat = grid_nc.variables['LAT'][:]
        lon = grid_nc.variables['LON'][:]
    
    # 2. Load province shapefile using pyshp
    sf = shapefile.Reader(shapefile_path)
    shapes = sf.shapes()
    records = sf.records()
    print(records[1])
    # Find the index for the province name field (e.g., 'NAME_1')
    field_names = [field[0] for field in sf.fields[1:]]
    try:
        name_idx = field_names.index('NAME_1')
    except ValueError:
        raise ValueError("Shapefile must contain a 'NAME_1' field for province names.")

    # 3. Create full-resolution province map
    # Assuming lat/lon might be 4D, slice to get the 2D grid
    if lat.ndim == 4:
        lat = lat[0, 0, :, :]
        lon = lon[0, 0, :, :]

    rows, cols = lat.shape
    full_province_map = np.empty(lat.shape, dtype=object)
    
    for r in tqdm(range(rows), desc="Mapping full grid"):
        for c in range(cols):
            point_lon, point_lat = lon[r, c], lat[r, c]
            province_found = False
            for i, shape in enumerate(shapes):
                # A shape can have multiple parts (e.g., islands)
                parts = list(shape.parts) + [len(shape.points)]
                in_any_part = False
                for j in range(len(parts) - 1):
                    path = shape.points[parts[j]:parts[j+1]]
                    if is_point_in_path(point_lon, point_lat, path):
                        in_any_part = True
                        break
                if in_any_part:
                    full_province_map[r, c] = records[i][name_idx]
                    province_found = True
                    break
            if not province_found:
                full_province_map[r, c] = 'Ocean'

    # 4. Reduce the map by finding the mode in each reduction block
    r_reduced = (rows + reduction_factor - 1) // reduction_factor
    c_reduced = (cols + reduction_factor - 1) // reduction_factor
    reduced_province_map = np.empty((r_reduced, c_reduced), dtype=object)

    for r in tqdm(range(r_reduced), desc="Reducing province map"):
        for c in range(c_reduced):
            r_start, c_start = r * reduction_factor, c * reduction_factor
            block = full_province_map[r_start:r_start+reduction_factor, c_start:c_start+reduction_factor]
            counts = Counter(block.flatten())
            counts.pop('Ocean', None)
            if counts:
                mode_province = counts.most_common(1)[0][0]
                reduced_province_map[r, c] = mode_province
            else:
                reduced_province_map[r, c] = 'Ocean'

    # 5. Get unique provinces and save
    provinces = sorted([p for p in pd.unique(reduced_province_map.flatten()) if p != 'Ocean'])
    
    np.save(cache_file, reduced_province_map)
    with open(provinces_file, 'w') as f:
        for p in provinces:
            f.write(f"{p}\n")
            
    print(f"Province map saved to {cache_file}")
    return reduced_province_map, provinces


def calculate_metrics(df, day, pred="Pred", true="Reanalysis"):
    """Calculates and prints classification metrics for a given forecast day."""
    print(f"\n--- Metrics for Forecast Day {day} ---")
    daily_total = df[df['day'] == day].copy()
    if daily_total.empty:
        print("No data available for this day.")
        return

    bins = [0, 30, 90, 150, np.inf]
    labels = ['A', 'B', 'C', 'D']
    
    daily_total['Reanalysis_c'] = pd.cut(daily_total[true], bins=bins, labels=labels, right=False)
    daily_total['Pred_c'] = pd.cut(daily_total[pred], bins=bins, labels=labels, right=False)

    categories = ['A', 'B', 'C', 'D']
    contingency = {}
    for pred_cat in categories:
        for real_cat in categories:
            count = ((daily_total['Pred_c'] == pred_cat) & (daily_total['Reanalysis_c'] == real_cat)).sum()
            contingency[f'{pred_cat}_vs_{real_cat}'] = count
    
    a1 = contingency['A_vs_A']; a2 = contingency['A_vs_B']; a3 = contingency['A_vs_C']; a4 = contingency['A_vs_D']
    b1 = contingency['B_vs_A']; b2 = contingency['B_vs_B']; b3 = contingency['B_vs_C']; b4 = contingency['B_vs_D']
    c1 = contingency['C_vs_A']; c2 = contingency['C_vs_B']; c3 = contingency['C_vs_C']; c4 = contingency['C_vs_D']
    d1 = contingency['D_vs_A']; d2 = contingency['D_vs_B']; d3 = contingency['D_vs_C']; d4 = contingency['D_vs_D']

    N = sum(contingency.values())
    eps = 1e-12

    I   = a1+a2+b1+b2
    II  = c1+c2+d1+d2
    III = a3+a4+b3+b4
    IV  = c3+c4+d3+d4

    ACC = 100 * (I + IV) / max(N, eps)
    POD = 100 * IV / max(III + IV, eps)
    FAR = 100 * II / max(II + IV, eps)
    F1  = 2 * (POD * (100 - FAR)) / max(POD + (100 - FAR), eps)
    CSI = IV / max(II + III + IV, eps)

    print(f"Classification Results for Day{day} and Prediction '{pred}':")
    print(f"Total Samples: {N}")
    print(f"Accuracy (ACC): {ACC:.2f}%")
    print(f"Probability of Detection (POD): {POD:.2f}%")
    print(f"False Alarm Ratio (FAR): {FAR:.2f}%")
    print(f"F1 Score: {F1:.2f}")
    print(f"Critical Success Index (CSI): {CSI:.4f}")

    return {'ACC': ACC, 'POD': POD, 'FAR': FAR, 'F1': F1, 'CSI': CSI}

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    province_grid, provinces = create_or_load_province_map(
        args.grid_info_path, args.shapefile_path, args.reduction
    )
    
    # normalization_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../Dataset/normalization_reduction_{args.reduction}.pt')
    # normalization_info = torch.load(normalization_path)
    # mean_o3 = normalization_info['mean'][10].item()
    # std_o3 = normalization_info['std'][10].item()

    model = load_model(args, device)
    output_subdir = "OutputDifference"
    test_dataset = OzoneTestDataset(args.root_dir, reduction=args.reduction, output_subdir=output_subdir)
    
    if len(test_dataset) > 0:
        _, _, sample_true_output, _ = test_dataset[0]
        # Permute from (C, R, T) to (R, C, T) to get model's spatial orientation
        sample_true_output_permuted = sample_true_output.permute(1, 0, 2)
        model_rows, model_cols = sample_true_output_permuted.shape[:2]

        # If shapes don't match, crop the province grid.
        if province_grid.shape[0] != model_rows or province_grid.shape[1] != model_cols:
            print(f"Warning: Province grid shape {province_grid.shape} differs from model output shape ({model_rows}, {model_cols}).")
            print("Cropping province grid to match model output.")
            province_grid = province_grid[:model_rows, :model_cols]

    daily_results = []

    normalization_path = os.path.join(os.path.dirname(__file__), f'../../Dataset/normalization_reduction_{args.reduction}.pt')
    normalization_info = torch.load(normalization_path)
    mean = normalization_info['mean'][10].item()
    std = normalization_info['std'][10].item()

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating Test Dataset"):
            encoder_input, decoder_input, true_output, _ = test_dataset[i]

            encoder_input = encoder_input.unsqueeze(0).to(device)
            decoder_input = decoder_input.unsqueeze(0).to(device)

            forecast_pred = encoder_input[:, :, :, -72:, 10]
            forecast_pred = forecast_pred.squeeze(0).squeeze(-1).cpu().numpy()

            forecast_pred = (forecast_pred * std) + mean
            forecast_pred = forecast_pred.transpose(1, 0, 2) # Final Shape: (R, C, T)

            pred = model(encoder_input, decoder_input)
            
            pred = pred.permute(0, 2, 1, 3).squeeze(0).squeeze(-1).cpu().numpy()
            true_output = true_output.permute(1, 0, 2).cpu().numpy()
            pred = pred + forecast_pred
            true_output = true_output + forecast_pred

            pred_ppb = pred * 1000
            true_ppb = true_output * 1000
            forecast_pred_ppb = forecast_pred * 1000
            
            day_slices = {1: slice(0, 24), 2: slice(24, 48), 3: slice(48, 72)}

            for day_idx, time_slice in day_slices.items():
                pred_day = pred_ppb[:, :, time_slice]
                true_day = true_ppb[:, :, time_slice]
                forecast_day = forecast_pred_ppb[:, :, time_slice]

                for province_name in provinces:
                    mask = (province_grid == province_name)
                    if not np.any(mask):
                        continue
                    
                    pred_max = np.max(pred_day[mask])
                    true_max = np.max(true_day[mask])
                    forecast_max = np.max(forecast_day[mask])

                    daily_results.append({
                        'day': day_idx,
                        'province': province_name,
                        'Pred': pred_max,
                        'Reanalysis': true_max,
                        'Forecast': forecast_max
                    })

                    print(f"Day {day_idx}, Province: {province_name}, Predicted Max: {pred_max:.2f}, Reanalysis Max: {true_max:.2f}, Forecast Max: {forecast_max:.2f}")

    results_df = pd.DataFrame(daily_results)
    # Save results to CSV
    safe_name_part = args.model_name_short.replace('/', '_').replace('.pth', '')
    results_csv_path = f"daily_maximum_province_{safe_name_part}.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nClassification results saved to {results_csv_path}")
    
    results = {}
    for day in [1, 2, 3]:
        results[f'pred_{day}'] = calculate_metrics(results_df, day)
        results[f'forecast_{day}'] = calculate_metrics(results_df, day, pred="Forecast", true="Reanalysis")

    # Save results
    results_summary_path = f"summary_{safe_name_part}.txt"
    with open(results_summary_path, 'w') as f:
        for key, metrics in results.items():
            f.write(f"--- {key} ---\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
if __name__ == '__main__':
    main()

