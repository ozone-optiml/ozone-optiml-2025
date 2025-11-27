import pandas as pd
import numpy as np
import argparse
import os

def calculate_metrics(df, day, pred="Pred", true="Reanalysis", province_name="Global"):
    """
    Calculates and prints classification metrics for a given forecast day
    and specific province.
    """
    print(f"\n--- Metrics for Province: {province_name}, Forecast Day {day}, Type: {pred} ---")
    
    # Filter for the specific day
    daily_total = df[df['day'] == day].copy()
    
    if daily_total.empty:
        print("No data available for this day and province.")
        # Return NaNs if no data, so the summary file is complete
        return {'ACC': np.nan, 'POD': np.nan, 'FAR': np.nan, 'F1': np.nan, 'CSI': np.nan}

    # Define bins and labels for classification
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

    if N == 0:
        print("No valid samples found for contingency table.")
        return {'ACC': np.nan, 'POD': np.nan, 'FAR': np.nan, 'F1': np.nan, 'CSI': np.nan}

    I   = a1+a2+b1+b2
    II  = c1+c2+d1+d2
    III = a3+a4+b3+b4
    IV  = c3+c4+d3+d4

    ACC = 100 * (I + IV) / max(N, eps)
    POD = 100 * IV / max(III + IV, eps)
    FAR = 100 * II / max(II + IV, eps)
    F1  = 2 * (POD * (100 - FAR)) / max(POD + (100 - FAR), eps)
    CSI = IV / max(II + III + IV, eps)

    print(f"Total Samples: {N}")
    print(f"Accuracy (ACC): {ACC:.2f}%")
    print(f"Probability of Detection (POD): {POD:.2f}%")
    print(f"False Alarm Ratio (FAR): {FAR:.2f}%")
    print(f"F1 Score: {F1:.2f}")
    print(f"Critical Success Index (CSI): {CSI:.4f}")

    return {'ACC': ACC, 'POD': POD, 'FAR': FAR, 'F1': F1, 'CSI': CSI}

def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate province-specific metrics from a directory of daily_maximum CSVs.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="total_results",
        required=False, 
        help="Path to the directory containing 'daily_maximum_province_...' CSV files to evaluate."
    )
    parser.add_argument(
        "--provinces_file", 
        type=str, 
        default="provinces_reduction_4.txt",
        help="Path to the .txt file containing province names. Defaults to 'provincess_reduction_4.txt'."
    )
    args = parser.parse_args()
    return args

def process_csv_file(csv_filepath, provinces):
    """
    Loads a single CSV file, calculates metrics for each province and day, 
    and saves a summary file.
    """
    print(f"\n{'='*80}")
    print(f"STARTING EVALUATION FOR FILE: {csv_filepath}")
    print(f"{'='*80}")

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return

    all_metrics_results = []
    count_results = []
    reanalysis_col = 'Reanalysis'
    bad_threshold = 90
    
    # --- Iterate and Evaluate ---
    for province in provinces:
        print(f"\n--- Processing Province: {province} ---")
        
        # Filter the main DataFrame for the current province
        df_province = df[df['province'] == province].copy()
        
        if df_province.empty:
            print(f"No data found for province: {province}. Skipping.")
            continue

        for day in [1, 2, 3]:
            # --- 1. CLASSIFICATION METRICS (Original Logic) ---
            
            # Evaluate Model Prediction
            pred_metrics = calculate_metrics(df_province, day, pred="Pred", true="Reanalysis", province_name=province)
            pred_metrics['province'] = province
            pred_metrics['day'] = day
            pred_metrics['type'] = 'Model_Pred'
            all_metrics_results.append(pred_metrics)
            
            # Evaluate Baseline Forecast
            forecast_metrics = calculate_metrics(df_province, day, pred="Forecast", true="Reanalysis", province_name=province)
            forecast_metrics['province'] = province
            forecast_metrics['day'] = day
            forecast_metrics['type'] = 'Baseline_Forecast'
            all_metrics_results.append(forecast_metrics)

            # --- 2. BAD/VERY BAD REANALYSIS COUNT (New Logic) ---
            daily_data = df_province[df_province['day'] == day]
            
            # Count where Reanalysis >= 90 (Categories C and D)
            bad_count = (daily_data[reanalysis_col] >= bad_threshold).sum()
            total_samples = len(daily_data)

            count_results.append({
                'province': province,
                'day': day,
                'bad_or_very_bad_count': bad_count,
                'total_samples': total_samples
            })
            
    # --- 3. Save Summary Results (Classification Metrics) ---
    metrics_df = pd.DataFrame(all_metrics_results)
    
    # Reorder columns for clarity
    cols = ['province', 'day', 'type', 'ACC', 'POD', 'FAR', 'F1', 'CSI']
    metrics_df = metrics_df[cols]
    
    # Create a summary filename based on the input CSV name
    base_name = os.path.basename(csv_filepath)
    metrics_summary_filename = base_name.replace('daily_maximum_province_', 'summary_by_province_')
    
    metrics_df.to_csv(metrics_summary_filename, index=False)
    print(f"\n[DONE] Classification metrics summary saved to: {metrics_summary_filename}")


    # --- 4. Save Count Results (New Count Data) ---
    counts_df = pd.DataFrame(count_results)
    
    # Create a count summary filename
    count_summary_filename = base_name.replace('daily_maximum_province_', 'counts_reanalysis_')
    
    counts_df.to_csv(count_summary_filename, index=False)
    print(f"[DONE] Bad/Very Bad Reanalysis count summary saved to: {count_summary_filename}")


def main():
    args = get_args()

    # --- 1. Load Provinces File ---
    if not os.path.exists(args.provinces_file):
        print(f"Error: Provinces file not found at {args.provinces_file}")
        print(f"Please ensure the default file '{args.provinces_file}' exists or provide a custom file path.")
        return

    print(f"Loading provinces from {args.provinces_file}...")
    try:
        with open(args.provinces_file, 'r') as f:
            provinces = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading provinces file: {e}")
        return

    # --- 2. Iterate through CSV files in the input directory ---
    input_dir = args.input_dir

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found or is not a directory: {input_dir}")
        return

    print(f"\nScanning directory: {input_dir} for CSV files...")
    
    # Filter for files that end with '.csv' and start with 'daily_maximum_province_'
    csv_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.csv') and f.startswith('daily_maximum_province_')
    ]
    
    if not csv_files:
        print(f"No CSV files starting with 'daily_maximum_province_' found in {input_dir}. Exiting.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")
    
    for csv_file_path in csv_files:
        try:
            process_csv_file(csv_file_path, provinces)
        except Exception as e:
            print(f"An unexpected error occurred while processing {csv_file_path}: {e}")
            continue

    print("\n\nAll evaluations complete.")


if __name__ == '__main__':
    main()
