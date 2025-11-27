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
import matplotlib.pyplot as plt  # Added for visualization

# --- Add project root to path to allow importing custom modules ---
# (Path logic remains as you provided)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Check if the path is correct, assuming 'Dataset' and 'Models' are top-level dirs in project_root
# print(f"Project Root (added to sys.path): {project_root}")
# print(f"Sys.path: {sys.path}")

# --- Original Imports (with error handling) ---
try:
    from Dataset.ozone_dataset import OzoneTestDataset
    from Models.ozone_informer.models.model import Informer
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure the project root is correctly identified and added to sys.path.")
    print(f"Calculated project root: {project_root}")
    # Exit or continue depending on whether these modules are strictly necessary for this script
    # For this script, they seem unnecessary, so we'll just print the warning.
    pass


def calculate_metrics(df, day, pred="Pred", true="Reanalysis"):
    """Calculates and prints classification metrics for a given forecast day."""
    print(f"\n--- Metrics for Forecast Day {day} (Pred: {pred}, True: {true}) ---")
    daily_total = df[df['day'] == day].copy()
    if daily_total.empty:
        print("No data available for this day.")
        return None

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
    if N == 0:
        print("No matching samples found for contingency table.")
        return None
        
    eps = 1e-12

    I   = a1+a2+b1+b2
    II  = c1+c2+d1+d2
    III = a3+a4+b3+b4
    IV  = c3+c4+d3+d4

    ACC = 100 * (I + IV) / max(N, eps)
    POD = 100 * IV / max(III + IV, eps)
    FAR = 100 * II / max(II + IV, eps)
    F1  = 2 * (POD * (100 - FAR)) / max(POD + (100 - FAR), eps)
    CSI = 100 * IV / max(II + III + IV, eps)  # <-- Changed to percentage

    print(f"Classification Results for Day{day} and Prediction '{pred}':")
    print(f"Total Samples: {N}")
    print(f"Accuracy (ACC): {ACC:.2f}%")
    print(f"Probability of Detection (POD): {POD:.2f}%")
    print(f"False Alarm Ratio (FAR): {FAR:.2f}%")
    print(f"F1 Score: {F1:.2f}%") # F1 is also 0-100
    print(f"Critical Success Index (CSI): {CSI:.2f}%") # <-- Updated print format

    return {'ACC': ACC, 'POD': POD, 'FAR': FAR, 'F1': F1, 'CSI': CSI}

def plot_day1_metrics(results, source_csv_file):
    """
    Visualizes the Day 1 metrics using a single-panel grouped bar chart.
    'results' is the dictionary populated in main().
    'source_csv_file' is the string filename of the CSV.
    """
    print("\nGenerating Day 1 metrics plot...")
    
    # 1. Prepare data
    plot_data = {}
    if 'pred_1' in results and results['pred_1'] is not None:
        plot_data['Model Pred'] = results['pred_1']
    if 'forecast_1' in results and results['forecast_1'] is not None:
        plot_data['Forecast'] = results['forecast_1']
    if 'reanalysis_1' in results and results['reanalysis_1'] is not None:
        plot_data['Reanalysis'] = results['reanalysis_1']
    
    if not plot_data:
        print("No valid Day 1 results ('pred_1', 'forecast_1', 'reanalysis_1') found to plot.")
        return
        
    df = pd.DataFrame(plot_data)
    
    # 2. All metrics are now on 0-100 scale
    percent_metrics = ['ACC', 'POD', 'FAR', 'F1', 'CSI'] # <-- Added CSI
    
    # Filter metrics that are actually present in the dataframe's index
    available_percent_metrics = [m for m in percent_metrics if m in df.index]

    if not available_percent_metrics:
        print("Missing metrics (ACC, POD, FAR, F1, CSI) in results. Cannot plot.")
        return

    df_percent = df.loc[available_percent_metrics]

    # 3. Create plot (now a single plot)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 7), sharex=False)
    # fig.suptitle(f'Day 1 Station-Wise Metrics Comparison (vs. Observed)', fontsize=16, y=1.02)

    # --- Plot 1: All Metrics ---
    df_percent.plot(kind='bar', ax=ax1, rot=0, colormap='Set2', width=0.8)
    ax1.set_title('Day 1 Station-Wise Metrics Comparison (vs. Observed)', fontsize=14)
    ax1.set_ylabel('Value (%)')
    ax1.set_ylim(0, 105) # Give a little padding
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(title='Prediction Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add value labels for Plot 1
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1f}", 
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', 
                     xytext=(0, 9), 
                     textcoords='offset points',
                     fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 0.85, 0.93]) # Adjust for suptitle and legend
    
    # 4. Save figure
    # Use the same safe name from main()
    safe_name_part = source_csv_file.replace("total_results/daily_station_values_", "").replace(" ", "_").replace(",", "_").replace("(", "").replace(")", "").replace(".csv", "")
    fig_path = f"metrics_{safe_name_part}.png"
    
    try:
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Metrics plot saved to {fig_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # plt.show() # Optional: uncomment to display plot during execution

def main():
    # load results csv from current directory
    # file = "daily_station_values_6. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed)_epoch99_loss0.00000_rmse0.00066.csv"
    file = "daily_station_values_diff_mse_epoch96_loss0.00009_rmse0.00922_red4.csv"
    
    try:
        results_df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file}'.")
        print("Please make sure the file is in the same directory as the script.")
        return

    results = {}
    # Only calculate for Day 1 as requested
    for day in [1]:
        print(f"\n--- Calculating metrics for Day {day} ---")
        results[f'pred_{day}'] = calculate_metrics(results_df, day, pred="Pred", true="Observed")
        results[f'forecast_{day}'] = calculate_metrics(results_df, day, pred="Forecast", true="Observed")
        results[f'reanalysis_{day}'] = calculate_metrics(results_df, day, pred="Reanalysis", true="Observed")

    # Save results to text file
    safe_name_part = file.replace("total_results/daily_station_values_", "").replace(" ", "_").replace(",", "_").replace("(", "").replace(")", "").replace(".csv", "")
    results_summary_path = f"summary_{safe_name_part}.txt"
    
    print(f"\nSaving metrics summary to {results_summary_path}...")
    with open(results_summary_path, 'w') as f:
        for key, metrics in results.items():
            if metrics is not None:
                f.write(f"--- {key} ---\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n") # Still save as 4 decimal places in text file for precision
                f.write("\n")
            else:
                f.write(f"--- {key} ---\nNo data\n\n")
    print("Summary saved.")

    # --- NEW: Call the visualization function ---
    plot_day1_metrics(results, file)

if __name__ == '__main__':
    main()

