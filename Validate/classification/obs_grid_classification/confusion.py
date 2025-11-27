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

# --- New Imports ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Add project root to path to allow importing custom modules ---
# (Keeping user's original path logic)
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from Dataset.ozone_dataset import OzoneTestDataset
    from Models.ozone_informer.models.model import Informer
except ImportError:
    print("Could not import custom project modules. Assuming standalone execution.")
    pass


def calculate_metrics(df, day, bins, labels, pred="Pred", true="Reanalysis"):
    """Calculates and prints classification metrics for a given forecast day."""
    print(f"\n--- Metrics for Forecast Day {day} (Comparing {pred} vs {true}) ---")
    daily_total = df[df['day'] == day].copy()
    if daily_total.empty:
        print("No data available for this day.")
        return None

    # --- Use bins and labels passed as arguments ---
    daily_total['Reanalysis_c'] = pd.cut(daily_total[true], bins=bins, labels=labels, right=False)
    daily_total['Pred_c'] = pd.cut(daily_total[pred], bins=bins, labels=labels, right=False)

    categories = labels
    contingency = {}
    for pred_cat in categories:
        for real_cat in categories:
            count = ((daily_total['Pred_c'] == pred_cat) & (daily_total['Reanalysis_c'] == real_cat)).sum()
            contingency[f'{pred_cat}_vs_{real_cat}'] = count
    
    # Check for missing keys and set to 0 if not present (can happen with no data)
    keys = [f'{p}_vs_{r}' for p in categories for r in categories]
    for k in keys:
        if k not in contingency:
            contingency[k] = 0

    a1 = contingency['A_vs_A']; a2 = contingency['A_vs_B']; a3 = contingency['A_vs_C']; a4 = contingency['A_vs_D']
    b1 = contingency['B_vs_A']; b2 = contingency['B_vs_B']; b3 = contingency['B_vs_C']; b4 = contingency['B_vs_D']
    c1 = contingency['C_vs_A']; c2 = contingency['C_vs_B']; c3 = contingency['C_vs_C']; c4 = contingency['C_vs_D']
    d1 = contingency['D_vs_A']; d2 = contingency['D_vs_B']; d3 = contingency['D_vs_C']; d4 = contingency['D_vs_D']

    N = sum(contingency.values())
    eps = 1e-12

    # These quadrants are based on a 2x2 collapse where:
    # "Event" = C or D (>= 90)
    # "Non-Event" = A or B (< 90)
    #
    # Quadrant I (True Negative): Predicted (A,B), True (A,B)
    # Quadrant II (False Positive): Predicted (C,D), True (A,B)
    # Quadrant III (False Negative): Predicted (A,B), True (C,D)
    # Quadrant IV (True Positive): Predicted (C,D), True (C,D)

    I   = a1+a2+b1+b2  # TN
    II  = c1+c2+d1+d2  # FP
    III = a3+a4+b3+b4  # FN
    IV  = c3+c4+d3+d4  # TP

    ACC = 100 * (I + IV) / max(N, eps)
    POD = 100 * IV / max(III + IV, eps) # Also known as Recall or True Positive Rate
    FAR = 100 * II / max(II + IV, eps) # False Alarm Ratio
    F1  = 2 * (POD * (100 - FAR)) / max(POD + (100 - FAR), eps)
    CSI = IV / max(II + III + IV, eps) # Critical Success Index (Threat Score)

    print(f"Classification Results for Day{day} and Prediction '{pred}':")
    print(f"Total Samples: {N}")
    print(f"Accuracy (ACC): {ACC:.2f}%")
    print(f"Probability of Detection (POD): {POD:.2f}%")
    print(f"False Alarm Ratio (FAR): {FAR:.2f}%")
    print(f"F1 Score: {F1:.2f}")
    print(f"Critical Success Index (CSI): {CSI:.4f}")

    return {'ACC': ACC, 'POD': POD, 'FAR': FAR, 'F1': F1, 'CSI': CSI}


# --- NEW FUNCTION ---
def plot_confusion_matrices(df, day, bins, labels, file_part, pred_cols, true_col="Observed"):
    """
    Generates and saves a figure with confusion matrices for a given day.
    Each plot compares a column from pred_cols against the true_col.
    """
    print(f"\nGenerating confusion matrix plots for Day {day}...")
    
    daily_total = df[df['day'] == day].copy()
    if daily_total.empty:
        print(f"No data to plot for Day {day}.")
        return

    num_plots = len(pred_cols)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Create the categorical column for the true values
    daily_total[f'{true_col}_c'] = pd.cut(daily_total[true_col], bins=bins, labels=labels, right=False)
    true_cats = daily_total[f'{true_col}_c'].dropna()

    for ax, pred_col in zip(axes, pred_cols):
        # Create the categorical column for the predicted values
        daily_total[f'{pred_col}_c'] = pd.cut(daily_total[pred_col], bins=bins, labels=labels, right=False)
        pred_cats = daily_total[f'{pred_col}_c'].dropna()

        # Align the series to ensure we only compare valid, non-NaN pairs
        aligned_true, aligned_pred = true_cats.align(pred_cats, join='inner')

        if aligned_true.empty:
            ax.text(0.5, 0.5, "No overlapping data to plot", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_title(f'{pred_col} vs {true_col}', fontsize=14)
            continue
            
        # Calculate the confusion matrix
        cm = confusion_matrix(aligned_true, aligned_pred, labels=labels)

        # Plot the heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 12})
        
        ax.set_xlabel(f'{pred_col} Category', fontsize=12)
        ax.set_ylabel('Observation Category', fontsize=12)
        ax.set_title(f'{pred_col} vs {true_col}', fontsize=14)

    fig.suptitle(f'Frequency Counts: A[0,30), B[30,90), C[90,150), D[150+)', fontsize=16, y=1.05)
    plt.tight_layout()
    
    save_path = f'confusion_matrices_day_{day}_{file_part}.png'
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved confusion matrix plot to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    plt.close(fig)


def main():
    # load results csv from current directory
    # file = "daily_station_values_6. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed)_epoch99_loss0.00000_rmse0.00066.csv"
    file = "daily_station_values_diff_mse_epoch96_loss0.00009_rmse0.00922.csv"
    
    try:
        results_df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file}'.")
        print("Please make sure the CSV file is in the same directory as this script.")
        return

    # --- Define bins and labels here to pass to both functions ---
    bins = [0, 30, 90, 150, np.inf]
    labels = ['A', 'B', 'C', 'D']
    
    results = {}
    
    # Generate a safe file name part for output files
    safe_name_part = file.replace(".csv", "").replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    safe_name_part = safe_name_part.split('/')[-1] # Get just the filename part

    for day in [1, 2, 3]:
        # --- Calculate Metrics ---
        # Pass bins and labels to the metrics function
        results[f'pred_{day}'] = calculate_metrics(results_df, day, bins, labels, pred="Pred", true="Observed")
        results[f'forecast_{day}'] = calculate_metrics(results_df, day, bins, labels, pred="Forecast", true="Observed")
        results[f'reanalysis_{day}'] = calculate_metrics(results_df, day, bins, labels, pred="Reanalysis", true="Observed")

        # --- Generate Plots ---
        plot_confusion_matrices(
            results_df, 
            day, 
            bins, 
            labels, 
            file_part=safe_name_part,
            pred_cols=['Pred', 'Forecast', 'Reanalysis'], 
            true_col='Observed'
        )

    # Save results summary
    results_summary_path = f"summary_{safe_name_part}.txt"
    print(f"\nSaving metrics summary to {results_summary_path}")
    with open(results_summary_path, 'w') as f:
        for key, metrics in results.items():
            if metrics: # Only write if metrics were successfully calculated
                f.write(f"--- {key} ---\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
                f.write("\n")
            else:
                f.write(f"--- {key} ---\n")
                f.write("No data available for calculation.\n\n")

if __name__ == '__main__':
    main()
