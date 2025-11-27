import os
import json
import numpy as np
import matplotlib.pyplot as plt
import validate_utils as utils

base_path = "results_red1"
plot_dir = "plots_red1/quantile"
os.makedirs(plot_dir, exist_ok=True)

metrics = {
    "RMSE": utils.RMSE,
    "MAE": utils.MAE,
    "IOA": utils.IOA,
    "R": utils.R,
    "NMB": utils.NMB,
    "BIAS": utils.BIAS,
}

def plot_metric(metric_name, split, num_bins=20):
    # dirs = [
    #     '5. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6 (new emb)',
    #     '6.3. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3 beta=1e4', 
    #     'diff_mse', 
    #     'diff_gamma3_beta1e2_p1', 
    #     'forecast', 
    #     ]
    # labels = [
    #     '1. MSE',
    #     '2. FocalLoss', 
    #     '3. MSE_Diff', 
    #     '4. FocalLoss_Diff', 
    #     'Forecast', 
    #     ]
    dirs = [
        'diff_mse', 
        'forecast'
        ]
    labels = [
        'MSE_Diff', 
        'Forecast'
        ]
    avg_dict_path = f"json/{split}_avg_dict.json"
    with open(avg_dict_path) as f:
        avg_dict = json.load(f)

    # cumulative plot
    plt.figure(figsize=(8, 6))
    for model_idx, name in enumerate(dirs):
        folder_path = os.path.join(base_path, name)
        if not os.path.isdir(folder_path):
            print(f"Directory not found: {folder_path}")
            continue
        json_file = os.path.join(folder_path, f"{split}_metrics.json")
        if not os.path.exists(json_file):
            print(f"json not found: {json_file}")
            continue

        with open(json_file) as f:
            results = json.load(f)

        sorted_dates = sorted(avg_dict, key=avg_dict.get, reverse=True)
        result_list = [results[d][metric_name] for d in sorted_dates]
        p_list = np.arange(len(result_list)) / len(result_list)
        color = "k" if "forecast" in name else None

        plt.plot(
            p_list,
            np.cumsum(result_list) / np.arange(1, len(result_list) + 1),
            label=labels[model_idx],
            color=color,
        )

    plt.xlabel("Quantile (high → low)")
    plt.ylabel(metric_name)
    plt.title(f"Cumulative Average {metric_name} ({split})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"cumulative_{metric_name.lower()}_{split}.png"))
    plt.close()

    # top 20%
    plt.figure(figsize=(8, 6))
    for model_idx, name in enumerate(dirs):
        folder_path = os.path.join(base_path, name)
        if not os.path.isdir(folder_path):
            continue
        json_file = os.path.join(folder_path, f"{split}_metrics.json")
        if not os.path.exists(json_file):
            continue

        with open(json_file) as f:
            results = json.load(f)

        sorted_dates = sorted(avg_dict, key=avg_dict.get, reverse=True)
        result_list = [results[d][metric_name] for d in sorted_dates]
        top_n = int(len(result_list) * 0.2)
        result_list = result_list[:top_n]
        p_list = np.arange(len(result_list)) / len(result_list) * 0.2
        color = "k" if "forecast" in name else None

        plt.plot(
            p_list,
            np.cumsum(result_list) / np.arange(1, len(result_list) + 1),
            label=labels[model_idx],
            color=color,
        )

    plt.xlabel("Quantile (top 20% high → low)")
    plt.ylabel(metric_name)
    plt.title(f"Cumulative Average {metric_name} (Top 20%) ({split})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"cumulative_{metric_name.lower()}_{split}_20pct.png"))
    plt.close()

    # quantile-conditioned plot
    plt.figure(figsize=(8, 6))
    for model_idx, name in enumerate(dirs):
        folder_path = os.path.join(base_path, name)
        if not os.path.isdir(folder_path):
            continue
        json_file = os.path.join(folder_path, f"{split}_metrics.json")
        if not os.path.exists(json_file):
            continue

        with open(json_file) as f:
            results = json.load(f)

        sorted_dates = sorted(avg_dict, key=avg_dict.get, reverse=True)
        y_true_sorted = np.array([avg_dict[d] for d in sorted_dates])

        q_edges = np.linspace(0, 1, num_bins + 1)
        vals, p_centers = [], []

        for i in range(num_bins):
            lo = int(len(y_true_sorted) * q_edges[i])
            hi = int(len(y_true_sorted) * q_edges[i + 1])
            if hi - lo == 0:
                continue
            v = np.mean([results[d][metric_name] for d in sorted_dates[lo:hi]])
            vals.append(v)
            p_centers.append((q_edges[i] + q_edges[i + 1]) / 2)

        color = "k" if "forecast" in name else None
        plt.plot(p_centers, vals, marker="o", label=labels[model_idx], color=color)

    plt.xlabel("Quantile (high → low)")
    plt.ylabel(metric_name)
    plt.title(f"Quantile-conditioned {metric_name}, Bins = {num_bins} ({split})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"quantile_{metric_name.lower()}_{split}.png"))
    plt.close()

for split in ["test"]:
    for m in metrics.keys():
        plot_metric(m, split)
