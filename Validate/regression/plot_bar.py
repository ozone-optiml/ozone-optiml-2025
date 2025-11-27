import os
import json
import numpy as np
import matplotlib.pyplot as plt
import validate_utils as utils

base_path = "results"
plot_dir = "plots_final/bar"
os.makedirs(plot_dir, exist_ok=True)

metrics = ["RMSE", "MAE", "IOA", "R", "NMB", "BIAS"]

dirs = [
    '5. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6 (new emb)',
    '6.3. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3 beta=1e4', 
    'diff_mse', 
    'diff_gamma3_beta1e2_p1', 
    'forecast'
]
labels = [
    '1. MSE',
    '2. FocalLoss', 
    '3. MSE_Diff', 
    '4. FocalLoss_Diff', 
    'Forecast'
]
splits = ["val", "test"]

for split in splits:
    avg_values = {m: [] for m in metrics}

    for d in dirs:
        json_path = os.path.join(base_path, d, f"{split}_metrics.json")
        if not os.path.exists(json_path):
            avg = {m: np.nan for m in metrics}
        else:
            with open(json_path) as f:
                data = json.load(f)
            arr = {m: [] for m in metrics}
            for day in data.values():
                for m in metrics:
                    arr[m].append(day[m])
            avg = {m: np.mean(arr[m]) for m in metrics}
        for m in metrics:
            avg_values[m].append(avg[m])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, m in enumerate(metrics):
        ax = axes[i]
        ax.bar(labels, avg_values[m], color="skyblue")
        ax.set_title(f"{m}", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f"{split.upper()} Metrics", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plot_dir, f"{split}_metrics_all.png"), dpi=300)
    plt.close()
