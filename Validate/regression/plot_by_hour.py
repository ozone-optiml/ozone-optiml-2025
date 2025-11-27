import os
import json
import numpy as np
import matplotlib.pyplot as plt

base_dir = "results"
group_24h = False
plot_parent_dir = "plots_final"
plot_dir = f"{plot_parent_dir}/by_hour_grouped" if group_24h else f"{plot_parent_dir}/by_hour"
os.makedirs(plot_dir, exist_ok=True)

dirs = [
    'forecast',
    'diff_gamma3_beta1e2_p1',
    '6.3. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3 beta=1e4',
    'diff_mse',
    '5. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6 (new emb)'
]
labels = ['Forecast', 'Diff (Focal)', 'Direct (Focal)', 'Diff (MSE)', 'Direct (MSE)']

splits = ["val", "test"]
months = ["01", "07"]
metrics = ["RMSE", "MAE", "IOA", "R", "NMB", "BIAS"]

for split in splits:
    for month in months:
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            for model_idx, name in enumerate(dirs):
                model_path = os.path.join(base_dir, name)
                if not os.path.isdir(model_path):
                    continue
                json_path = os.path.join(model_path, f"{split}_{month}_by_hour.json")
                if not os.path.exists(json_path):
                    continue

                with open(json_path) as f:
                    data = json.load(f)

                hours = np.arange(72)
                vals = []
                for h in hours:
                    vals_h = [data[d][metric][str(h)] for d in data if metric in data[d]]
                    vals.append(np.mean(vals_h))

                if group_24h:
                    vals = np.array(vals).reshape(3, 24).mean(axis=1)
                    x = np.array([12, 36, 60])
                else:
                    x = hours

                color = "k" if "forecast" in name else None
                plt.plot(x, vals, marker="o" if group_24h else None, label=labels[model_idx], color=color)

            plt.xlabel("Forecast hour group (24h avg)" if group_24h else "Forecast hour")
            plt.ylabel(metric)
            plt.title(f"{metric} ({'24h avg' if group_24h else 'hourly'}) - {split.upper()} {month}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            suffix = "_24havg" if group_24h else "_72h"
            out_path = os.path.join(plot_dir, f"{split}_{month}_{metric}{suffix}.png")
            plt.savefig(out_path)
            plt.close()