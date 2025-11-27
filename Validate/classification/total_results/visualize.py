import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read all summary files
summary_files = [f for f in os.listdir('.') if f.startswith('summary') and f.endswith('.txt')]
summary_files.sort()  # ensure consistent order

# Regular expression to capture each section
section_pattern = re.compile(r'--- (pred_\d+|forecast_\d+) ---\s*(.*?)\n(?=---|$)', re.S)
metric_pattern = re.compile(r'(\w+):\s*([\d.]+)')

# Step 2: Parse files
data = {}

for filename in summary_files:
    with open(filename, 'r') as f:
        text = f.read()
    matches = section_pattern.findall(text)
    model_name = filename.replace('summary', '').replace('.txt', '').strip('_')[:22]
    if not model_name:
        model_name = filename
    for section, block in matches:
        metrics = dict(metric_pattern.findall(block))
        metrics = {k: float(v) for k, v in metrics.items()}
        data.setdefault(section, {})[model_name] = metrics

# Step 3: Prepare forecast (same across all files)
forecast_data = {}
for k in list(data.keys()):
    if 'forecast' in k:
        forecast_data[k] = next(iter(data[k].values()))  # take one (they are identical)

# Step 4: Plot each pair
pairs = [('pred_1', 'forecast_1'), ('pred_2', 'forecast_2'), ('pred_3', 'forecast_3')]
metrics = ['ACC', 'POD', 'FAR', 'F1', 'CSI']

os.makedirs('comparison_plots', exist_ok=True)

for pred_key, forecast_key in pairs:
    df = pd.DataFrame(data[pred_key]).T[metrics]
    # Multiply CSI by 100 for better visibility
    df.loc['Forecast'] = [forecast_data[forecast_key][m] for m in metrics]
    df['CSI'] = df['CSI'] * 100

    # Plot
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', width=0.8)
    plt.title(f'Comparison of Models on Day{pred_key[-1]}')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'comparison_plots/{pred_key}_comparison.png', dpi=300)
    plt.close()

print("Plots saved in the 'comparison_plots/' directory.")
