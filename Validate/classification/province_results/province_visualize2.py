import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

def create_visualizations(file_name):
    """
    Loads the summary CSV and generates ONE wide plot per day
    using Seaborn and Matplotlib.
    
    Uses a custom color palette derived from the user's image:
    - ACC: Blue
    - POD: Orange
    - FAR: Green
    - F1:  Red
    - CSI: Purple
    
    Model_Pred is a darker shade, Baseline_Forecast is a lighter shade.
    """
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(file_name)
        print(f"Successfully loaded '{file_name}'")
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found in the current directory.")
        print("Please make sure the file is in the same folder as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # --- 2. Prepare Data for Plotting (Melt) ---
    id_vars = ['province', 'day', 'type']
    value_vars = ['ACC', 'POD', 'FAR', 'F1', 'CSI']
    
    required_cols = id_vars + value_vars
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: The CSV is missing the following required columns: {missing_cols}")
        sys.exit(1)

    df_melted = df.melt(id_vars=id_vars,
                        value_vars=value_vars,
                        var_name='Metric',
                        value_name='Value')

    # --- 3. Normalize Data (CRITICAL for single plot) ---
    # Check if CSI is on 0-1 scale. If so, multiply by 100
    if 'CSI' in df_melted['Metric'].unique():
        max_csi = df_melted.loc[df_melted['Metric'] == 'CSI', 'Value'].max()
        # Check if max_csi is low (e.g., <= 1.0) and not zero
        if max_csi > 0 and max_csi <= 1.0:
            print("CSI appears to be on a 0-1 scale. Scaling by 100 for visualization.")
            df_melted.loc[df_melted['Metric'] == 'CSI', 'Value'] *= 100
            
        # Rename for clarity in the legend
        # df_melted.loc[df_melted['Metric'] == 'CSI', 'Metric'] = 'CSI (x100)'

    # --- 4. Create Grouping and CUSTOM Color Palette ---
    
    # Define the order of metrics
    metrics_order = ['ACC', 'POD', 'FAR', 'F1', 'CSI']
    types_order = ['Model_Pred', 'Baseline_Forecast']
    
    # Create a new column for the 10-bar grouping
    df_melted['Metric_Type'] = df_melted['Metric'] + ' - ' + df_melted['type']
    
    # Define base color palettes from the image (light/dark pairs)
    base_colors = {
        'ACC': sns.color_palette("Blues", 2),
        'POD': sns.color_palette("Oranges", 2),
        'FAR': sns.color_palette("Greens", 2),
        'F1': sns.color_palette("Reds", 2),
        'CSI' : sns.color_palette("Purples", 2)
    }
    
    hue_order = []
    palette_map = {}
    
    for metric in metrics_order:
        # Ensure the metric exists in our color map
        if metric in base_colors:
            colors = base_colors[metric]
            for type_name in types_order:
                key = f"{metric} - {type_name}"
                hue_order.append(key)
                
                # Map Model_Pred to the darker shade [1]
                # Map Baseline_Forecast to the lighter shade [0]
                if type_name == 'Model_Pred':
                    palette_map[key] = colors[1] # Darker
                else:
                    palette_map[key] = colors[0] # Lighter
        else:
            # Fallback for any metrics not defined (like the original CSI)
            pass

    # --- 5. Generate Charts (One per Day) ---
    days = sorted(df_melted['day'].unique())
    print(f"Found data for days: {days}. Generating {len(days)} chart(s)...")

    for day in days:
        print(f"Generating chart for Day {day}...")
        
        # Filter data for the specific day
        day_data = df_melted[df_melted['day'] == day]

        # Create the single, wide faceted chart
        g = sns.catplot(
            data=day_data,
            kind='bar',
            x='province',     # Provinces on the x-axis
            y='Value',        # Metric value on the y-axis
            hue='Metric_Type',# Use the combined 10-value column for hue
            hue_order=hue_order,  # Enforce our metric/type order
            palette=palette_map,  # Use our custom light/dark palette
            height=7,         # Taller plot
            aspect=2.5,       # Wider plot (2.5x width to height)
            legend_out=True   # Move the legend outside the plot
        )

        # --- Chart refinements ---
        
        # Set a main title for the entire figure
        g.fig.suptitle(f'By Province Metrics Comparison for Day {day}', y=1.03, fontsize=16)
        
        # Rotate x-axis labels for better readability
        g.set_xticklabels(rotation=45, horizontalalignment='right')
        
        # Set axis labels
        g.set_axis_labels("Province", "Metric Value (0-100 Scale)")
        
        # Clean up the legend title
        g.legend.set_title("Metric - Type")

        # Save the chart as a PNG image file
        chart_filename = f'day_{day}_metrics_custom_color.png'
        
        # Use bbox_inches='tight' to prevent labels from being cut off
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close(g.fig) # Close the figure to free up memory
        
        print(f"Successfully saved chart to {chart_filename}")

    print("\nAll charts generated successfully.")

# --- Main execution ---
if __name__ == "__main__":
    # Define the file name from your prompt
    CSV_FILE = "summary_by_province_diff_mse_epoch96_loss0.00009_rmse0.00922.csv"
    
    # Run the function
    create_visualizations(CSV_FILE)