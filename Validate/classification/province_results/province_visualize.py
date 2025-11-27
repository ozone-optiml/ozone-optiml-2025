import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

def create_visualizations(file_name):
    """
    Loads the summary CSV and generates one chart per day using
    Seaborn and Matplotlib.
    
    Compares Model_Pred vs. Baseline_Forecast across all metrics
    and provinces.
    """
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(file_name)
        print(f"Successfully loaded '{file_name}'")
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found in the current directory.")
        print("Please make sure the file is in the same folder as this script.")
        sys.exit(1) # Exit the script if file not found
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # --- 2. Prepare Data for Plotting (Melt) ---
    # This "melts" the dataframe to make it "long" format, 
    # which is ideal for Seaborn.
    
    id_vars = ['province', 'day', 'type']
    value_vars = ['ACC', 'POD', 'FAR', 'F1', 'CSI']
    
    # Check if all required columns exist
    required_cols = id_vars + value_vars
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: The CSV is missing the following required columns: {missing_cols}")
        sys.exit(1)

    df_melted = df.melt(id_vars=id_vars,
                        value_vars=value_vars,
                        var_name='Metric',
                        value_name='Value')

    # --- 3. Generate Charts (One per Day) ---
    days = sorted(df_melted['day'].unique())
    print(f"Found data for days: {days}. Generating {len(days)} chart(s)...")

    for day in days:
        print(f"Generating chart for Day {day}...")
        
        # Filter data for the specific day
        day_data = df_melted[df_melted['day'] == day]

        # Create the faceted chart using Seaborn's catplot
        # This one command creates the grid of plots
        g = sns.catplot(
            data=day_data,
            kind='bar',       # Use a bar chart
            x='province',     # Provinces on the x-axis
            y='Value',        # Metric value on the y-axis
            hue='type',       # Group bars by 'type' (Model vs Baseline)
            col='Metric',     # Create separate columns (subplots) for each Metric
            col_wrap=3,       # Wrap to a new row after 3 subplots
            sharey=False,     # CRITICAL: Each metric has its own y-axis scale
            height=4,         # Height of each subplot
            aspect=1.2,       # Aspect ratio of each subplot
            legend_out=True   # Move the legend outside the plots
        )

        # --- Chart refinements ---
        
        # Set a main title for the entire figure
        g.fig.suptitle(f'Forecast Metrics Comparison for Day {day}', y=1.03, fontsize=16)
        
        # Rotate x-axis labels for better readability
        g.set_xticklabels(rotation=45, horizontalalignment='right')
        
        # Adjust subplot titles to be cleaner (e.g., "Metric = ACC" -> "ACC")
        for ax in g.axes.flat:
            title = ax.get_title()
            if 'Metric = ' in title:
                ax.set_title(title.replace('Metric = ', ''), fontsize=12)

        # Save the chart as a PNG image file
        chart_filename = f'day_{day}_metrics_comparison.png'
        
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