import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset.ozone_dataset import OzoneTrainDataset, OzoneValidationDataset, OzoneTestDataset
# from Models.ozone_informer.models.model import Informer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
from datetime import datetime, timedelta

def get_frame_time(time_str, frame_number):
    """Parses a YYYYMMDD string and adds frame hours, starting from 13:00."""
    # Parse date
    base_date = datetime.strptime(time_str, "%Y%m%d")
    
    # Start at 13:00 (1 PM)
    start_time = base_date.replace(hour=13, minute=0, second=0)
    
    # Each frame is one hour later
    offset_hours = frame_number
    
    current_time = start_time + timedelta(hours=offset_hours)
    return current_time


def visualize_reanalysis_grid_as_gif(reanalysis_data, time_str, tag=''):
    """
    Generates and saves a GIF of the reanalysis data.
    
    Args:
        reanalysis_data (np.array): A numpy array of shape (R, C, T) to be plotted.
        time_str (str): The YYYYMMDD date string for titles and filenames.
        tag (str, optional): An optional tag for the output directory.
    """
    save_path=f"gifs/reanalysis_{time_str}.gif"
    
    # Check if data is valid
    if reanalysis_data is None or reanalysis_data.size == 0:
        print(f"Error: Reanalysis data for {time_str} is empty.")
        return
        
    R, C, T = reanalysis_data.shape

    # Setup figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 7), dpi=100)
    fig.suptitle('Ozone Concentration Forecast')

    # Set fixed color range
    global_min = 0
    global_max = 0.15

    # Plot the first frame
    im_reanalysis = axes.imshow(reanalysis_data[:, :, 0], cmap="viridis",
                         vmin=global_min, vmax=global_max,
                         origin="lower", animated=True)
    axes.set_title(f"Reanalysis with Observation Inserted (time: {time_str})")

    # Use a single colorbar with shared scale
    fig.colorbar(im_reanalysis, ax=axes, orientation="vertical", fraction=0.02)

    def update(frame):
        """Function to update the plot for each frame of the animation."""
        # Update the image data
        im_reanalysis.set_array(reanalysis_data[:, :, frame])
        
        # Get the timestamp for the current frame
        time = get_frame_time(time_str, frame)

        # Update the super title and capture the artist object
        title_obj = fig.suptitle(f'Reanalysis with Observation Inserted (time: {time.strftime("%Y-%m-%d %H")})')
        
        # Save individual frames (optional)
        frame_dir = f"Reanalysis/{tag}/{time_str}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        plt.savefig(f"{frame_dir}/ozone_forecast_frame_{frame}.png", dpi=200)
        
        # --- THE FIX ---
        # blit=True requires returning a sequence (tuple) of all modified artists.
        return im_reanalysis, title_obj

    # Create and save the animation
    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=200, blit=True
    )

    try:
        ani.save(save_path, writer="pillow")
        print(f"GIF saved to {save_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("This can sometimes happen if the figure window is closed manually.")
    
    plt.close(fig) # Close the figure to free up memory

if __name__=='__main__':
    root_dir = "/storage/dataset/ozone/NIER_AI_v8/Dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    parser = argparse.ArgumentParser(description="Train the Informer model")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--e_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--d_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model Dimension")
    parser.add_argument("--n_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward layer dimension")
    parser.add_argument("--dropout", type=float, default=0.00, help="Dropout rate")
    parser.add_argument("--reduction", type=int, default=1, help="Spatial reduction factor")
    parser.add_argument("--model_name", type=str, default="Trained_Models_Small_Normalized/epoch198_loss0.00027_rmse0.01640.pth", help="Name of the model")
    parser.add_argument("--tag", type=str, default="", help="Tag for the output gif")
    args = parser.parse_args()
    args.model_name = f"../Train/checkpoints/{args.model_name}"
    
    # --- Configuration ---
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training Hyperparameters
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    FLATTENED_FEATURE_DIM = 57
    SEQ_LEN = 120   # Input sequence length
    LABEL_LEN = 48
    PRED_LEN = 72

    # --- Instantiate Model (Commented out as in original) ---
    # print("Initializing Informer model...")
    # trained_model = Informer(...)
    
    # Load validation dataset
    output_subdir = "OutputDifference"
    val_dataset = OzoneValidationDataset(root_dir, reduction=args.reduction, output_subdir=output_subdir)
    train_dataset = OzoneTrainDataset(root_dir, reduction=args.reduction, output_subdir=output_subdir)
    test_dataset = OzoneTestDataset(root_dir, reduction=args.reduction, output_subdir=output_subdir)

    # Note: The 'no_red' datasets are loaded but not used in the plotting logic
    val_no_red_dataset = OzoneValidationDataset(root_dir, reduction=1, output_subdir=output_subdir)
    train_no_red_dataset = OzoneTrainDataset(root_dir, reduction=1, output_subdir=output_subdir)
    test_no_red_dataset = OzoneTestDataset(root_dir, reduction=1, output_subdir=output_subdir)

    test_date_list = [150, 200, 220, 250]

    for index in test_date_list:
        print(f"\nProcessing index: {index}")
        # Get reduced resolution data
        encoder_input, decoder_input, true_output, paths = test_dataset[index]
        
        # Get full resolution data (loaded but not used for plot)
        # _, _, true_output_no_red, _ = test_no_red_dataset[index]

        encoder_input = encoder_input.unsqueeze(0).to(device)   # add batch dim
        decoder_input = decoder_input.unsqueeze(0).to(device)

        # Get forecast prediction from input (as in original)
        forecast_pred = encoder_input[:, :, :, -72:, 10]
        forecast_pred = forecast_pred.squeeze(0).squeeze(-1).cpu().numpy()

        # De-normalize the forecast
        normalization_path = os.path.join(os.path.dirname(__file__), f'../Dataset/normalization_reduction_{args.reduction}.pt')
        normalization_info = torch.load(normalization_path)
        mean = normalization_info['mean'][10].item()
        std = normalization_info['std'][10].item()
        forecast_pred = (forecast_pred * std) + mean
        forecast_pred = forecast_pred.transpose(1, 0, 2) # Final Shape: (R, C, T)

        print(f"Shapes - Encoder Input: {encoder_input.shape}, Decoder Input: {decoder_input.shape}, True Output (Diff): {true_output.shape}")

        # Reshape ground truth for plotting: (C, R, T) -> (R, C, T)
        true_output = true_output.permute(1, 0, 2)
        true_output = true_output.cpu().numpy()

        # true_output_no_red = true_output_no_red.permute(1, 0, 2)
        # true_output_no_red = true_output_no_red.cpu().numpy()

        encoder_path = paths['encoder_cmaq'].replace('/', '_')
        decoder_path = paths['decoder_cmaq'].replace('/', '_')
        time_str = encoder_path.split('_')[-1].split(".")[0]  # Extract YYYYMMDD from filename

        # --- Cleaned up function call ---
        # Create the reanalysis data by adding the predicted forecast
        # to the ground truth difference (as done in the original function)
        reanalysis_data_for_plot = true_output + forecast_pred

        # Call the visualization function with the data it needs
        visualize_reanalysis_grid_as_gif(
            reanalysis_data_for_plot, 
            time_str=time_str, 
            tag=args.tag
        )
