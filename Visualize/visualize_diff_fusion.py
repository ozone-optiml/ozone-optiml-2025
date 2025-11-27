import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset.ozone_dataset import OzoneTrainDataset, OzoneValidationDataset, OzoneTestDataset
from Models.ozone_informer.models.model import Informer
from Models.ozone_informer.models_rope.model import Informer_RoPE
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
from datetime import datetime, timedelta

def get_frame_time(time_str, frame_number):
    # Parse date
    base_date = datetime.strptime(time_str, "%Y%m%d")
    
    # Start at 13:00
    start_time = base_date.replace(hour=13, minute=0, second=0)
    
    offset_hours = frame_number
    
    current_time = start_time + timedelta(hours=offset_hours)
    return current_time


def visualize_prediction_grid_as_gif(pred, forecast_pred, true_output, true_output_no_red, device, time_str, tag=''):
    """
    Generates and saves a GIF comparing the ground truth and the model's prediction
    for a single data sample.
    """
    save_path=f"gifs/{tag} ozone_pred_{time_str}.gif"
    R, C, T = true_output.shape

    # --- FIX: Calculate global min/max for a consistent color scale ---
    vmin = min(np.min(true_output), np.min(pred))
    vmax = max(np.max(true_output), np.max(pred))
    print(f"Setting fixed color bar range: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # Setup figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=100)
    fig.suptitle('Ozone Concentration Forecast')

    # Compute global min/max across both true and predicted sequences
    diff_global_min = min(true_output.min(), pred.min())
    diff_global_max = max(true_output.max(), pred.max())

    # global_min = 0
    # global_max = max((true_output + forecast_pred).max(), forecast_pred.max())

    global_min = 0
    global_max = 0.15

    diff_global_min = -0.03
    diff_global_max = 0.03

    diff_pred = np.abs(pred - true_output)
    diff_forecast = np.abs(forecast_pred - true_output)

    model_pred = pred + forecast_pred
    reanalysis = true_output + forecast_pred
    # Setup figure
    im_true = axes[0][0].imshow(true_output[:, :, 0], cmap="Spectral",
                         vmin=diff_global_min, vmax=diff_global_max,
                         origin="lower", animated=True)
    im_pred = axes[0][1].imshow(pred[:, :, 0], cmap="Spectral",
                         vmin=diff_global_min, vmax=diff_global_max,
                         origin="lower", animated=True)
    im_forecast = axes[0][2].imshow(forecast_pred[:, :, 0], cmap="viridis",
                         vmin=0, vmax=global_max,
                         origin="lower", animated=True)
    im_true_no_red = axes[1][0].imshow(true_output_no_red[:, :, 0], cmap="Spectral",
                         vmin=diff_global_min, vmax=diff_global_max,
                         origin="lower", animated=True)
    im_diff_pred = axes[1][1].imshow(model_pred[:, :, 0], cmap="viridis",
                         vmin=0, vmax=global_max,
                         origin="lower", animated=True)
    im_reanalysis = axes[1][2].imshow(reanalysis[:, :, 0], cmap="viridis",
                         vmin=0, vmax=global_max,
                         origin="lower", animated=True)
    axes[0][0].set_title(f"GT Diff (RMSE: {np.sqrt(np.mean(true_output ** 2)):.4f})")
    axes[0][1].set_title(f"Our Diff (RMSE: {np.sqrt(np.mean((pred - true_output) ** 2)):.4f})")
    axes[0][2].set_title(f"Forecast")
    axes[1][0].set_title(f"GT Diff with No Red ((RMSE: {np.sqrt(np.mean(true_output_no_red ** 2)):.4f}))")
    axes[1][1].set_title("Model Pred (Our Diff + Forecast)")
    axes[1][2].set_title("Reanalysis (Forecast + GT Diff)")

    # Use a single colorbar with shared scale
    fig.colorbar(im_true, ax=axes, orientation="vertical", fraction=0.02)
    fig.colorbar(im_forecast, ax=axes, orientation="vertical", fraction=0.02)

    def update(frame):
        """Function to update the plot for each frame of the animation."""
        im_true.set_array(true_output[:, :, frame])
        im_pred.set_array(pred[:, :, frame])
        im_true_no_red.set_array(true_output_no_red[:, :, frame])
        im_forecast.set_array(forecast_pred[:, :, frame])
        im_diff_pred.set_array(model_pred[:, :, frame])
        im_reanalysis.set_array(reanalysis[:, :, frame])
        # times_str in yyyymmdd format
        # Current frame's time with hour, first frame is 1pm using time_str
        time = get_frame_time(time_str, frame)

        fig.suptitle(f'Ozone Concentration Forecast (time: {time.strftime("%Y-%m-%d %H")})')
        # Save frame
        frame_dir = f"Frames/{tag}/{time_str}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        plt.savefig(f"{frame_dir}/ozone_forecast_frame_{frame}.png", dpi=200)
        return im_true, im_pred, im_diff_pred, im_forecast, im_reanalysis

    # Create and save the animation
    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=200, blit=True
    )

    ani.save(save_path, writer="pillow")
    print(f"GIF saved to {save_path}")

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
    parser.add_argument("--reduction", type=int, default=8, help="Spatial reduction factor")
    parser.add_argument("--rope", action='store_true', help="Use RoPE in the model")
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

    # --- Instantiate Model ---
    print("Initializing Informer model...")
    model_class = Informer_RoPE if args.rope else Informer
    trained_model = model_class(
        enc_in=FLATTENED_FEATURE_DIM, 
        dec_in=FLATTENED_FEATURE_DIM, 
        c_out=FLATTENED_FEATURE_DIM,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=5,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn='flash',
        embed='fixed',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=False,
        mix=False,
        device=device
    )
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, args.model_name)
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"

    # --- PATCH: Load checkpoint while stripping 'module.' prefix if present ---
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    from collections import OrderedDict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state_dict[new_k] = v
    trained_model.load_state_dict(new_state_dict)
    # --- PATCH END ---

    if torch.cuda.device_count() > 1:
        trained_model = nn.DataParallel(trained_model)
    trained_model.to(device)

    # Load validation dataset
    output_subdir = "FusionOutputDifference"
    val_dataset = OzoneValidationDataset(root_dir, reduction=args.reduction, output_subdir=output_subdir)
    train_dataset = OzoneTrainDataset(root_dir, reduction=args.reduction, output_subdir=output_subdir)
    test_dataset = OzoneTestDataset(root_dir, reduction=args.reduction, output_subdir=output_subdir)

    val_no_red_dataset = OzoneValidationDataset(root_dir, reduction=1, output_subdir=output_subdir)
    train_no_red_dataset = OzoneTrainDataset(root_dir, reduction=1, output_subdir=output_subdir)
    test_no_red_dataset = OzoneTestDataset(root_dir, reduction=1, output_subdir=output_subdir)

    trained_model.eval()
    train_date_list = [220, 500, 800]
    validation_date_list = [180]
    test_date_list = [150, 200, 220, 250]

    # # Take one validation sample (e.g., the last one)
    # for index in train_date_list:
    #     encoder_input, decoder_input, true_output, paths = train_dataset[index]
    #     _, _, true_output_no_red, _ = train_no_red_dataset[index]
    #     # encoder_input, decoder_input, true_output, paths = val_dataset[180]
    #     # _, _, true_output_no_red, _ = val_no_red_dataset[180]

    #     encoder_input = encoder_input.unsqueeze(0).to(device)   # add batch dim
    #     decoder_input = decoder_input.unsqueeze(0).to(device)


    #     forecast_pred = encoder_input[:, :, :, -72:, 10]
    #     forecast_pred = forecast_pred.squeeze(0).squeeze(-1).cpu().numpy()

    #     normalization_path = os.path.join(os.path.dirname(__file__), f'../Dataset/normalization_reduction_{args.reduction}.pt')
    #     normalization_info = torch.load(normalization_path)
    #     mean = normalization_info['mean'][10].item()
    #     std = normalization_info['std'][10].item()
    #     forecast_pred = (forecast_pred * std) + mean
    #     forecast_pred = forecast_pred.transpose(1, 0, 2) # Final Shape: (R, C, T)

    #     print(f"Shapes - Encoder Input: {encoder_input.shape}, Decoder Input: {decoder_input.shape}, True Output: {true_output.shape}")

    #     with torch.no_grad():
    #         # --- FIX: Correctly handle model output ---
    #         # The model returns a single tensor when output_attention=False
    #         pred = trained_model(encoder_input, decoder_input)  # Shape: (B, C, R, T)
            
    #         # --- FIX: Correctly reshape prediction tensor ---
    #         # Permute to (B, R, C, T) then squeeze batch and feature dims
    #         pred = pred.permute(0, 2, 1, 3)
    #         pred = pred.squeeze(0).squeeze(-1).cpu().numpy() # Final Shape: (R, C, T)

    #     # Reshape ground truth for plotting: (C, R, T) -> (R, C, T)
    #     true_output = true_output.permute(1, 0, 2)
    #     true_output = true_output.cpu().numpy()

    #     true_output_no_red = true_output_no_red.permute(1, 0, 2)
    #     true_output_no_red = true_output_no_red.cpu().numpy()

    #     encoder_path = paths['encoder_cmaq'].replace('/', '_')
    #     decoder_path = paths['decoder_cmaq'].replace('/', '_')
    #     time_str = encoder_path.split('_')[-1].split(".")[0]  # Extract time part from filename
    #     # RMSE between forecaset and true output
    #     rmse_forecast = np.sqrt(np.mean((true_output) ** 2))
    #     rmse_model = np.sqrt(np.mean((pred - true_output) ** 2))
    #     print(f"RMSE - Forecast: {rmse_forecast:.5f}, Model: {rmse_model:.5f}")
    #     visualize_prediction_grid_as_gif(pred, forecast_pred, true_output, true_output_no_red, device, time_str=time_str, tag=args.tag)
    

    # for index in validation_date_list:
    #     # encoder_input, decoder_input, true_output, paths = val_dataset[index]
    #     # _, _, true_output_no_red, _ = val_no_red_dataset[index]
    #     encoder_input, decoder_input, true_output, paths = val_dataset[index]
    #     _, _, true_output_no_red, _ = val_no_red_dataset[index]

    #     encoder_input = encoder_input.unsqueeze(0).to(device)   # add batch dim
    #     decoder_input = decoder_input.unsqueeze(0).to(device)


    #     forecast_pred = encoder_input[:, :, :, -72:, 10]
    #     forecast_pred = forecast_pred.squeeze(0).squeeze(-1).cpu().numpy()

    #     normalization_path = os.path.join(os.path.dirname(__file__), f'../Dataset/normalization_reduction_{args.reduction}.pt')
    #     normalization_info = torch.load(normalization_path)
    #     mean = normalization_info['mean'][10].item()
    #     std = normalization_info['std'][10].item()
    #     forecast_pred = (forecast_pred * std) + mean
    #     forecast_pred = forecast_pred.transpose(1, 0, 2) # Final Shape: (R, C, T)

    #     print(f"Shapes - Encoder Input: {encoder_input.shape}, Decoder Input: {decoder_input.shape}, True Output: {true_output.shape}")

    #     with torch.no_grad():
    #         # --- FIX: Correctly handle model output ---
    #         # The model returns a single tensor when output_attention=False
    #         pred = trained_model(encoder_input, decoder_input)  # Shape: (B, C, R, T)
            
    #         # --- FIX: Correctly reshape prediction tensor ---
    #         # Permute to (B, R, C, T) then squeeze batch and feature dims
    #         pred = pred.permute(0, 2, 1, 3)
    #         pred = pred.squeeze(0).squeeze(-1).cpu().numpy() # Final Shape: (R, C, T)

    #     # Reshape ground truth for plotting: (C, R, T) -> (R, C, T)
    #     true_output = true_output.permute(1, 0, 2)
    #     true_output = true_output.cpu().numpy()

    #     true_output_no_red = true_output_no_red.permute(1, 0, 2)
    #     true_output_no_red = true_output_no_red.cpu().numpy()

    #     encoder_path = paths['encoder_cmaq'].replace('/', '_')
    #     decoder_path = paths['decoder_cmaq'].replace('/', '_')
    #     time_str = encoder_path.split('_')[-1].split(".")[0]  # Extract time part from filename
    #     # RMSE between forecaset and true output
    #     rmse_forecast = np.sqrt(np.mean((forecast_pred - true_output) ** 2))
    #     rmse_model = np.sqrt(np.mean((pred - true_output) ** 2))
    #     print(f"RMSE - Forecast: {rmse_forecast:.5f}, Model: {rmse_model:.5f}")
    #     visualize_prediction_grid_as_gif(pred, forecast_pred, true_output, true_output_no_red, device, time_str=time_str)

    for index in test_date_list:
        encoder_input, decoder_input, true_output, paths = test_dataset[index]
        _, _, true_output_no_red, _ = test_no_red_dataset[index]

        encoder_input = encoder_input.unsqueeze(0).to(device)   # add batch dim
        decoder_input = decoder_input.unsqueeze(0).to(device)


        forecast_pred = encoder_input[:, :, :, -72:, 10]
        forecast_pred = forecast_pred.squeeze(0).squeeze(-1).cpu().numpy()

        normalization_path = os.path.join(os.path.dirname(__file__), f'../Dataset/normalization_reduction_{args.reduction}.pt')
        normalization_info = torch.load(normalization_path)
        mean = normalization_info['mean'][10].item()
        std = normalization_info['std'][10].item()
        forecast_pred = (forecast_pred * std) + mean
        forecast_pred = forecast_pred.transpose(1, 0, 2) # Final Shape: (R, C, T)

        print(f"Shapes - Encoder Input: {encoder_input.shape}, Decoder Input: {decoder_input.shape}, True Output: {true_output.shape}")

        with torch.no_grad():
            # --- FIX: Correctly handle model output ---
            # The model returns a single tensor when output_attention=False
            pred = trained_model(encoder_input, decoder_input)  # Shape: (B, C, R, T)
            
            # --- FIX: Correctly reshape prediction tensor ---
            # Permute to (B, R, C, T) then squeeze batch and feature dims
            pred = pred.permute(0, 2, 1, 3)
            pred = pred.squeeze(0).squeeze(-1).cpu().numpy() # Final Shape: (R, C, T)

        # Reshape ground truth for plotting: (C, R, T) -> (R, C, T)
        true_output = true_output.permute(1, 0, 2)
        true_output = true_output.cpu().numpy()

        true_output_no_red = true_output_no_red.permute(1, 0, 2)
        true_output_no_red = true_output_no_red.cpu().numpy()

        encoder_path = paths['encoder_cmaq'].replace('/', '_')
        decoder_path = paths['decoder_cmaq'].replace('/', '_')
        time_str = encoder_path.split('_')[-1].split(".")[0]  # Extract time part from filename
        # RMSE between forecaset and true output
        rmse_forecast = np.sqrt(np.mean((forecast_pred - true_output) ** 2))
        rmse_model = np.sqrt(np.mean((pred - true_output) ** 2))
        print(f"RMSE - Forecast: {rmse_forecast:.5f}, Model: {rmse_model:.5f}")
        visualize_prediction_grid_as_gif(pred, forecast_pred, true_output, true_output_no_red, device, time_str=time_str, tag=args.tag)
