import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

from Dataset.ozone_dataset import OzoneTrainDataset, OzoneValidationDataset, OzoneTestDataset
from Models.ozone_informer.models.model import Informer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import json
from datetime import datetime, timedelta
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from collections import OrderedDict
import validate_utils as utils

if __name__=='__main__':
    root_dir = "/storage/dataset/ozone/NIER_AI_v8/Dataset"

    parser = argparse.ArgumentParser(description="Train the Informer model")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--e_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--d_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model Dimension")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward layer dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--reduction", type=int, default=4, help="Spatial reduction factor")
    parser.add_argument("--model_dir", type=str, default="../Train/checkpoints/6.3. lr_0.0001_normalized_d_ff2048_d_model512_reduction4_heads16_elayers6_dlayers6_focalTrue (new embed) gamma=3 beta=1e4", help="Parent directory of the model")
    parser.add_argument("--model_name", type=str, default="best_model.pth", help="Name of the model")
    parser.add_argument("--split", type=str, default=None, help="val, test, or None for all")
    parser.add_argument("--results_dir", type=str, default='results', help="results path")

    args = parser.parse_args()
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
    trained_model = Informer(
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
    
    experiment_name = "forecast"

    os.makedirs(args.results_dir, exist_ok=True)

    save_dir = os.path.join(args.results_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Save directory: {save_dir}")

    normalization_path = os.path.join(os.path.dirname(__file__), f'../../Dataset/normalization_reduction_{args.reduction}.pt')
    normalization_info = torch.load(normalization_path)
    mean = normalization_info['mean'][10].item()
    std = normalization_info['std'][10].item()

    splits = {
        'val': OzoneValidationDataset(root_dir, reduction=args.reduction),
        'test': OzoneTestDataset(root_dir, reduction=args.reduction)
    }

    def evaluate(loader, split_name, months=[1, 7]):
        results_dict = {}
        for encoder_input, decoder_input, true_output, paths in tqdm(loader, desc=f"{split_name}"):

            # --- check month before inference ---
            keep_idx = []
            for i, p in enumerate(paths["output"]):
                date_str = os.path.basename(p).split("_")[1].split(".")[0]
                month = int(date_str[4:6])
                if month in months:
                    keep_idx.append(i)
            if len(keep_idx) == 0:
                continue

            encoder_input = encoder_input[keep_idx]

            forecast_pred = encoder_input[:, :, :, -72:, 10].cpu().numpy()
            forecast_pred = (forecast_pred * std) + mean
            forecast_pred = forecast_pred.transpose(0, 2, 1, 3)

            true_output = true_output[keep_idx]
            true_output = true_output.squeeze(-1).permute(0, 2, 1, 3).cpu().numpy()

            pred = forecast_pred

            for i, idx in enumerate(keep_idx):
                date_str = os.path.basename(paths["output"][idx]).split("_")[1].split(".")[0]
                p = pred[i]
                t = true_output[i]

                hourly_metrics = {m: {} for m in ["RMSE", "MAE", "IOA", "R", "NMB", "BIAS"]}
                for h in range(p.shape[-1]):
                    ph = p[..., h]
                    th = t[..., h]
                    hourly_metrics["RMSE"][h] = utils.RMSE(ph, th)
                    hourly_metrics["MAE"][h] = utils.MAE(ph, th)
                    hourly_metrics["IOA"][h] = utils.IOA(ph, th)
                    hourly_metrics["R"][h] = utils.R(ph, th)
                    hourly_metrics["NMB"][h] = utils.NMB(ph, th)
                    hourly_metrics["BIAS"][h] = utils.BIAS(ph, th)
                results_dict[date_str] = hourly_metrics

        for month in months:
            filtered = {d: v for d, v in results_dict.items() if int(d[4:6]) == month}
            if not filtered:
                continue
            month_name = f"{month:02d}"
            out_path = os.path.join(save_dir, f"{split_name}_{month_name}_by_hour.json")
            with open(out_path, "w") as f:
                json.dump(filtered, f, indent=2)
            print(f"Saved results to {out_path}")


    selected_splits = [args.split] if args.split in ["val", "test"] else ["val", "test"]

    for split_name in selected_splits:
        full_dataset = splits[split_name]
        print(f"\n--- Filtering {split_name} split for Jan and Jul ---")

        filtered_indices = []
        for idx, paths in enumerate(full_dataset.file_paths):
            output_path = paths["output"]
            date_str = os.path.basename(output_path).split("_")[1].split(".")[0]
            month = int(date_str[4:6])
            if month in [1, 7]:
                filtered_indices.append(idx)

        if len(filtered_indices) == 0:
            raise RuntimeError(f"No Jan/Jul samples found in {split_name} dataset")

        filtered_dataset = Subset(full_dataset, filtered_indices)
        loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        print(f"Filtered {split_name} dataset size: {len(filtered_dataset)}")
        evaluate(loader, split_name)
        print(f"Completed {split_name} split")

    print("\n--- Completed Evaluation ---")