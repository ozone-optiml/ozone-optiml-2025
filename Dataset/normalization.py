import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ozone_dataset_small import OzoneTrainDatasetSmall
import argparse
### Code to compute mean and std for normalization of reduced datasets ###

def compute_mean_std(dataset, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = 0
    mean = None
    M2 = None  # for numerically stable variance

    for encoder_input, decoder_input, output in tqdm(loader):
        # Concatenate all inputs that should share normalization
        x = encoder_input
        # flatten everything except D
        x = x.reshape(-1, x.shape[-1])  # [N, D]

        if mean is None:
            mean = x.mean(dim=0)
            M2 = ((x - mean) ** 2).sum(dim=0)
            n = x.shape[0]
        else:
            n_new = n + x.shape[0]
            delta = x.mean(dim=0) - mean
            mean = mean + delta * x.shape[0] / n_new
            M2 = M2 + ((x - mean) ** 2).sum(dim=0)
            n = n_new

    std = torch.sqrt(M2 / (n - 1))
    return mean, std

if __name__ == "__main__":
    root_dir = "/storage/dataset/ozone/NIER_AI_v8/Dataset"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduction", type=int, default=8, help="Spatial reduction factor")
    args = parser.parse_args()
    reduction = args.reduction

    dataset = OzoneTrainDatasetSmall(
        root_dir=root_dir,
        reduction=reduction
    )

    mean, std = compute_mean_std(dataset, batch_size=8)
    torch.save({'mean': mean, 'std': std}, f'normalization_reduction_{reduction}.pt')