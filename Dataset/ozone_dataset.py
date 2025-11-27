import os
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import date, timedelta
import torch.nn.functional as F
class OzoneTrainDataset(Dataset):
    """
    Dataset for training the ozone prediction model.
    Covers data from 2019 to 2021.
    """
    def __init__(self, root_dir, reduction=8, output_subdir="Output"):
        """
        Initializes the dataset.

        Args:
            root_dir (str): The root directory of the dataset.
        """
        self.root_dir = root_dir
        self.reduction = reduction
        self.output_subdir = os.path.join(self.root_dir, output_subdir)
        self.file_paths = self._get_file_paths(2019, 2021)

    def _get_file_paths(self, start_year, end_year):
        """
        Gathers all the file paths for the given date range.

        Args:
            start_year (int): The starting year.
            end_year (int): The ending year.

        Returns:
            list: A list of dictionaries, each containing paths for a specific day.
        """
        paths = []
        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        delta = timedelta(days=1)

        while start_date <= end_date:
            date_str = start_date.strftime('%Y%m%d')
            year = start_date.strftime('%Y')
            month = start_date.strftime('%m')
            day = start_date.strftime('%d')
            
            base_path = os.path.join(self.root_dir, '{component}', '{source}', year, month, day)
            file_name = f"{date_str}.pt"

            # Check if all required files exist for the date
            encoder_cmaq_path = os.path.join(base_path.format(component='EncoderInput', source='CMAQ'), file_name)
            encoder_mcip_path = os.path.join(base_path.format(component='EncoderInput', source='MCIP'), file_name)
            decoder_cmaq_path = os.path.join(base_path.format(component='DecoderInput', source='CMAQ'), file_name)
            decoder_mcip_path = os.path.join(base_path.format(component='DecoderInput', source='MCIP'), file_name)
            output_path = os.path.join(self.output_subdir, year, month, day, f"y_{file_name}")

            if all(os.path.exists(p) for p in [encoder_cmaq_path, encoder_mcip_path, decoder_cmaq_path, decoder_mcip_path, output_path]):
                paths.append({
                    'encoder_cmaq': encoder_cmaq_path,
                    'encoder_mcip': encoder_mcip_path,
                    'decoder_cmaq': decoder_cmaq_path,
                    'decoder_mcip': decoder_mcip_path,
                    'output': output_path
                })
            start_date += delta
        return paths

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Fetches a data sample for a given index.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            tuple: A tuple containing the loaded tensors for the sample.
        """
        paths = self.file_paths[idx]
        
        # Load tensors from the .pt files
        encoder_cmaq = torch.load(paths['encoder_cmaq']) # C, R, T, D
        encoder_mcip = torch.load(paths['encoder_mcip']) # C, R, T, D
        decoder_cmaq = torch.load(paths['decoder_cmaq']) # C, R, T, D
        decoder_mcip = torch.load(paths['decoder_mcip']) # C, R, T, D

        try:
            # print("Loading:", paths['output'])
            output = torch.load(paths['output'])
        except FileNotFoundError:
            output = None
            print(f"Warning: Output file not found for index {idx} at {paths['output']}")

        encoder_input = torch.cat([encoder_cmaq, encoder_mcip], dim=-1)
        decoder_input = torch.cat([decoder_cmaq, decoder_mcip], dim=-1)

        # Reduce C, R using 4 by 4 average pooling
        reduction = self.reduction
        encoder_input = self.pool_cr(encoder_input, kernel_size=reduction, stride=reduction)
        decoder_input = self.pool_cr(decoder_input, kernel_size=reduction, stride=reduction)
        output = self.pool_cr(output, kernel_size=reduction, stride=reduction)

        # Z-normalization with precomputed mean and std
        # File is in the same directory as this script
        normalization_path = os.path.join(os.path.dirname(__file__), f'normalization_reduction_{reduction}.pt')
        assert os.path.exists(normalization_path), f"Normalization file not found at {normalization_path}"

        norm_data = torch.load(normalization_path)
        mean = norm_data['mean'] # D
        std = norm_data['std'] # D

        mean = mean.view(1, 1, 1, -1)  # 1, 1, 1, D
        std = std.view(1, 1, 1, -1)  # 1, 1, 1, D

        encoder_input = (encoder_input - mean) / std
        decoder_input = (decoder_input - mean) / std
        return encoder_input, decoder_input, output, paths

    def pool_cr(self, x, kernel_size=4, stride=4):
        if x.ndim == 3:
            # x: [C, R, T]
            C, R, T = x.shape
            # Rearrange so pooling applies to C and R
            x = x.permute(2, 0, 1)  # [T, C, R]
            x = x.reshape(T, 1, C, R)  # [T, 1, C, R]
            x_pooled = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)  # [T, 1, C//reduction, R//reduction]
            x_pooled = x_pooled.permute(2, 3, 0, 1)  # [C//reduction, R//reduction, T, 1]
            x_pooled = x_pooled.squeeze(-1)  # [C//reduction, R//reduction, T]
        else:
            # x: [C, R, T, D]
            C, R, T, D = x.shape
            # Rearrange so pooling applies to C and R
            x = x.permute(2, 3, 0, 1)  # [T, D, C, R]
            x = x.reshape(T * D, 1, C, R)  # [T*D, 1, C, R]
            
            # Apply 2D average pooling
            x_pooled = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)  # [T*D, 1, C//reduction, R//reduction]

            # Restore shape
            C_new, R_new = x_pooled.shape[-2:]
            x_pooled = x_pooled.view(T, D, C_new, R_new).permute(2, 3, 0, 1)  # [C_new, R_new, T, D]
        return x_pooled


class OzoneValidationDataset(OzoneTrainDataset):
    """
    Dataset for validating the ozone prediction model.
    Covers data for the year 2022.
    """
    def __init__(self, root_dir, reduction, output_subdir="Output"):
        """
        Initializes the dataset.

        Args:
            root_dir (str): The root directory of the dataset.
        """
        self.root_dir = root_dir
        self.reduction = reduction
        self.output_subdir = os.path.join(self.root_dir, output_subdir)
        self.file_paths = self._get_file_paths(2022, 2022)

class OzoneTestDataset(OzoneTrainDataset):
    """
    Dataset for testing the ozone prediction model.
    Covers data for the year 2023.
    """
    def __init__(self, root_dir, reduction, output_subdir="Output"):
        """
        Initializes the dataset.

        Args:
            root_dir (str): The root directory of the dataset.
        """
        self.root_dir = root_dir
        self.reduction = reduction
        self.output_subdir = os.path.join(self.root_dir, output_subdir)
        self.file_paths = self._get_file_paths(2023, 2023)
