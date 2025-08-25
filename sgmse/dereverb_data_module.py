"""
Enhanced data module for dereverberation tasks with CSV-based dataset management.
Extends the base SpecsDataModule to support reverb-specific functionality.
"""

import os
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
from os.path import join, basename, dirname, exists
import logging

from .data_module import SpecsDataModule, get_window

logger = logging.getLogger(__name__)


class ReverbSpecs(Dataset):
    """Dataset class for dereverberation with CSV-based metadata support."""
    
    def __init__(self, data_source, subset, dummy, shuffle_spec, num_frames,
                 format='csv', normalize="noisy", spec_transform=None,
                 stft_kwargs=None, csv_columns=None, filter_conditions=None,
                 **ignored_kwargs):
        """
        Initialize ReverbSpecs dataset.
        
        Args:
            data_source: Path to CSV file or directory containing audio files
            subset: Dataset subset ('train', 'valid', 'test')
            dummy: Use reduced dataset for debugging
            shuffle_spec: Shuffle spectrograms during training
            num_frames: Number of frames for the dataset
            format: Dataset format ('csv', 'directory', 'reverb')
            normalize: Normalization strategy ('noisy', 'clean', 'not')
            spec_transform: Spectrogram transformation function
            stft_kwargs: STFT parameters
            csv_columns: Expected CSV columns (default: ['clean_path', 'reverb_path'])
            filter_conditions: Dictionary of filtering conditions for metadata
        """
        
        self.format = format
        self.subset = subset
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        
        # Default CSV columns
        if csv_columns is None:
            csv_columns = ['clean_path', 'reverb_path']
        self.csv_columns = csv_columns
        
        # Load file paths and metadata
        if format == 'csv':
            self._load_from_csv(data_source, subset, filter_conditions)
        elif format == 'directory':
            self._load_from_directory(data_source, subset)
        elif format == 'reverb':
            self._load_reverb_format(data_source, subset)
        else:
            raise NotImplementedError(f"Dataset format {format} unknown!")
        
        # Validate STFT kwargs
        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), \
            "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, \
            "'center' must be True for current implementation"
    
    def _load_from_csv(self, csv_path, subset, filter_conditions):
        """Load dataset from CSV file with metadata."""
        
        if not exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Filter by subset if subset column exists
        if 'subset' in df.columns:
            df = df[df['subset'] == subset]
        
        # Apply filter conditions
        if filter_conditions:
            for column, condition in filter_conditions.items():
                if column in df.columns:
                    if isinstance(condition, dict):
                        # Range filtering
                        if 'min' in condition:
                            df = df[df[column] >= condition['min']]
                        if 'max' in condition:
                            df = df[df[column] <= condition['max']]
                    elif isinstance(condition, (list, tuple)):
                        # Include specific values
                        df = df[df[column].isin(condition)]
                    else:
                        # Exact match
                        df = df[df[column] == condition]
        
        # Extract file paths
        self.clean_files = df[self.csv_columns[0]].tolist()
        self.noisy_files = df[self.csv_columns[1]].tolist()
        
        # Store metadata
        self.metadata = df.to_dict('records')
        
        # Validate file paths
        self._validate_file_paths()
        
        logger.info(f"Loaded {len(self.clean_files)} samples from CSV for {subset} subset")
    
    def _load_from_directory(self, data_dir, subset):
        """Load dataset from directory structure."""
        
        self.clean_files = []
        self.clean_files += sorted(glob(join(data_dir, subset, "clean", "*.wav")))
        self.clean_files += sorted(glob(join(data_dir, subset, "clean", "**", "*.wav")))
        
        self.noisy_files = []
        self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
        self.noisy_files += sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav")))
        
        self.metadata = [{}] * len(self.clean_files)  # Empty metadata
    
    def _load_reverb_format(self, data_dir, subset):
        """Load dataset from reverb format (anechoic/reverb)."""
        
        self.clean_files = []
        self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "*.wav")))
        self.clean_files += sorted(glob(join(data_dir, subset, "anechoic", "**", "*.wav")))
        
        self.noisy_files = []
        self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "*.wav")))
        self.noisy_files += sorted(glob(join(data_dir, subset, "reverb", "**", "*.wav")))
        
        self.metadata = [{}] * len(self.clean_files)  # Empty metadata
    
    def _validate_file_paths(self):
        """Validate that all audio files exist."""
        
        missing_files = []
        
        for i, (clean_path, noisy_path) in enumerate(zip(self.clean_files, self.noisy_files)):
            if not exists(clean_path):
                missing_files.append(f"Clean file missing: {clean_path}")
            if not exists(noisy_path):
                missing_files.append(f"Noisy file missing: {noisy_path}")
        
        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing files")
            for missing in missing_files[:10]:  # Show first 10
                logger.warning(missing)
            if len(missing_files) > 10:
                logger.warning(f"... and {len(missing_files) - 10} more")
    
    def get_metadata(self, index):
        """Get metadata for a specific sample."""
        if index < len(self.metadata):
            return self.metadata[index]
        return {}
    
    def __getitem__(self, i):
        """Get a sample from the dataset."""
        
        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])
        
        # Handle tensor dimensions
        if x.dim() > 1:
            x = x[0]  # Take first channel if multi-channel
        if y.dim() > 1:
            y = y[0]  # Take first channel if multi-channel
        
        # Formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        
        if pad == 0:
            # Extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            x = x[..., start:start + target_len]
            y = y[..., start:start + target_len]
        else:
            # Pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
            y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
        
        # Normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        
        # Avoid division by zero
        if normfac == 0:
            normfac = 1.0
            
        x = x / normfac
        y = y / normfac
        
        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        
        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y
    
    def __len__(self):
        if self.dummy:
            # For debugging shrink the data set size
            return min(int(len(self.clean_files) / 200), 10)
        else:
            return len(self.clean_files)


class ReverbDataModule(SpecsDataModule):
    """Enhanced data module for dereverberation tasks."""
    
    @staticmethod
    def add_argparse_args(parser):
        """Add arguments for ReverbDataModule."""
        
        # Add base arguments
        SpecsDataModule.add_argparse_args(parser)
        
        # Add reverb-specific arguments
        parser.add_argument("--csv_path", type=str, default=None, 
                          help="Path to CSV file with dataset metadata")
        parser.add_argument("--csv_columns", type=str, nargs='+', 
                          default=['clean_path', 'reverb_path'],
                          help="CSV column names for clean and reverb audio paths")
        parser.add_argument("--filter_t60_min", type=float, default=None,
                          help="Minimum T60 value for filtering")
        parser.add_argument("--filter_t60_max", type=float, default=None,
                          help="Maximum T60 value for filtering")
        parser.add_argument("--filter_drr_min", type=float, default=None,
                          help="Minimum DRR value for filtering")
        parser.add_argument("--filter_drr_max", type=float, default=None,
                          help="Maximum DRR value for filtering")
        parser.add_argument("--filter_room_type", type=str, nargs='+', default=None,
                          help="Room types to include (e.g., 'office', 'hall')")
        
        return parser
    
    def __init__(self, base_dir=None, csv_path=None, csv_columns=None,
                 filter_t60_min=None, filter_t60_max=None,
                 filter_drr_min=None, filter_drr_max=None,
                 filter_room_type=None, format='csv', **kwargs):
        """
        Initialize ReverbDataModule.
        
        Args:
            base_dir: Base directory for audio files (used as fallback)
            csv_path: Path to CSV file with dataset metadata
            csv_columns: CSV column names for clean and reverb paths
            filter_*: Filtering parameters for T60, DRR, room type
            format: Dataset format ('csv', 'directory', 'reverb')
            **kwargs: Additional arguments passed to SpecsDataModule
        """
        
        # Determine data source
        if csv_path:
            self.data_source = csv_path
            self.dataset_format = 'csv'
        elif base_dir:
            self.data_source = base_dir
            self.dataset_format = format
        else:
            raise ValueError("Either csv_path or base_dir must be provided")
        
        self.csv_columns = csv_columns or ['clean_path', 'reverb_path']
        
        # Set up filter conditions
        self.filter_conditions = {}
        if filter_t60_min is not None or filter_t60_max is not None:
            self.filter_conditions['t60'] = {}
            if filter_t60_min is not None:
                self.filter_conditions['t60']['min'] = filter_t60_min
            if filter_t60_max is not None:
                self.filter_conditions['t60']['max'] = filter_t60_max
        
        if filter_drr_min is not None or filter_drr_max is not None:
            self.filter_conditions['drr'] = {}
            if filter_drr_min is not None:
                self.filter_conditions['drr']['min'] = filter_drr_min
            if filter_drr_max is not None:
                self.filter_conditions['drr']['max'] = filter_drr_max
        
        if filter_room_type:
            self.filter_conditions['room_type'] = filter_room_type
        
        # Initialize parent class
        super().__init__(base_dir=base_dir or '', format=self.dataset_format, **kwargs)
    
    def setup(self, stage=None):
        """Setup datasets for different stages."""
        
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, format=self.dataset_format,
            csv_columns=self.csv_columns, filter_conditions=self.filter_conditions
        )
        
        if stage == 'fit' or stage is None:
            self.train_set = ReverbSpecs(
                data_source=self.data_source, subset='train',
                dummy=self.dummy, shuffle_spec=True, 
                normalize=self.normalize, **specs_kwargs
            )
            self.valid_set = ReverbSpecs(
                data_source=self.data_source, subset='valid',
                dummy=self.dummy, shuffle_spec=False,
                normalize=self.normalize, **specs_kwargs
            )
        
        if stage == 'test' or stage is None:
            self.test_set = ReverbSpecs(
                data_source=self.data_source, subset='test',
                dummy=self.dummy, shuffle_spec=False,
                normalize=self.normalize, **specs_kwargs
            )
    
    def get_dataset_statistics(self, subset='train'):
        """Get statistics about the dataset."""
        
        if subset == 'train' and hasattr(self, 'train_set'):
            dataset = self.train_set
        elif subset == 'valid' and hasattr(self, 'valid_set'):
            dataset = self.valid_set
        elif subset == 'test' and hasattr(self, 'test_set'):
            dataset = self.test_set
        else:
            return {}
        
        stats = {
            'num_samples': len(dataset),
            'format': dataset.format
        }
        
        # Collect metadata statistics if available
        if hasattr(dataset, 'metadata') and dataset.metadata:
            metadata_df = pd.DataFrame(dataset.metadata)
            
            # T60 statistics
            if 't60' in metadata_df.columns:
                stats['t60_mean'] = metadata_df['t60'].mean()
                stats['t60_std'] = metadata_df['t60'].std()
                stats['t60_min'] = metadata_df['t60'].min()
                stats['t60_max'] = metadata_df['t60'].max()
            
            # DRR statistics  
            if 'drr' in metadata_df.columns:
                stats['drr_mean'] = metadata_df['drr'].mean()
                stats['drr_std'] = metadata_df['drr'].std()
                stats['drr_min'] = metadata_df['drr'].min()
                stats['drr_max'] = metadata_df['drr'].max()
            
            # Room type distribution
            if 'room_type' in metadata_df.columns:
                stats['room_types'] = metadata_df['room_type'].value_counts().to_dict()
        
        return stats