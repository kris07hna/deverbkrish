"""
Dataset utilities for dereverberation tasks.
Includes functions for creating, validating, and analyzing CSV-based datasets.
"""

import os
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_dataset_csv(audio_dir: str, output_csv: str, 
                      clean_subdir: str = 'anechoic',
                      reverb_subdir: str = 'reverb',
                      rir_subdir: Optional[str] = None,
                      extract_metadata: bool = True,
                      file_extensions: List[str] = ['.wav'],
                      subset_column: bool = True) -> pd.DataFrame:
    """
    Create a CSV dataset from directory structure.
    
    Args:
        audio_dir: Root directory containing audio files
        output_csv: Path to output CSV file
        clean_subdir: Subdirectory name for clean audio files
        reverb_subdir: Subdirectory name for reverberant audio files  
        rir_subdir: Subdirectory name for RIR files (optional)
        extract_metadata: Whether to extract T60/DRR from filenames
        file_extensions: List of audio file extensions to include
        subset_column: Whether to include subset column based on directory structure
        
    Returns:
        DataFrame with dataset information
    """
    
    data_records = []
    
    # Find all subsets (train, valid, test) or use root directory
    if subset_column:
        subset_dirs = [d for d in os.listdir(audio_dir) 
                      if os.path.isdir(os.path.join(audio_dir, d)) 
                      and d in ['train', 'valid', 'test', 'validation']]
        if not subset_dirs:
            subset_dirs = ['']  # Use root directory
    else:
        subset_dirs = ['']
    
    for subset in subset_dirs:
        subset_path = os.path.join(audio_dir, subset) if subset else audio_dir
        
        # Find clean files
        clean_path = os.path.join(subset_path, clean_subdir)
        if not os.path.exists(clean_path):
            logger.warning(f"Clean directory not found: {clean_path}")
            continue
            
        clean_files = []
        for ext in file_extensions:
            clean_files.extend(glob(os.path.join(clean_path, f"**/*{ext}"), recursive=True))
        
        clean_files = sorted(clean_files)
        logger.info(f"Found {len(clean_files)} clean files in {subset or 'root'}")
        
        for clean_file in tqdm(clean_files, desc=f"Processing {subset or 'root'}"):
            record = {'clean_path': clean_file}
            
            # Add subset information
            if subset_column and subset:
                record['subset'] = subset
            
            # Find corresponding reverb file
            clean_basename = os.path.basename(clean_file)
            reverb_path = os.path.join(subset_path, reverb_subdir)
            reverb_file = os.path.join(reverb_path, clean_basename)
            
            # Try to find reverb file with same name in reverb directory
            if os.path.exists(reverb_file):
                record['reverb_path'] = reverb_file
            else:
                # Try recursive search
                reverb_matches = glob(os.path.join(reverb_path, f"**/{clean_basename}"), 
                                    recursive=True)
                if reverb_matches:
                    record['reverb_path'] = reverb_matches[0]
                else:
                    logger.warning(f"No matching reverb file for {clean_file}")
                    continue
            
            # Find corresponding RIR file if specified
            if rir_subdir:
                rir_path = os.path.join(subset_path, rir_subdir)
                rir_file = os.path.join(rir_path, clean_basename)
                if os.path.exists(rir_file):
                    record['rir_path'] = rir_file
                else:
                    rir_matches = glob(os.path.join(rir_path, f"**/{clean_basename}"), 
                                     recursive=True)
                    if rir_matches:
                        record['rir_path'] = rir_matches[0]
            
            # Extract metadata from filename if requested
            if extract_metadata:
                metadata = extract_metadata_from_filename(clean_basename)
                record.update(metadata)
            
            # Extract audio metadata
            try:
                clean_info = sf.info(clean_file)
                reverb_info = sf.info(record['reverb_path'])
                
                record['duration'] = clean_info.duration
                record['sample_rate'] = clean_info.samplerate
                record['channels'] = clean_info.channels
                
                # Check if files have matching properties
                if (clean_info.duration != reverb_info.duration or 
                    clean_info.samplerate != reverb_info.samplerate):
                    logger.warning(f"Mismatched audio properties for {clean_basename}")
                    
            except Exception as e:
                logger.warning(f"Error reading audio metadata for {clean_basename}: {e}")
            
            data_records.append(record)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_records)
    
    if len(df) == 0:
        logger.error("No valid audio pairs found!")
        return df
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Created dataset CSV with {len(df)} samples: {output_csv}")
    
    return df


def extract_metadata_from_filename(filename: str) -> Dict[str, Union[str, float]]:
    """
    Extract metadata from filename using common naming conventions.
    
    Args:
        filename: Audio filename (without path)
        
    Returns:
        Dictionary with extracted metadata
    """
    
    metadata = {}
    
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # Common patterns for T60 and DRR in filenames
    # Example patterns: "speech_room1_0.5_-2.1.wav" (T60=0.5, DRR=-2.1)
    parts = name.split('_')
    
    for i, part in enumerate(parts):
        try:
            # Look for T60 values (typically 0.1 to 2.0 seconds)
            if part.replace('.', '').isdigit():
                val = float(part)
                if 0.1 <= val <= 2.0:
                    metadata['t60'] = val
                # Look for DRR values (typically -20 to 20 dB)
                elif -20 <= val <= 20 and '.' in part:
                    metadata['drr'] = val
        except ValueError:
            continue
    
    # Look for room information
    room_keywords = ['office', 'hall', 'room', 'chamber', 'studio', 'classroom', 
                    'auditorium', 'kitchen', 'bathroom', 'living']
    
    for part in parts:
        part_lower = part.lower()
        for keyword in room_keywords:
            if keyword in part_lower:
                metadata['room_type'] = keyword
                break
    
    # Look for speaker/microphone distance
    for part in parts:
        if part.startswith('d') and part[1:].replace('.', '').isdigit():
            try:
                metadata['distance'] = float(part[1:])
            except ValueError:
                pass
    
    return metadata


def validate_dataset_csv(csv_path: str, check_files: bool = True) -> Dict[str, any]:
    """
    Validate a dataset CSV file.
    
    Args:
        csv_path: Path to CSV file
        check_files: Whether to check if audio files exist
        
    Returns:
        Dictionary with validation results
    """
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Failed to read CSV: {e}")
        return results
    
    # Check required columns
    required_columns = ['clean_path', 'reverb_path']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for empty values
    for col in required_columns:
        if col in df.columns:
            empty_count = df[col].isna().sum()
            if empty_count > 0:
                results['warnings'].append(f"Column '{col}' has {empty_count} empty values")
    
    # Check file existence if requested
    if check_files and results['valid']:
        missing_files = []
        
        for idx, row in df.iterrows():
            for col in ['clean_path', 'reverb_path', 'rir_path']:
                if col in df.columns and pd.notna(row[col]):
                    if not os.path.exists(row[col]):
                        missing_files.append(f"Row {idx}: {col} - {row[col]}")
        
        if missing_files:
            results['warnings'].append(f"Found {len(missing_files)} missing files")
            if len(missing_files) <= 10:
                results['warnings'].extend(missing_files)
            else:
                results['warnings'].extend(missing_files[:10])
                results['warnings'].append(f"... and {len(missing_files) - 10} more")
    
    # Calculate statistics
    if results['valid']:
        stats = {
            'total_samples': len(df),
            'columns': list(df.columns)
        }
        
        # Subset distribution
        if 'subset' in df.columns:
            stats['subset_distribution'] = df['subset'].value_counts().to_dict()
        
        # Metadata statistics
        if 't60' in df.columns:
            t60_values = df['t60'].dropna()
            if len(t60_values) > 0:
                stats['t60_stats'] = {
                    'count': len(t60_values),
                    'mean': t60_values.mean(),
                    'std': t60_values.std(),
                    'min': t60_values.min(),
                    'max': t60_values.max()
                }
        
        if 'drr' in df.columns:
            drr_values = df['drr'].dropna()
            if len(drr_values) > 0:
                stats['drr_stats'] = {
                    'count': len(drr_values),
                    'mean': drr_values.mean(),
                    'std': drr_values.std(),
                    'min': drr_values.min(),
                    'max': drr_values.max()
                }
        
        if 'room_type' in df.columns:
            stats['room_type_distribution'] = df['room_type'].value_counts().to_dict()
        
        results['statistics'] = stats
    
    return results


def analyze_dataset_statistics(csv_path: str, output_dir: Optional[str] = None) -> Dict[str, any]:
    """
    Analyze and visualize dataset statistics.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Directory to save plots (optional)
        
    Returns:
        Dictionary with analysis results
    """
    
    df = pd.read_csv(csv_path)
    
    analysis = {
        'basic_stats': {},
        'distributions': {},
        'correlations': {}
    }
    
    # Basic statistics
    analysis['basic_stats'] = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    if 'subset' in df.columns:
        analysis['basic_stats']['subset_distribution'] = df['subset'].value_counts().to_dict()
    
    # Set up plotting if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
    
    # Analyze T60 distribution
    if 't60' in df.columns:
        t60_values = df['t60'].dropna()
        if len(t60_values) > 0:
            analysis['distributions']['t60'] = {
                'count': len(t60_values),
                'mean': t60_values.mean(),
                'std': t60_values.std(),
                'min': t60_values.min(),
                'max': t60_values.max(),
                'percentiles': t60_values.quantile([0.25, 0.5, 0.75]).to_dict()
            }
            
            if output_dir:
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.hist(t60_values, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('T60 (seconds)')
                plt.ylabel('Frequency')
                plt.title('T60 Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.boxplot(t60_values)
                plt.ylabel('T60 (seconds)')
                plt.title('T60 Box Plot')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 't60_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    # Analyze DRR distribution
    if 'drr' in df.columns:
        drr_values = df['drr'].dropna()
        if len(drr_values) > 0:
            analysis['distributions']['drr'] = {
                'count': len(drr_values),
                'mean': drr_values.mean(),
                'std': drr_values.std(),
                'min': drr_values.min(),
                'max': drr_values.max(),
                'percentiles': drr_values.quantile([0.25, 0.5, 0.75]).to_dict()
            }
            
            if output_dir:
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.hist(drr_values, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('DRR (dB)')
                plt.ylabel('Frequency')
                plt.title('DRR Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.boxplot(drr_values)
                plt.ylabel('DRR (dB)')
                plt.title('DRR Box Plot')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'drr_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    # Analyze room type distribution
    if 'room_type' in df.columns:
        room_counts = df['room_type'].value_counts()
        analysis['distributions']['room_type'] = room_counts.to_dict()
        
        if output_dir and len(room_counts) > 0:
            plt.figure(figsize=(10, 6))
            room_counts.plot(kind='bar')
            plt.xlabel('Room Type')
            plt.ylabel('Count')
            plt.title('Room Type Distribution')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'room_type_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Analyze correlations between numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        analysis['correlations'] = correlation_matrix.to_dict()
        
        if output_dir:
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # T60 vs DRR scatter plot
    if 't60' in df.columns and 'drr' in df.columns and output_dir:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['t60'], df['drr'], alpha=0.6)
        plt.xlabel('T60 (seconds)')
        plt.ylabel('DRR (dB)')
        plt.title('T60 vs DRR')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 't60_vs_drr.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return analysis


def split_dataset_csv(csv_path: str, output_dir: str, 
                     train_ratio: float = 0.7, 
                     valid_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     stratify_column: Optional[str] = None,
                     random_seed: int = 42) -> Tuple[str, str, str]:
    """
    Split a dataset CSV into train/validation/test sets.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save split CSV files
        train_ratio: Ratio for training set
        valid_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        stratify_column: Column to use for stratified splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of paths to (train_csv, valid_csv, test_csv)
    """
    
    df = pd.read_csv(csv_path)
    
    # Validate ratios
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    if stratify_column and stratify_column in df.columns:
        # Stratified split
        train_dfs, valid_dfs, test_dfs = [], [], []
        
        for group_value in df[stratify_column].unique():
            group_df = df[df[stratify_column] == group_value]
            n = len(group_df)
            
            train_end = int(n * train_ratio)
            valid_end = train_end + int(n * valid_ratio)
            
            train_dfs.append(group_df.iloc[:train_end])
            valid_dfs.append(group_df.iloc[train_end:valid_end])
            test_dfs.append(group_df.iloc[valid_end:])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        valid_df = pd.concat(valid_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
    
    else:
        # Simple split
        n = len(df)
        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)
        
        train_df = df.iloc[:train_end]
        valid_df = df.iloc[train_end:valid_end]
        test_df = df.iloc[valid_end:]
    
    # Add subset column
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    
    train_df['subset'] = 'train'
    valid_df['subset'] = 'valid'
    test_df['subset'] = 'test'
    
    # Save split files
    train_csv = os.path.join(output_dir, 'train.csv')
    valid_csv = os.path.join(output_dir, 'valid.csv')
    test_csv = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    logger.info(f"Dataset split completed:")
    logger.info(f"  Training: {len(train_df)} samples -> {train_csv}")
    logger.info(f"  Validation: {len(valid_df)} samples -> {valid_csv}")
    logger.info(f"  Test: {len(test_df)} samples -> {test_csv}")
    
    return train_csv, valid_csv, test_csv


def filter_dataset_csv(csv_path: str, output_path: str, **filter_conditions) -> pd.DataFrame:
    """
    Filter a dataset CSV based on conditions.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output filtered CSV file
        **filter_conditions: Filtering conditions
        
    Returns:
        Filtered DataFrame
    """
    
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    for column, condition in filter_conditions.items():
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in dataset")
            continue
        
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
    
    # Save filtered dataset
    df.to_csv(output_path, index=False)
    
    logger.info(f"Filtered dataset: {original_count} -> {len(df)} samples")
    logger.info(f"Saved to: {output_path}")
    
    return df