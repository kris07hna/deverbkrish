"""
Evaluation script for dereverberation with reverb-specific metrics.
Provides comprehensive evaluation including T60 estimation, DRR improvement, and room-specific analysis.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import logging

# Import evaluation metrics
from pesq import pesq
from pystoi import stoi

# Import SGMSE components
from sgmse.model import ScoreModel
from sgmse.util.other import energy_ratios, mean_std
from sgmse.sampling import get_pc_sampler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def estimate_t60_from_audio(audio: np.ndarray, sr: int) -> float:
    """
    Estimate T60 from audio signal using energy decay analysis.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Estimated T60 in seconds
    """
    
    try:
        # Calculate energy in overlapping windows
        window_size = int(0.1 * sr)  # 100ms windows
        hop_size = int(0.05 * sr)   # 50ms hop
        
        energy_frames = []
        for i in range(0, len(audio) - window_size, hop_size):
            frame = audio[i:i + window_size]
            energy = np.sum(frame ** 2)
            energy_frames.append(energy)
        
        energy_frames = np.array(energy_frames)
        
        # Smooth the energy curve
        smoothed_energy = signal.savgol_filter(energy_frames, 
                                              window_length=min(21, len(energy_frames)), 
                                              polyorder=3)
        
        # Convert to dB
        energy_db = 10 * np.log10(smoothed_energy + 1e-10)
        
        # Find peak and estimate decay
        peak_idx = np.argmax(energy_db)
        
        if peak_idx < len(energy_db) - 10:  # Need some decay data
            decay_region = energy_db[peak_idx:]
            
            # Fit exponential decay
            time_points = np.arange(len(decay_region)) * hop_size / sr
            
            # Use first 80% of decay for fitting
            fit_length = int(0.8 * len(decay_region))
            if fit_length > 10:
                x_fit = time_points[:fit_length]
                y_fit = decay_region[:fit_length]
                
                # Linear fit in dB domain (exponential decay)
                coeffs = np.polyfit(x_fit, y_fit, 1)
                slope = coeffs[0]  # dB/second
                
                if slope < 0:
                    # T60 is time for 60dB decay
                    t60 = -60 / slope
                    return np.clip(t60, 0.1, 5.0)
        
        return 1.0  # Default fallback
        
    except Exception as e:
        logger.warning(f"Error estimating T60: {e}")
        return 1.0


def calculate_drr_ratio(clean_audio: np.ndarray, reverb_audio: np.ndarray) -> float:
    """
    Calculate Direct-to-Reverberant Ratio (DRR).
    
    Args:
        clean_audio: Clean (direct) audio signal
        reverb_audio: Reverberant audio signal
        
    Returns:
        DRR in dB
    """
    
    try:
        # Ensure same length
        min_len = min(len(clean_audio), len(reverb_audio))
        clean_audio = clean_audio[:min_len]
        reverb_audio = reverb_audio[:min_len]
        
        # Calculate energy
        clean_energy = np.mean(clean_audio ** 2)
        reverb_energy = np.mean(reverb_audio ** 2)
        
        # Calculate DRR
        if reverb_energy > 0:
            drr = 10 * np.log10((clean_energy / reverb_energy) + 1e-8)
        else:
            drr = 0.0
        
        return drr
        
    except Exception as e:
        logger.warning(f"Error calculating DRR: {e}")
        return 0.0


def calculate_clarity_c50_c80(audio: np.ndarray, sr: int) -> tuple:
    """
    Calculate clarity measures C50 and C80.
    
    Args:
        audio: Audio signal (should be impulse response)
        sr: Sample rate
        
    Returns:
        Tuple of (C50, C80) in dB
    """
    
    try:
        # Find peak of impulse response
        peak_idx = np.argmax(np.abs(audio))
        
        # Align to peak
        if peak_idx > 0:
            audio = audio[peak_idx:]
        
        # Calculate energy for different time windows
        energy = audio ** 2
        
        # C50: energy ratio before/after 50ms
        ms50_samples = int(0.05 * sr)
        if len(energy) > ms50_samples:
            early_energy_50 = np.sum(energy[:ms50_samples])
            late_energy_50 = np.sum(energy[ms50_samples:])
            
            if late_energy_50 > 0:
                c50 = 10 * np.log10(early_energy_50 / late_energy_50)
            else:
                c50 = np.inf
        else:
            c50 = np.inf
        
        # C80: energy ratio before/after 80ms
        ms80_samples = int(0.08 * sr)
        if len(energy) > ms80_samples:
            early_energy_80 = np.sum(energy[:ms80_samples])
            late_energy_80 = np.sum(energy[ms80_samples:])
            
            if late_energy_80 > 0:
                c80 = 10 * np.log10(early_energy_80 / late_energy_80)
            else:
                c80 = np.inf
        else:
            c80 = np.inf
        
        return c50, c80
        
    except Exception as e:
        logger.warning(f"Error calculating clarity measures: {e}")
        return 0.0, 0.0


def evaluate_sample(clean_path: str, reverb_path: str, enhanced_path: str, 
                   metadata: dict = None) -> dict:
    """
    Evaluate a single audio sample with comprehensive metrics.
    
    Args:
        clean_path: Path to clean audio file
        reverb_path: Path to reverberant audio file
        enhanced_path: Path to enhanced audio file
        metadata: Optional metadata about the sample
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    results = {'filename': os.path.basename(enhanced_path)}
    
    try:
        # Load audio files
        clean_audio, sr_clean = sf.read(clean_path)
        reverb_audio, sr_reverb = sf.read(reverb_path)
        enhanced_audio, sr_enhanced = sf.read(enhanced_path)
        
        # Ensure single channel
        if clean_audio.ndim > 1:
            clean_audio = clean_audio[:, 0]
        if reverb_audio.ndim > 1:
            reverb_audio = reverb_audio[:, 0]
        if enhanced_audio.ndim > 1:
            enhanced_audio = enhanced_audio[:, 0]
        
        # Ensure same sample rate
        if sr_clean != 16000:
            clean_audio = torchaudio.functional.resample(
                torch.from_numpy(clean_audio), sr_clean, 16000).numpy()
        if sr_reverb != 16000:
            reverb_audio = torchaudio.functional.resample(
                torch.from_numpy(reverb_audio), sr_reverb, 16000).numpy()
        if sr_enhanced != 16000:
            enhanced_audio = torchaudio.functional.resample(
                torch.from_numpy(enhanced_audio), sr_enhanced, 16000).numpy()
        
        sr = 16000
        
        # Ensure same length
        min_len = min(len(clean_audio), len(reverb_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        reverb_audio = reverb_audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]
        
        # Standard speech enhancement metrics
        results['pesq_reverb'] = pesq(sr, clean_audio, reverb_audio, 'wb')
        results['pesq_enhanced'] = pesq(sr, clean_audio, enhanced_audio, 'wb')
        results['pesq_improvement'] = results['pesq_enhanced'] - results['pesq_reverb']
        
        results['estoi_reverb'] = stoi(clean_audio, reverb_audio, sr, extended=True)
        results['estoi_enhanced'] = stoi(clean_audio, enhanced_audio, sr, extended=True)
        results['estoi_improvement'] = results['estoi_enhanced'] - results['estoi_reverb']
        
        # SI-SDR metrics
        si_sdr_reverb = energy_ratios(reverb_audio, clean_audio, enhanced_audio)[0]
        si_sdr_enhanced = energy_ratios(enhanced_audio, clean_audio, reverb_audio)[0]
        results['si_sdr_reverb'] = si_sdr_reverb
        results['si_sdr_enhanced'] = si_sdr_enhanced
        results['si_sdr_improvement'] = si_sdr_enhanced - si_sdr_reverb
        
        # Dereverberation-specific metrics
        # T60 estimation
        t60_reverb = estimate_t60_from_audio(reverb_audio, sr)
        t60_enhanced = estimate_t60_from_audio(enhanced_audio, sr)
        results['t60_reverb'] = t60_reverb
        results['t60_enhanced'] = t60_enhanced
        results['t60_reduction'] = t60_reverb - t60_enhanced
        results['t60_reduction_percent'] = (t60_reverb - t60_enhanced) / t60_reverb * 100
        
        # DRR calculation
        drr_reverb = calculate_drr_ratio(clean_audio, reverb_audio)
        drr_enhanced = calculate_drr_ratio(clean_audio, enhanced_audio)
        results['drr_reverb'] = drr_reverb
        results['drr_enhanced'] = drr_enhanced
        results['drr_improvement'] = drr_enhanced - drr_reverb
        
        # Add metadata if available
        if metadata:
            for key, value in metadata.items():
                if key not in results:  # Don't override calculated values
                    results[f'meta_{key}'] = value
        
    except Exception as e:
        logger.error(f"Error evaluating {enhanced_path}: {e}")
        # Fill with default values
        for key in ['pesq_reverb', 'pesq_enhanced', 'pesq_improvement',
                   'estoi_reverb', 'estoi_enhanced', 'estoi_improvement',
                   'si_sdr_reverb', 'si_sdr_enhanced', 'si_sdr_improvement',
                   't60_reverb', 't60_enhanced', 't60_reduction', 't60_reduction_percent',
                   'drr_reverb', 'drr_enhanced', 'drr_improvement']:
            results[key] = 0.0
    
    return results


def create_evaluation_plots(results_df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive evaluation plots.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. PESQ Improvement
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results_df['pesq_improvement'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('PESQ Improvement')
    plt.ylabel('Frequency')
    plt.title('PESQ Improvement Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(results_df['pesq_reverb'], results_df['pesq_enhanced'], alpha=0.6)
    plt.plot([results_df['pesq_reverb'].min(), results_df['pesq_reverb'].max()],
             [results_df['pesq_reverb'].min(), results_df['pesq_reverb'].max()], 'r--')
    plt.xlabel('PESQ (Reverberant)')
    plt.ylabel('PESQ (Enhanced)')
    plt.title('PESQ: Reverberant vs Enhanced')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pesq_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. T60 Reduction
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['t60_reduction'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('T60 Reduction (seconds)')
    plt.ylabel('Frequency')
    plt.title('T60 Reduction Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(results_df['t60_reduction_percent'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('T60 Reduction (%)')
    plt.ylabel('Frequency')
    plt.title('T60 Reduction Percentage Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(results_df['t60_reverb'], results_df['t60_enhanced'], alpha=0.6)
    plt.plot([results_df['t60_reverb'].min(), results_df['t60_reverb'].max()],
             [results_df['t60_reverb'].min(), results_df['t60_reverb'].max()], 'r--')
    plt.xlabel('T60 Reverberant (seconds)')
    plt.ylabel('T60 Enhanced (seconds)')
    plt.title('T60: Reverberant vs Enhanced')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['t60_reverb'], results_df['t60_reduction'], alpha=0.6)
    plt.xlabel('T60 Reverberant (seconds)')
    plt.ylabel('T60 Reduction (seconds)')
    plt.title('T60 Reduction vs Initial T60')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 't60_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. DRR Improvement
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results_df['drr_improvement'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('DRR Improvement (dB)')
    plt.ylabel('Frequency')
    plt.title('DRR Improvement Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(results_df['drr_reverb'], results_df['drr_enhanced'], alpha=0.6)
    plt.plot([results_df['drr_reverb'].min(), results_df['drr_reverb'].max()],
             [results_df['drr_reverb'].min(), results_df['drr_reverb'].max()], 'r--')
    plt.xlabel('DRR Reverberant (dB)')
    plt.ylabel('DRR Enhanced (dB)')
    plt.title('DRR: Reverberant vs Enhanced')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drr_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation matrix
    metrics_columns = ['pesq_improvement', 'estoi_improvement', 'si_sdr_improvement',
                      't60_reduction', 'drr_improvement']
    
    available_columns = [col for col in metrics_columns if col in results_df.columns]
    
    if len(available_columns) > 1:
        correlation_matrix = results_df[available_columns].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Room type analysis if available
    room_type_col = None
    for col in results_df.columns:
        if 'room_type' in col.lower():
            room_type_col = col
            break
    
    if room_type_col and results_df[room_type_col].notna().sum() > 0:
        room_types = results_df[room_type_col].unique()
        room_types = [rt for rt in room_types if pd.notna(rt)]
        
        if len(room_types) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # PESQ by room type
            room_pesq_data = [results_df[results_df[room_type_col] == rt]['pesq_improvement'].dropna()
                             for rt in room_types]
            axes[0, 0].boxplot(room_pesq_data, labels=room_types)
            axes[0, 0].set_ylabel('PESQ Improvement')
            axes[0, 0].set_title('PESQ Improvement by Room Type')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # T60 reduction by room type
            room_t60_data = [results_df[results_df[room_type_col] == rt]['t60_reduction'].dropna()
                            for rt in room_types]
            axes[0, 1].boxplot(room_t60_data, labels=room_types)
            axes[0, 1].set_ylabel('T60 Reduction (seconds)')
            axes[0, 1].set_title('T60 Reduction by Room Type')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # DRR improvement by room type
            room_drr_data = [results_df[results_df[room_type_col] == rt]['drr_improvement'].dropna()
                            for rt in room_types]
            axes[1, 0].boxplot(room_drr_data, labels=room_types)
            axes[1, 0].set_ylabel('DRR Improvement (dB)')
            axes[1, 0].set_title('DRR Improvement by Room Type')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # SI-SDR improvement by room type
            room_sisdr_data = [results_df[results_df[room_type_col] == rt]['si_sdr_improvement'].dropna()
                              for rt in room_types]
            axes[1, 1].boxplot(room_sisdr_data, labels=room_types)
            axes[1, 1].set_ylabel('SI-SDR Improvement (dB)')
            axes[1, 1].set_title('SI-SDR Improvement by Room Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'room_type_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()


def evaluate_dereverberation(clean_dir: str, reverb_dir: str, enhanced_dir: str,
                           csv_path: str = None, output_dir: str = None) -> pd.DataFrame:
    """
    Evaluate dereverberation performance with comprehensive metrics.
    
    Args:
        clean_dir: Directory containing clean audio files
        reverb_dir: Directory containing reverberant audio files
        enhanced_dir: Directory containing enhanced audio files
        csv_path: Optional path to CSV with metadata
        output_dir: Optional directory to save results and plots
        
    Returns:
        DataFrame with evaluation results
    """
    
    # Load metadata if available
    metadata_dict = {}
    if csv_path and os.path.exists(csv_path):
        metadata_df = pd.read_csv(csv_path)
        
        # Create lookup by filename
        for _, row in metadata_df.iterrows():
            if 'clean_path' in row:
                filename = os.path.basename(row['clean_path'])
                metadata_dict[filename] = row.to_dict()
    
    # Find enhanced files
    enhanced_files = sorted(glob(os.path.join(enhanced_dir, '*.wav')))
    
    if not enhanced_files:
        raise ValueError(f"No enhanced audio files found in {enhanced_dir}")
    
    logger.info(f"Found {len(enhanced_files)} enhanced audio files")
    
    # Evaluate each file
    results = []
    
    for enhanced_path in tqdm(enhanced_files, desc="Evaluating files"):
        enhanced_filename = os.path.basename(enhanced_path)
        
        # Find corresponding clean and reverb files
        clean_path = os.path.join(clean_dir, enhanced_filename)
        reverb_path = os.path.join(reverb_dir, enhanced_filename)
        
        if not os.path.exists(clean_path):
            logger.warning(f"Clean file not found: {clean_path}")
            continue
            
        if not os.path.exists(reverb_path):
            logger.warning(f"Reverberant file not found: {reverb_path}")
            continue
        
        # Get metadata
        metadata = metadata_dict.get(enhanced_filename, {})
        
        # Evaluate sample
        sample_results = evaluate_sample(clean_path, reverb_path, enhanced_path, metadata)
        results.append(sample_results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        raise ValueError("No files were successfully evaluated")
    
    # Calculate summary statistics
    summary_stats = {}
    
    for metric in ['pesq_improvement', 'estoi_improvement', 'si_sdr_improvement',
                  't60_reduction', 't60_reduction_percent', 'drr_improvement']:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                summary_stats[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
    
    # Print summary
    logger.info("=== Evaluation Results ===")
    logger.info(f"Evaluated {len(results_df)} files")
    
    for metric, stats in summary_stats.items():
        logger.info(f"{metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_csv = os.path.join(output_dir, 'detailed_results.csv')
        results_df.to_csv(results_csv, index=False)
        
        # Save summary statistics
        summary_file = os.path.join(output_dir, 'summary_results.txt')
        with open(summary_file, 'w') as f:
            f.write("Dereverberation Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluated files: {len(results_df)}\n\n")
            
            for metric, stats in summary_stats.items():
                f.write(f"{metric}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std:  {stats['std']:.4f}\n")
                f.write(f"  Median: {stats['median']:.4f}\n")
                f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n\n")
        
        # Create plots
        create_evaluation_plots(results_df, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate dereverberation performance")
    
    parser.add_argument('--clean_dir', type=str, required=True,
                       help='Directory containing clean audio files')
    parser.add_argument('--reverb_dir', type=str, required=True,
                       help='Directory containing reverberant audio files')
    parser.add_argument('--enhanced_dir', type=str, required=True,
                       help='Directory containing enhanced audio files')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to CSV file with metadata')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results and plots')
    
    args = parser.parse_args()
    
    # Run evaluation
    results_df = evaluate_dereverberation(
        clean_dir=args.clean_dir,
        reverb_dir=args.reverb_dir,
        enhanced_dir=args.enhanced_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation completed!")
    print(f"Evaluated {len(results_df)} files")
    
    if args.output_dir:
        print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()