"""
Dataset analysis tools for reverberant speech datasets.
Provides comprehensive analysis and visualization of T60, DRR, and room characteristics.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import soundfile as sf
from scipy import signal
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_audio_file(audio_path: str) -> Dict[str, float]:
    """
    Analyze an audio file to extract acoustic properties.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio analysis results
    """
    
    try:
        audio, sr = sf.read(audio_path)
        
        # Ensure single channel
        if audio.ndim > 1:
            audio = audio[:, 0]
        
        # Basic properties
        duration = len(audio) / sr
        rms_energy = np.sqrt(np.mean(audio ** 2))
        peak_amplitude = np.max(np.abs(audio))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        zcr = zero_crossings / len(audio)
        
        # Spectral properties
        f, psd = signal.welch(audio, sr, nperseg=1024)
        spectral_centroid = np.sum(f * psd) / np.sum(psd)
        spectral_bandwidth = np.sqrt(np.sum(((f - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        return {
            'duration': duration,
            'rms_energy': rms_energy,
            'peak_amplitude': peak_amplitude,
            'dynamic_range': dynamic_range,
            'zero_crossing_rate': zcr,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth
        }
        
    except Exception as e:
        logger.warning(f"Error analyzing {audio_path}: {e}")
        return {}


def estimate_t60_schroeder(audio: np.ndarray, sr: int, freq_bands: List[Tuple] = None) -> Dict[str, float]:
    """
    Estimate T60 using Schroeder's backward integration method.
    
    Args:
        audio: Audio signal (should be room impulse response)
        sr: Sample rate
        freq_bands: List of frequency bands for analysis
        
    Returns:
        Dictionary with T60 estimates
    """
    
    if freq_bands is None:
        freq_bands = [(125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    
    results = {}
    
    try:
        # Overall T60 estimation
        # Square the signal to get energy
        energy = audio ** 2
        
        # Backward integration (Schroeder integration)
        integrated_energy = np.cumsum(energy[::-1])[::-1]
        
        # Convert to dB
        integrated_db = 10 * np.log10(integrated_energy + 1e-10)
        
        # Normalize so peak is 0 dB
        integrated_db = integrated_db - np.max(integrated_db)
        
        # Find -5dB and -35dB points for T30 calculation
        idx_5db = np.where(integrated_db <= -5)[0]
        idx_35db = np.where(integrated_db <= -35)[0]
        
        if len(idx_5db) > 0 and len(idx_35db) > 0:
            # Linear fit between -5dB and -35dB
            start_idx = idx_5db[0]
            end_idx = idx_35db[0]
            
            if end_idx > start_idx:
                x = np.arange(start_idx, end_idx) / sr
                y = integrated_db[start_idx:end_idx]
                
                # Robust linear fit
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                
                if slope < 0:
                    # T30 calculation (time for 30dB decay)
                    t30 = -30 / slope
                    # Convert T30 to T60
                    t60 = 2 * t30
                    results['t60_overall'] = np.clip(t60, 0.1, 5.0)
                else:
                    results['t60_overall'] = 1.0
            else:
                results['t60_overall'] = 1.0
        else:
            results['t60_overall'] = 1.0
        
        # Frequency band analysis
        for i, (low_freq, high_freq) in enumerate(freq_bands):
            try:
                # Apply bandpass filter
                nyquist = sr / 2
                low = low_freq / nyquist
                high = min(high_freq / nyquist, 0.99)
                
                if high > low:
                    b, a = signal.butter(4, [low, high], btype='band')
                    filtered_audio = signal.filtfilt(b, a, audio)
                    
                    # Calculate T60 for this band
                    band_energy = filtered_audio ** 2
                    band_integrated = np.cumsum(band_energy[::-1])[::-1]
                    band_db = 10 * np.log10(band_integrated + 1e-10)
                    band_db = band_db - np.max(band_db)
                    
                    # Find decay points
                    idx_5db_band = np.where(band_db <= -5)[0]
                    idx_35db_band = np.where(band_db <= -35)[0]
                    
                    if len(idx_5db_band) > 0 and len(idx_35db_band) > 0:
                        start_idx = idx_5db_band[0]
                        end_idx = idx_35db_band[0]
                        
                        if end_idx > start_idx:
                            x_band = np.arange(start_idx, end_idx) / sr
                            y_band = band_db[start_idx:end_idx]
                            
                            coeffs_band = np.polyfit(x_band, y_band, 1)
                            slope_band = coeffs_band[0]
                            
                            if slope_band < 0:
                                t30_band = -30 / slope_band
                                t60_band = 2 * t30_band
                                results[f't60_{low_freq}_{high_freq}Hz'] = np.clip(t60_band, 0.1, 5.0)
                            else:
                                results[f't60_{low_freq}_{high_freq}Hz'] = 1.0
                        else:
                            results[f't60_{low_freq}_{high_freq}Hz'] = 1.0
                    else:
                        results[f't60_{low_freq}_{high_freq}Hz'] = 1.0
                else:
                    results[f't60_{low_freq}_{high_freq}Hz'] = 1.0
                    
            except Exception as e:
                logger.warning(f"Error in band {low_freq}-{high_freq}Hz: {e}")
                results[f't60_{low_freq}_{high_freq}Hz'] = 1.0
        
    except Exception as e:
        logger.warning(f"Error in T60 estimation: {e}")
        results['t60_overall'] = 1.0
    
    return results


def calculate_clarity_measures(clean_audio: np.ndarray, reverb_audio: np.ndarray, 
                             sr: int) -> Dict[str, float]:
    """
    Calculate clarity measures (C50, C80, DRR) between clean and reverberant audio.
    
    Args:
        clean_audio: Clean audio signal
        reverb_audio: Reverberant audio signal
        sr: Sample rate
        
    Returns:
        Dictionary with clarity measures
    """
    
    results = {}
    
    try:
        # Ensure same length
        min_len = min(len(clean_audio), len(reverb_audio))
        clean_audio = clean_audio[:min_len]
        reverb_audio = reverb_audio[:min_len]
        
        # Direct-to-Reverberant Ratio (DRR)
        clean_energy = np.mean(clean_audio ** 2)
        reverb_energy = np.mean(reverb_audio ** 2)
        
        if reverb_energy > 0:
            drr = 10 * np.log10((clean_energy / reverb_energy) + 1e-8)
        else:
            drr = 0.0
        
        results['drr'] = drr
        
        # Estimate impulse response from reverberant signal
        # This is a simplified approach - in practice, you would have the actual RIR
        try:
            # Cross-correlation to estimate impulse response
            correlation = np.correlate(reverb_audio, clean_audio, mode='full')
            max_idx = np.argmax(np.abs(correlation))
            
            # Extract estimated impulse response
            ir_length = min(sr, len(correlation) - max_idx)  # 1 second max
            estimated_ir = correlation[max_idx:max_idx + ir_length]
            
            # Normalize
            estimated_ir = estimated_ir / np.max(np.abs(estimated_ir))
            
            # Calculate C50 and C80
            # C50: ratio of energy in first 50ms to energy after 50ms
            ms50_samples = int(0.05 * sr)  # 50ms
            ms80_samples = int(0.08 * sr)  # 80ms
            
            if len(estimated_ir) > ms50_samples:
                early_energy_50 = np.sum(estimated_ir[:ms50_samples] ** 2)
                late_energy_50 = np.sum(estimated_ir[ms50_samples:] ** 2)
                
                if late_energy_50 > 0:
                    c50 = 10 * np.log10(early_energy_50 / late_energy_50)
                    results['c50'] = c50
            
            if len(estimated_ir) > ms80_samples:
                early_energy_80 = np.sum(estimated_ir[:ms80_samples] ** 2)
                late_energy_80 = np.sum(estimated_ir[ms80_samples:] ** 2)
                
                if late_energy_80 > 0:
                    c80 = 10 * np.log10(early_energy_80 / late_energy_80)
                    results['c80'] = c80
        
        except Exception as e:
            logger.warning(f"Error calculating clarity measures: {e}")
    
    except Exception as e:
        logger.warning(f"Error in clarity calculation: {e}")
    
    return results


def analyze_dataset_comprehensive(csv_path: str, output_dir: str, 
                                sample_size: Optional[int] = None,
                                analyze_audio: bool = False) -> Dict:
    """
    Perform comprehensive analysis of a reverberant speech dataset.
    
    Args:
        csv_path: Path to dataset CSV file
        output_dir: Directory to save analysis results
        sample_size: Number of samples to analyze (None for all)
        analyze_audio: Whether to perform detailed audio analysis
        
    Returns:
        Dictionary with analysis results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset with {len(df)} samples")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Analyzing sample of {len(df)} files")
    
    analysis_results = {
        'dataset_info': {
            'total_samples': len(df),
            'columns': list(df.columns)
        },
        'metadata_analysis': {},
        'audio_analysis': {},
        'visualizations': []
    }
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Basic metadata analysis
    logger.info("Analyzing metadata...")
    
    # Subset distribution
    if 'subset' in df.columns:
        subset_dist = df['subset'].value_counts().to_dict()
        analysis_results['metadata_analysis']['subset_distribution'] = subset_dist
        
        # Plot subset distribution
        plt.figure(figsize=(8, 6))
        df['subset'].value_counts().plot(kind='bar')
        plt.title('Dataset Subset Distribution')
        plt.xlabel('Subset')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subset_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        analysis_results['visualizations'].append('subset_distribution.png')
    
    # T60 analysis
    if 't60' in df.columns:
        t60_values = df['t60'].dropna()
        if len(t60_values) > 0:
            t60_stats = {
                'count': len(t60_values),
                'mean': float(t60_values.mean()),
                'std': float(t60_values.std()),
                'min': float(t60_values.min()),
                'max': float(t60_values.max()),
                'percentiles': t60_values.quantile([0.25, 0.5, 0.75]).to_dict()
            }
            analysis_results['metadata_analysis']['t60_statistics'] = t60_stats
            
            # Plot T60 distribution
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histogram
            axes[0, 0].hist(t60_values, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('T60 (seconds)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('T60 Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[0, 1].boxplot(t60_values)
            axes[0, 1].set_ylabel('T60 (seconds)')
            axes[0, 1].set_title('T60 Box Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # CDF
            sorted_t60 = np.sort(t60_values)
            cdf = np.arange(1, len(sorted_t60) + 1) / len(sorted_t60)
            axes[1, 0].plot(sorted_t60, cdf)
            axes[1, 0].set_xlabel('T60 (seconds)')
            axes[1, 0].set_ylabel('Cumulative Probability')
            axes[1, 0].set_title('T60 Cumulative Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # By subset if available
            if 'subset' in df.columns:
                for subset in df['subset'].unique():
                    subset_t60 = df[df['subset'] == subset]['t60'].dropna()
                    if len(subset_t60) > 0:
                        axes[1, 1].hist(subset_t60, alpha=0.6, label=subset, bins=20)
                
                axes[1, 1].set_xlabel('T60 (seconds)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('T60 Distribution by Subset')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 't60_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            analysis_results['visualizations'].append('t60_analysis.png')
    
    # DRR analysis
    if 'drr' in df.columns:
        drr_values = df['drr'].dropna()
        if len(drr_values) > 0:
            drr_stats = {
                'count': len(drr_values),
                'mean': float(drr_values.mean()),
                'std': float(drr_values.std()),
                'min': float(drr_values.min()),
                'max': float(drr_values.max()),
                'percentiles': drr_values.quantile([0.25, 0.5, 0.75]).to_dict()
            }
            analysis_results['metadata_analysis']['drr_statistics'] = drr_stats
            
            # Plot DRR distribution
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histogram
            axes[0, 0].hist(drr_values, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('DRR (dB)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('DRR Distribution')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[0, 1].boxplot(drr_values)
            axes[0, 1].set_ylabel('DRR (dB)')
            axes[0, 1].set_title('DRR Box Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # CDF
            sorted_drr = np.sort(drr_values)
            cdf = np.arange(1, len(sorted_drr) + 1) / len(sorted_drr)
            axes[1, 0].plot(sorted_drr, cdf)
            axes[1, 0].set_xlabel('DRR (dB)')
            axes[1, 0].set_ylabel('Cumulative Probability')
            axes[1, 0].set_title('DRR Cumulative Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # By subset if available
            if 'subset' in df.columns:
                for subset in df['subset'].unique():
                    subset_drr = df[df['subset'] == subset]['drr'].dropna()
                    if len(subset_drr) > 0:
                        axes[1, 1].hist(subset_drr, alpha=0.6, label=subset, bins=20)
                
                axes[1, 1].set_xlabel('DRR (dB)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('DRR Distribution by Subset')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'drr_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            analysis_results['visualizations'].append('drr_analysis.png')
    
    # Room type analysis
    if 'room_type' in df.columns:
        room_dist = df['room_type'].value_counts().to_dict()
        analysis_results['metadata_analysis']['room_type_distribution'] = room_dist
        
        # Plot room type distribution
        plt.figure(figsize=(10, 6))
        df['room_type'].value_counts().plot(kind='bar')
        plt.title('Room Type Distribution')
        plt.xlabel('Room Type')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'room_type_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        analysis_results['visualizations'].append('room_type_distribution.png')
    
    # T60 vs DRR scatter plot
    if 't60' in df.columns and 'drr' in df.columns:
        plt.figure(figsize=(10, 8))
        
        if 'room_type' in df.columns:
            # Color by room type
            for room_type in df['room_type'].unique():
                if pd.notna(room_type):
                    subset_df = df[df['room_type'] == room_type]
                    plt.scatter(subset_df['t60'], subset_df['drr'], 
                              label=room_type, alpha=0.6, s=50)
            plt.legend()
        else:
            plt.scatter(df['t60'], df['drr'], alpha=0.6, s=50)
        
        plt.xlabel('T60 (seconds)')
        plt.ylabel('DRR (dB)')
        plt.title('T60 vs DRR')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 't60_vs_drr.png'), dpi=300, bbox_inches='tight')
        plt.close()
        analysis_results['visualizations'].append('t60_vs_drr.png')
    
    # Correlation matrix for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        analysis_results['metadata_analysis']['correlations'] = correlation_matrix.to_dict()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        analysis_results['visualizations'].append('correlation_matrix.png')
    
    # Audio analysis if requested
    if analyze_audio:
        logger.info("Performing audio analysis...")
        
        audio_stats = {
            'analyzed_samples': 0,
            'duration_stats': {},
            'energy_stats': {},
            'spectral_stats': {}
        }
        
        # Analyze a subset of audio files
        analysis_sample_size = min(100, len(df))  # Limit to 100 files for speed
        analysis_df = df.sample(n=analysis_sample_size, random_state=42)
        
        durations = []
        energies = []
        centroids = []
        
        for idx, row in tqdm(analysis_df.iterrows(), total=len(analysis_df), desc="Analyzing audio"):
            # Analyze clean file
            if 'clean_path' in row and pd.notna(row['clean_path']) and os.path.exists(row['clean_path']):
                clean_analysis = analyze_audio_file(row['clean_path'])
                if clean_analysis:
                    durations.append(clean_analysis.get('duration', 0))
                    energies.append(clean_analysis.get('rms_energy', 0))
                    centroids.append(clean_analysis.get('spectral_centroid', 0))
                    
                    audio_stats['analyzed_samples'] += 1
        
        # Calculate statistics
        if durations:
            audio_stats['duration_stats'] = {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations))
            }
        
        if energies:
            audio_stats['energy_stats'] = {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies))
            }
        
        if centroids:
            audio_stats['spectral_stats'] = {
                'centroid_mean': float(np.mean(centroids)),
                'centroid_std': float(np.std(centroids))
            }
        
        analysis_results['audio_analysis'] = audio_stats
    
    # Save analysis results
    results_file = os.path.join(output_dir, 'analysis_results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Create summary report
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Dataset Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total samples: {analysis_results['dataset_info']['total_samples']}\n")
        f.write(f"Columns: {', '.join(analysis_results['dataset_info']['columns'])}\n\n")
        
        if 'subset_distribution' in analysis_results['metadata_analysis']:
            f.write("Subset Distribution:\n")
            for subset, count in analysis_results['metadata_analysis']['subset_distribution'].items():
                f.write(f"  {subset}: {count}\n")
            f.write("\n")
        
        if 't60_statistics' in analysis_results['metadata_analysis']:
            stats = analysis_results['metadata_analysis']['t60_statistics']
            f.write(f"T60 Statistics:\n")
            f.write(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f} seconds\n")
            f.write(f"  Range: {stats['min']:.3f} - {stats['max']:.3f} seconds\n")
            f.write(f"  Median: {stats['percentiles'][0.5]:.3f} seconds\n\n")
        
        if 'drr_statistics' in analysis_results['metadata_analysis']:
            stats = analysis_results['metadata_analysis']['drr_statistics']
            f.write(f"DRR Statistics:\n")
            f.write(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f} dB\n")
            f.write(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} dB\n")
            f.write(f"  Median: {stats['percentiles'][0.5]:.2f} dB\n\n")
        
        if 'room_type_distribution' in analysis_results['metadata_analysis']:
            f.write("Room Type Distribution:\n")
            for room_type, count in analysis_results['metadata_analysis']['room_type_distribution'].items():
                f.write(f"  {room_type}: {count}\n")
            f.write("\n")
        
        if analysis_results['visualizations']:
            f.write("Generated Visualizations:\n")
            for viz in analysis_results['visualizations']:
                f.write(f"  {viz}\n")
    
    logger.info(f"Analysis completed. Results saved to {output_dir}")
    
    return analysis_results


def main():
    parser = argparse.ArgumentParser(description="Analyze reverberant speech dataset")
    
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to analyze (default: all)')
    parser.add_argument('--analyze_audio', action='store_true',
                       help='Perform detailed audio analysis (slower)')
    
    args = parser.parse_args()
    
    # Perform analysis
    results = analyze_dataset_comprehensive(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        analyze_audio=args.analyze_audio
    )
    
    print(f"\nAnalysis completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Analyzed {results['dataset_info']['total_samples']} samples")


if __name__ == '__main__':
    main()