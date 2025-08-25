#!/usr/bin/env python3
"""
Advanced Metrics Calculation for Dereverberation

This module provides comprehensive evaluation metrics for both speech and music
dereverberation, including:
- PESQ (Perceptual Evaluation of Speech Quality) for speech
- SDR (Source-to-Distortion Ratio) for music  
- Additional perceptual and objective metrics
- CSV output for Kaggle competition format
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

def detect_content_type(audio_file: str, duration_threshold: float = 10.0) -> str:
    """
    Detect if audio content is speech or music based on various features
    
    Args:
        audio_file: Path to audio file
        duration_threshold: Threshold for duration-based classification
        
    Returns:
        'speech' or 'music'
    """
    try:
        import librosa
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        duration = len(y) / sr
        
        # Feature extraction for classification
        # 1. Duration-based heuristic
        if duration < duration_threshold:
            content_bias = "speech"
        else:
            content_bias = "music"
        
        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # 3. Tempo and rhythm features
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo_stability = np.std(np.diff(beats))
        except:
            tempo = 0
            tempo_stability = 0
        
        # 4. Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-10)
        
        # Classification logic
        music_indicators = 0
        speech_indicators = 0
        
        # Music indicators
        if np.mean(spectral_centroids) > 2000:  # Higher spectral centroid
            music_indicators += 1
        if tempo > 60 and tempo < 200:  # Typical music tempo
            music_indicators += 1
        if harmonic_ratio > 0.3:  # Strong harmonic content
            music_indicators += 1
        if duration > duration_threshold:  # Longer duration
            music_indicators += 1
        
        # Speech indicators  
        if np.mean(zero_crossing_rate) > 0.1:  # Higher ZCR typical of speech
            speech_indicators += 1
        if tempo_stability < 0.5:  # Less rhythmic stability
            speech_indicators += 1
        if duration < duration_threshold:  # Shorter duration
            speech_indicators += 1
        
        # Final decision
        if music_indicators > speech_indicators:
            return "music"
        else:
            return "speech"
            
    except Exception as e:
        print(f"Error in content detection for {audio_file}: {e}")
        # Fallback to duration-based classification
        try:
            import soundfile as sf
            info = sf.info(audio_file)
            if info.duration > duration_threshold:
                return "music"
            else:
                return "speech"
        except:
            return "speech"  # Default fallback

def calculate_pesq_score(reference: np.ndarray, enhanced: np.ndarray, 
                        sr: int = 16000) -> float:
    """
    Calculate PESQ score for speech quality evaluation
    
    Args:
        reference: Clean reference audio
        enhanced: Enhanced audio
        sr: Sample rate
        
    Returns:
        PESQ score
    """
    try:
        from pesq import pesq
        
        # Ensure 16kHz for PESQ
        if sr != 16000:
            import librosa
            reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
            enhanced = librosa.resample(enhanced, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Calculate PESQ
        pesq_score = pesq(sr, reference, enhanced, 'wb')
        return pesq_score
        
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return 0.0

def calculate_sdr_score(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Calculate SDR (Source-to-Distortion Ratio) for music quality evaluation
    
    Args:
        reference: Clean reference audio
        enhanced: Enhanced audio
        
    Returns:
        SDR score in dB
    """
    try:
        # Calculate signal and noise+distortion powers
        signal_power = np.mean(reference ** 2)
        error = enhanced - reference
        noise_power = np.mean(error ** 2)
        
        # SDR calculation
        if noise_power > 0:
            sdr = 10 * np.log10(signal_power / noise_power)
        else:
            sdr = 100.0  # Perfect reconstruction
            
        return sdr
        
    except Exception as e:
        print(f"Error calculating SDR: {e}")
        return 0.0

def calculate_stoi_score(reference: np.ndarray, enhanced: np.ndarray, 
                        sr: int = 16000) -> float:
    """
    Calculate STOI (Short-Time Objective Intelligibility) score
    
    Args:
        reference: Clean reference audio
        enhanced: Enhanced audio  
        sr: Sample rate
        
    Returns:
        STOI score
    """
    try:
        from pystoi import stoi
        
        stoi_score = stoi(reference, enhanced, sr, extended=True)
        return stoi_score
        
    except Exception as e:
        print(f"Error calculating STOI: {e}")
        return 0.0

def calculate_si_sdr(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Calculate SI-SDR (Scale-Invariant Source-to-Distortion Ratio)
    
    Args:
        reference: Clean reference audio
        enhanced: Enhanced audio
        
    Returns:
        SI-SDR score in dB
    """
    try:
        # Normalize
        reference = reference / np.sqrt(np.sum(reference**2))
        enhanced = enhanced / np.sqrt(np.sum(enhanced**2))
        
        # Scale-invariant target
        alpha = np.dot(enhanced, reference)
        target = alpha * reference
        noise = enhanced - target
        
        # SI-SDR calculation
        si_sdr = 10 * np.log10(np.sum(target**2) / (np.sum(noise**2) + 1e-10))
        
        return si_sdr
        
    except Exception as e:
        print(f"Error calculating SI-SDR: {e}")
        return 0.0

def calculate_spectral_convergence(reference: np.ndarray, enhanced: np.ndarray, 
                                 sr: int = 16000) -> float:
    """
    Calculate spectral convergence metric
    
    Args:
        reference: Clean reference audio
        enhanced: Enhanced audio
        sr: Sample rate
        
    Returns:
        Spectral convergence score
    """
    try:
        import librosa
        
        # Compute spectrograms
        ref_spec = np.abs(librosa.stft(reference))
        enh_spec = np.abs(librosa.stft(enhanced))
        
        # Align dimensions
        min_frames = min(ref_spec.shape[1], enh_spec.shape[1])
        ref_spec = ref_spec[:, :min_frames]
        enh_spec = enh_spec[:, :min_frames]
        
        # Spectral convergence
        numerator = np.sum((ref_spec - enh_spec) ** 2)
        denominator = np.sum(ref_spec ** 2)
        
        if denominator > 0:
            sc = np.sqrt(numerator / denominator)
        else:
            sc = 1.0
            
        return sc
        
    except Exception as e:
        print(f"Error calculating spectral convergence: {e}")
        return 1.0

def calculate_log_spectral_distance(reference: np.ndarray, enhanced: np.ndarray,
                                  sr: int = 16000) -> float:
    """
    Calculate Log-Spectral Distance (LSD)
    
    Args:
        reference: Clean reference audio
        enhanced: Enhanced audio
        sr: Sample rate
        
    Returns:
        LSD score
    """
    try:
        import librosa
        
        # Compute magnitude spectrograms
        ref_spec = np.abs(librosa.stft(reference))
        enh_spec = np.abs(librosa.stft(enhanced))
        
        # Align dimensions
        min_frames = min(ref_spec.shape[1], enh_spec.shape[1])
        ref_spec = ref_spec[:, :min_frames]
        enh_spec = enh_spec[:, :min_frames]
        
        # Convert to log scale
        ref_log = np.log(ref_spec + 1e-10)
        enh_log = np.log(enh_spec + 1e-10)
        
        # LSD calculation
        lsd = np.sqrt(np.mean((ref_log - enh_log) ** 2))
        
        return lsd
        
    except Exception as e:
        print(f"Error calculating LSD: {e}")
        return 10.0

class AdvancedMetricsCalculator:
    """
    Advanced metrics calculator for comprehensive evaluation
    """
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.results = []
        
    def load_audio_pair(self, reference_path: str, enhanced_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Load and align reference and enhanced audio files
        
        Args:
            reference_path: Path to reference audio
            enhanced_path: Path to enhanced audio
            
        Returns:
            Tuple of (reference, enhanced, sample_rate)
        """
        try:
            import soundfile as sf
            
            # Load both files
            reference, sr_ref = sf.read(reference_path)
            enhanced, sr_enh = sf.read(enhanced_path)
            
            # Ensure same sample rate
            if sr_ref != sr_enh:
                import librosa
                if sr_ref != self.target_sr:
                    reference = librosa.resample(reference, orig_sr=sr_ref, target_sr=self.target_sr)
                if sr_enh != self.target_sr:
                    enhanced = librosa.resample(enhanced, orig_sr=sr_enh, target_sr=self.target_sr)
                sr = self.target_sr
            else:
                sr = sr_ref
            
            # Ensure mono
            if reference.ndim > 1:
                reference = np.mean(reference, axis=1)
            if enhanced.ndim > 1:
                enhanced = np.mean(enhanced, axis=1)
            
            # Align lengths
            min_length = min(len(reference), len(enhanced))
            reference = reference[:min_length]
            enhanced = enhanced[:min_length]
            
            return reference, enhanced, sr
            
        except Exception as e:
            print(f"Error loading audio pair {reference_path}, {enhanced_path}: {e}")
            return None, None, None
    
    def calculate_all_metrics(self, reference: np.ndarray, enhanced: np.ndarray, 
                            sr: int, content_type: str) -> Dict:
        """
        Calculate all relevant metrics based on content type
        
        Args:
            reference: Clean reference audio
            enhanced: Enhanced audio
            sr: Sample rate
            content_type: 'speech' or 'music'
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'content_type': content_type,
            'sample_rate': sr,
            'duration': len(reference) / sr
        }
        
        # Common metrics for both speech and music
        metrics['si_sdr'] = calculate_si_sdr(reference, enhanced)
        metrics['spectral_convergence'] = calculate_spectral_convergence(reference, enhanced, sr)
        metrics['log_spectral_distance'] = calculate_log_spectral_distance(reference, enhanced, sr)
        
        # Content-specific metrics
        if content_type == 'speech':
            metrics['pesq'] = calculate_pesq_score(reference, enhanced, sr)
            metrics['stoi'] = calculate_stoi_score(reference, enhanced, sr)
            # Primary score for speech is PESQ
            metrics['primary_score'] = metrics['pesq']
            
        elif content_type == 'music':
            metrics['sdr'] = calculate_sdr_score(reference, enhanced)
            # Primary score for music is SDR
            metrics['primary_score'] = metrics['sdr']
            
        # Additional perceptual metrics
        try:
            # Energy ratio
            ref_energy = np.mean(reference ** 2)
            enh_energy = np.mean(enhanced ** 2)
            metrics['energy_ratio'] = enh_energy / (ref_energy + 1e-10)
            
            # Dynamic range
            metrics['ref_dynamic_range'] = np.max(np.abs(reference)) - np.min(np.abs(reference))
            metrics['enh_dynamic_range'] = np.max(np.abs(enhanced)) - np.min(np.abs(enhanced))
            
        except Exception as e:
            print(f"Error calculating additional metrics: {e}")
        
        return metrics
    
    def process_file_pair(self, reference_path: str, enhanced_path: str, 
                         filename: str = None) -> Dict:
        """
        Process a single file pair and calculate all metrics
        
        Args:
            reference_path: Path to reference audio
            enhanced_path: Path to enhanced audio
            filename: Optional filename for identification
            
        Returns:
            Dictionary with all results
        """
        if filename is None:
            filename = os.path.basename(enhanced_path)
        
        # Load audio
        reference, enhanced, sr = self.load_audio_pair(reference_path, enhanced_path)
        
        if reference is None or enhanced is None:
            return {
                'filename': filename,
                'error': 'Failed to load audio files'
            }
        
        # Detect content type
        content_type = detect_content_type(reference_path)
        
        # Calculate metrics
        metrics = self.calculate_all_metrics(reference, enhanced, sr, content_type)
        metrics['filename'] = filename
        metrics['reference_path'] = reference_path
        metrics['enhanced_path'] = enhanced_path
        
        return metrics
    
    def process_directory_pair(self, reference_dir: str, enhanced_dir: str) -> pd.DataFrame:
        """
        Process entire directories of reference and enhanced audio files
        
        Args:
            reference_dir: Directory with reference audio files
            enhanced_dir: Directory with enhanced audio files
            
        Returns:
            DataFrame with all results
        """
        # Find all audio files in enhanced directory
        audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
        enhanced_files = []
        
        for ext in audio_extensions:
            enhanced_files.extend(Path(enhanced_dir).rglob(f'*{ext}'))
        
        results = []
        
        for enhanced_file in tqdm(enhanced_files, desc="Calculating metrics"):
            # Find corresponding reference file
            rel_path = enhanced_file.relative_to(enhanced_dir)
            
            # Try different naming conventions
            possible_ref_paths = [
                Path(reference_dir) / rel_path,
                Path(reference_dir) / rel_path.with_suffix('.wav'),
                Path(reference_dir) / rel_path.stem / rel_path.with_suffix('.wav'),
            ]
            
            reference_file = None
            for ref_path in possible_ref_paths:
                if ref_path.exists():
                    reference_file = ref_path
                    break
            
            if reference_file is None:
                print(f"Warning: No reference file found for {enhanced_file}")
                continue
            
            # Process file pair
            result = self.process_file_pair(str(reference_file), str(enhanced_file), str(rel_path))
            results.append(result)
        
        return pd.DataFrame(results)
    
    def save_kaggle_format(self, results_df: pd.DataFrame, output_path: str):
        """
        Save results in Kaggle submission format
        
        Args:
            results_df: DataFrame with results
            output_path: Path to save CSV file
        """
        # Separate speech and music results
        speech_results = results_df[results_df['content_type'] == 'speech'].copy()
        music_results = results_df[results_df['content_type'] == 'music'].copy()
        
        # Create Kaggle submission format
        kaggle_data = []
        
        for _, row in results_df.iterrows():
            if 'error' not in row:
                entry = {
                    'filename': row['filename'],
                    'content_type': row['content_type'],
                    'primary_score': row['primary_score'],
                    'si_sdr': row['si_sdr']
                }
                
                if row['content_type'] == 'speech':
                    entry['pesq'] = row.get('pesq', 0.0)
                    entry['stoi'] = row.get('stoi', 0.0)
                elif row['content_type'] == 'music':
                    entry['sdr'] = row.get('sdr', 0.0)
                
                kaggle_data.append(entry)
        
        kaggle_df = pd.DataFrame(kaggle_data)
        kaggle_df.to_csv(output_path, index=False)
        
        # Print summary statistics
        if len(speech_results) > 0:
            speech_pesq_mean = speech_results['pesq'].mean()
            speech_pesq_std = speech_results['pesq'].std()
            print(f"Speech PESQ: {speech_pesq_mean:.3f} ± {speech_pesq_std:.3f}")
        
        if len(music_results) > 0:
            music_sdr_mean = music_results['sdr'].mean()
            music_sdr_std = music_results['sdr'].std()
            print(f"Music SDR: {music_sdr_mean:.3f} ± {music_sdr_std:.3f}")
        
        overall_score = kaggle_df['primary_score'].mean()
        print(f"Overall Primary Score: {overall_score:.3f}")
        
        print(f"Kaggle submission saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Metrics Calculation for Dereverberation")
    parser.add_argument("--reference_dir", type=str, required=True,
                       help="Directory containing reference (clean) audio files")
    parser.add_argument("--enhanced_dir", type=str, required=True,
                       help="Directory containing enhanced audio files")
    parser.add_argument("--output_csv", type=str, default="metrics_results.csv",
                       help="Path to save metrics CSV file")
    parser.add_argument("--target_sr", type=int, default=16000,
                       help="Target sample rate for processing")
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = AdvancedMetricsCalculator(target_sr=args.target_sr)
    
    # Process all files
    print("Starting metrics calculation...")
    results_df = calculator.process_directory_pair(args.reference_dir, args.enhanced_dir)
    
    # Save detailed results
    detailed_path = args.output_csv.replace('.csv', '_detailed.csv')
    results_df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")
    
    # Save Kaggle format
    calculator.save_kaggle_format(results_df, args.output_csv)

if __name__ == "__main__":
    main()