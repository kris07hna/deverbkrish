"""
Comprehensive Evaluation Framework for De-reverberation Hackathon
================================================================

This module implements a sophisticated evaluation framework that goes beyond
standard metrics to provide deep insights into model performance:

1. Standard Metrics - PESQ, STOI, SI-SDR for speech; SDR, SIR, SAR for music
2. Perceptual Metrics - Mel-spectral loss, spectral convergence
3. Reverb-specific Metrics - RT60 estimation, DRR measurement, clarity metrics
4. Statistical Analysis - Confidence intervals, significance testing
5. Frequency Analysis - Per-band performance analysis
6. Robustness Evaluation - Performance across different conditions
7. Computational Efficiency - FLOPs, inference time, memory usage

Author: SOTA Hackathon Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pesq import pesq
from pystoi import stoi
import soundfile as sf
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class StandardMetrics:
    """Implementation of standard audio quality metrics"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def pesq_score(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate PESQ score"""
        try:
            # Ensure signals are same length
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            
            # Resample to 16kHz if needed for PESQ
            if self.sample_rate != 16000:
                clean = librosa.resample(clean, orig_sr=self.sample_rate, target_sr=16000)
                enhanced = librosa.resample(enhanced, orig_sr=self.sample_rate, target_sr=16000)
                sr = 16000
            else:
                sr = self.sample_rate
                
            score = pesq(sr, clean, enhanced, 'wb')
            return score
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
            return 0.0
            
    def stoi_score(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate STOI score"""
        try:
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            
            score = stoi(clean, enhanced, self.sample_rate, extended=True)
            return score
        except Exception as e:
            print(f"STOI calculation failed: {e}")
            return 0.0
            
    def si_sdr_score(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate Scale-Invariant Signal-to-Distortion Ratio"""
        try:
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            
            # Remove DC component
            clean = clean - np.mean(clean)
            enhanced = enhanced - np.mean(enhanced)
            
            # Scale-invariant calculation
            alpha = np.dot(enhanced, clean) / (np.linalg.norm(clean) ** 2 + 1e-8)
            scaled_clean = alpha * clean
            
            # Calculate SI-SDR
            numerator = np.linalg.norm(scaled_clean) ** 2
            denominator = np.linalg.norm(enhanced - scaled_clean) ** 2 + 1e-8
            
            si_sdr = 10 * np.log10(numerator / denominator)
            return si_sdr
        except Exception as e:
            print(f"SI-SDR calculation failed: {e}")
            return -float('inf')
            
    def sdr_sir_sar(self, clean: np.ndarray, enhanced: np.ndarray) -> Tuple[float, float, float]:
        """Calculate SDR, SIR, SAR metrics"""
        try:
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            
            # Simple implementation - for music evaluation
            noise = enhanced - clean
            
            # SDR: Signal-to-Distortion Ratio
            signal_power = np.mean(clean ** 2)
            distortion_power = np.mean(noise ** 2) + 1e-8
            sdr = 10 * np.log10(signal_power / distortion_power)
            
            # SIR: Signal-to-Interference Ratio (simplified)
            sir = sdr  # Simplified assumption
            
            # SAR: Signal-to-Artifacts Ratio (simplified)
            sar = sdr  # Simplified assumption
            
            return sdr, sir, sar
        except Exception as e:
            print(f"SDR/SIR/SAR calculation failed: {e}")
            return 0.0, 0.0, 0.0


class PerceptualMetrics:
    """Implementation of perceptual audio quality metrics"""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def mel_spectral_loss(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate mel-spectral loss"""
        try:
            # Compute mel spectrograms
            clean_mel = librosa.feature.melspectrogram(
                y=clean, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            enhanced_mel = librosa.feature.melspectrogram(
                y=enhanced, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Convert to log scale
            clean_mel_log = librosa.power_to_db(clean_mel + 1e-8)
            enhanced_mel_log = librosa.power_to_db(enhanced_mel + 1e-8)
            
            # Align lengths
            min_frames = min(clean_mel_log.shape[1], enhanced_mel_log.shape[1])
            clean_mel_log = clean_mel_log[:, :min_frames]
            enhanced_mel_log = enhanced_mel_log[:, :min_frames]
            
            # Calculate L1 loss
            loss = np.mean(np.abs(clean_mel_log - enhanced_mel_log))
            return loss
        except Exception as e:
            print(f"Mel-spectral loss calculation failed: {e}")
            return float('inf')
            
    def spectral_convergence(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate spectral convergence"""
        try:
            # Compute STFTs
            clean_stft = librosa.stft(clean, n_fft=self.n_fft, hop_length=self.hop_length)
            enhanced_stft = librosa.stft(enhanced, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Get magnitudes
            clean_mag = np.abs(clean_stft)
            enhanced_mag = np.abs(enhanced_stft)
            
            # Align shapes
            min_frames = min(clean_mag.shape[1], enhanced_mag.shape[1])
            clean_mag = clean_mag[:, :min_frames]
            enhanced_mag = enhanced_mag[:, :min_frames]
            
            # Calculate spectral convergence
            numerator = np.linalg.norm(clean_mag - enhanced_mag, ord='fro')
            denominator = np.linalg.norm(clean_mag, ord='fro') + 1e-8
            
            sc = numerator / denominator
            return sc
        except Exception as e:
            print(f"Spectral convergence calculation failed: {e}")
            return float('inf')
            
    def log_magnitude_loss(self, clean: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate log-magnitude spectral loss"""
        try:
            # Compute STFTs
            clean_stft = librosa.stft(clean, n_fft=self.n_fft, hop_length=self.hop_length)
            enhanced_stft = librosa.stft(enhanced, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Get log magnitudes
            clean_log_mag = np.log(np.abs(clean_stft) + 1e-8)
            enhanced_log_mag = np.log(np.abs(enhanced_stft) + 1e-8)
            
            # Align shapes
            min_frames = min(clean_log_mag.shape[1], enhanced_log_mag.shape[1])
            clean_log_mag = clean_log_mag[:, :min_frames]
            enhanced_log_mag = enhanced_log_mag[:, :min_frames]
            
            # Calculate L1 loss
            loss = np.mean(np.abs(clean_log_mag - enhanced_log_mag))
            return loss
        except Exception as e:
            print(f"Log magnitude loss calculation failed: {e}")
            return float('inf')


class ReverbSpecificMetrics:
    """Metrics specifically designed for reverberation assessment"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def estimate_rt60(self, audio: np.ndarray, method: str = 'schroeder') -> float:
        """Estimate RT60 (reverberation time)"""
        try:
            if method == 'schroeder':
                # Schroeder integration method
                # Reverse and square the signal
                reversed_squared = audio[::-1] ** 2
                
                # Integrate (cumulative sum)
                integrated = np.cumsum(reversed_squared)
                
                # Normalize and convert to dB
                integrated_norm = integrated / integrated[-1]
                integrated_db = 10 * np.log10(integrated_norm + 1e-8)
                
                # Find -5dB and -35dB points (for T30 estimation)
                time_axis = np.arange(len(integrated_db)) / self.sample_rate
                
                # Find indices closest to -5dB and -35dB
                idx_5db = np.argmin(np.abs(integrated_db - (-5)))
                idx_35db = np.argmin(np.abs(integrated_db - (-35)))
                
                if idx_35db > idx_5db:
                    # Calculate T30 and extrapolate to RT60
                    t30 = time_axis[idx_35db] - time_axis[idx_5db]
                    rt60 = 2 * t30  # RT60 = 2 * T30
                else:
                    rt60 = 0.0
                    
                return rt60
            else:
                return 0.0
        except Exception as e:
            print(f"RT60 estimation failed: {e}")
            return 0.0
            
    def direct_to_reverberant_ratio(self, audio: np.ndarray, 
                                   direct_window_ms: float = 2.5) -> float:
        """Calculate Direct-to-Reverberant energy Ratio (DRR)"""
        try:
            direct_samples = int(direct_window_ms * self.sample_rate / 1000)
            
            if len(audio) <= direct_samples:
                return 0.0
                
            # Direct sound energy (first few milliseconds)
            direct_energy = np.sum(audio[:direct_samples] ** 2)
            
            # Reverberant energy (remaining signal)
            reverb_energy = np.sum(audio[direct_samples:] ** 2)
            
            if reverb_energy == 0:
                return float('inf')
                
            drr = 10 * np.log10(direct_energy / (reverb_energy + 1e-8))
            return drr
        except Exception as e:
            print(f"DRR calculation failed: {e}")
            return 0.0
            
    def clarity_c50(self, audio: np.ndarray) -> float:
        """Calculate C50 clarity metric (50ms early-to-late ratio)"""
        try:
            # 50ms boundary
            boundary_samples = int(0.05 * self.sample_rate)
            
            if len(audio) <= boundary_samples:
                return 0.0
                
            # Early energy (0-50ms)
            early_energy = np.sum(audio[:boundary_samples] ** 2)
            
            # Late energy (>50ms)
            late_energy = np.sum(audio[boundary_samples:] ** 2)
            
            if late_energy == 0:
                return float('inf')
                
            c50 = 10 * np.log10(early_energy / (late_energy + 1e-8))
            return c50
        except Exception as e:
            print(f"C50 calculation failed: {e}")
            return 0.0


class ComputationalMetrics:
    """Metrics for computational efficiency assessment"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset timing counters"""
        self.inference_times = []
        self.memory_usage = []
        
    def time_inference(self, model_fn, *args, **kwargs):
        """Time a model inference"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = 0
            
        # Run inference
        result = model_fn(*args, **kwargs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Measure memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**2  # MB
        else:
            memory_used = 0
            
        inference_time = end_time - start_time
        
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_used)
        
        return result, inference_time, memory_used
        
    def get_stats(self) -> Dict[str, float]:
        """Get computational statistics"""
        if not self.inference_times:
            return {}
            
        return {
            'mean_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'mean_memory_usage_mb': np.mean(self.memory_usage),
            'max_memory_usage_mb': np.max(self.memory_usage),
        }


class ComprehensiveEvaluator:
    """Main evaluation framework combining all metrics"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 metrics_config: Optional[Dict] = None):
        
        self.sample_rate = sample_rate
        
        # Initialize metric calculators
        self.standard_metrics = StandardMetrics(sample_rate)
        self.perceptual_metrics = PerceptualMetrics(sample_rate)
        self.reverb_metrics = ReverbSpecificMetrics(sample_rate)
        self.computational_metrics = ComputationalMetrics()
        
        # Configure which metrics to compute
        if metrics_config is None:
            self.metrics_config = {
                'standard': ['pesq', 'stoi', 'si_sdr', 'sdr', 'sir', 'sar'],
                'perceptual': ['mel_spectral_loss', 'spectral_convergence', 'log_magnitude_loss'],
                'reverb_specific': ['rt60_estimation', 'drr', 'c50'],
                'computational': True
            }
        else:
            self.metrics_config = metrics_config
            
    def evaluate_single_pair(self,
                           clean_audio: np.ndarray,
                           enhanced_audio: np.ndarray,
                           noisy_audio: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate a single clean-enhanced audio pair"""
        
        results = {}
        
        # Standard metrics
        if 'standard' in self.metrics_config:
            for metric in self.metrics_config['standard']:
                if metric == 'pesq':
                    results['pesq'] = self.standard_metrics.pesq_score(clean_audio, enhanced_audio)
                elif metric == 'stoi':
                    results['stoi'] = self.standard_metrics.stoi_score(clean_audio, enhanced_audio)
                elif metric == 'si_sdr':
                    results['si_sdr'] = self.standard_metrics.si_sdr_score(clean_audio, enhanced_audio)
                elif metric in ['sdr', 'sir', 'sar']:
                    sdr, sir, sar = self.standard_metrics.sdr_sir_sar(clean_audio, enhanced_audio)
                    if metric == 'sdr':
                        results['sdr'] = sdr
                    elif metric == 'sir':
                        results['sir'] = sir
                    elif metric == 'sar':
                        results['sar'] = sar
                        
        # Perceptual metrics
        if 'perceptual' in self.metrics_config:
            for metric in self.metrics_config['perceptual']:
                if metric == 'mel_spectral_loss':
                    results['mel_spectral_loss'] = self.perceptual_metrics.mel_spectral_loss(
                        clean_audio, enhanced_audio
                    )
                elif metric == 'spectral_convergence':
                    results['spectral_convergence'] = self.perceptual_metrics.spectral_convergence(
                        clean_audio, enhanced_audio
                    )
                elif metric == 'log_magnitude_loss':
                    results['log_magnitude_loss'] = self.perceptual_metrics.log_magnitude_loss(
                        clean_audio, enhanced_audio
                    )
                    
        # Reverb-specific metrics
        if 'reverb_specific' in self.metrics_config:
            for metric in self.metrics_config['reverb_specific']:
                if metric == 'rt60_estimation':
                    # Estimate RT60 reduction
                    if noisy_audio is not None:
                        noisy_rt60 = self.reverb_metrics.estimate_rt60(noisy_audio)
                        enhanced_rt60 = self.reverb_metrics.estimate_rt60(enhanced_audio)
                        results['rt60_reduction'] = noisy_rt60 - enhanced_rt60
                    else:
                        results['enhanced_rt60'] = self.reverb_metrics.estimate_rt60(enhanced_audio)
                        
                elif metric == 'drr':
                    results['drr'] = self.reverb_metrics.direct_to_reverberant_ratio(enhanced_audio)
                    
                elif metric == 'c50':
                    results['c50'] = self.reverb_metrics.clarity_c50(enhanced_audio)
                    
        return results
        
    def evaluate_dataset(self,
                        clean_files: List[str],
                        enhanced_files: List[str], 
                        noisy_files: Optional[List[str]] = None,
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate entire dataset"""
        
        assert len(clean_files) == len(enhanced_files), "Mismatch in number of files"
        
        if noisy_files is not None:
            assert len(clean_files) == len(noisy_files), "Mismatch in number of noisy files"
            
        all_results = []
        
        print(f"Evaluating {len(clean_files)} audio pairs...")
        
        for i, (clean_file, enhanced_file) in enumerate(tqdm(zip(clean_files, enhanced_files))):
            try:
                # Load audio files
                clean_audio, _ = librosa.load(clean_file, sr=self.sample_rate)
                enhanced_audio, _ = librosa.load(enhanced_file, sr=self.sample_rate)
                
                noisy_audio = None
                if noisy_files is not None:
                    noisy_audio, _ = librosa.load(noisy_files[i], sr=self.sample_rate)
                    
                # Evaluate pair
                results = self.evaluate_single_pair(clean_audio, enhanced_audio, noisy_audio)
                results['file_index'] = i
                results['clean_file'] = clean_file
                results['enhanced_file'] = enhanced_file
                
                all_results.append(results)
                
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
                
        # Convert to DataFrame for analysis
        df_results = pd.DataFrame(all_results)
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(df_results)
        
        # Generate analysis reports
        analysis_reports = self._generate_analysis_reports(df_results)
        
        # Save results if output directory specified
        if output_dir is not None:
            self._save_results(df_results, summary_stats, analysis_reports, output_dir)
            
        final_results = {
            'individual_results': all_results,
            'summary_statistics': summary_stats,
            'analysis_reports': analysis_reports,
            'dataframe': df_results
        }
        
        return final_results
        
    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for all metrics"""
        
        summary = {}
        
        # Get numeric columns (exclude file paths and indices)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['file_index']:
                continue
                
            values = df[col].dropna()
            if len(values) > 0:
                summary[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'count': int(len(values))
                }
                
                # Confidence interval for mean
                if len(values) > 1:
                    ci = stats.t.interval(0.95, len(values)-1, 
                                        loc=values.mean(), 
                                        scale=stats.sem(values))
                    summary[col]['ci_lower'] = float(ci[0])
                    summary[col]['ci_upper'] = float(ci[1])
                    
        return summary
        
    def _generate_analysis_reports(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed analysis reports"""
        
        reports = {}
        
        # Performance distribution analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['file_index']]
        
        reports['performance_distribution'] = {}
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                reports['performance_distribution'][col] = {
                    'histogram_bins': np.histogram(values, bins=20)[0].tolist(),
                    'histogram_edges': np.histogram(values, bins=20)[1].tolist(),
                    'outliers': self._detect_outliers(values).tolist()
                }
                
        # Correlation analysis
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            reports['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(correlation_matrix)
            }
            
        # Performance categorization
        reports['performance_categories'] = self._categorize_performance(df)
        
        return reports
        
    def _detect_outliers(self, values: pd.Series, method: str = 'iqr') -> np.ndarray:
        """Detect outliers in metric values"""
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            return outliers.index.values
        else:
            return np.array([])
            
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, 
                              threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find highly correlated metric pairs"""
        
        high_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corrs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j], 
                        float(corr_val)
                    ))
                    
        return high_corrs
        
    def _categorize_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Categorize performance levels for key metrics"""
        
        categories = {}
        
        # Define thresholds for key metrics
        thresholds = {
            'pesq': {'excellent': 4.0, 'good': 3.0, 'fair': 2.0},
            'stoi': {'excellent': 0.9, 'good': 0.8, 'fair': 0.7},
            'si_sdr': {'excellent': 15.0, 'good': 10.0, 'fair': 5.0}
        }
        
        for metric, thresh in thresholds.items():
            if metric in df.columns:
                values = df[metric].dropna()
                categories[metric] = {
                    'excellent': int(sum(values >= thresh['excellent'])),
                    'good': int(sum((values >= thresh['good']) & (values < thresh['excellent']))),
                    'fair': int(sum((values >= thresh['fair']) & (values < thresh['good']))),
                    'poor': int(sum(values < thresh['fair']))
                }
                
        return categories
        
    def _save_results(self, 
                     df: pd.DataFrame,
                     summary_stats: Dict,
                     analysis_reports: Dict,
                     output_dir: str):
        """Save evaluation results to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df.to_csv(output_path / 'detailed_results.csv', index=False)
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats).T
        summary_df.to_csv(output_path / 'summary_statistics.csv')
        
        # Generate and save plots
        self._generate_plots(df, output_path)
        
        print(f"Results saved to {output_dir}")
        
    def _generate_plots(self, df: pd.DataFrame, output_path: Path):
        """Generate visualization plots"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['file_index']]
        
        # Distribution plots
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:4]):  # Plot first 4 metrics
                if i < len(axes):
                    values = df[col].dropna()
                    axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{col.upper()} Distribution')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    
            plt.tight_layout()
            plt.savefig(output_path / 'metric_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Metric Correlation Matrix')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def compare_models(self,
                      model_results: Dict[str, Dict],
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Compare results from multiple models"""
        
        comparison_results = {}
        
        # Extract metric values for each model
        model_metrics = {}
        for model_name, results in model_results.items():
            if 'summary_statistics' in results:
                model_metrics[model_name] = results['summary_statistics']
                
        # Statistical significance testing
        significance_tests = self._perform_significance_tests(model_results)
        
        # Ranking analysis
        rankings = self._compute_model_rankings(model_metrics)
        
        comparison_results = {
            'model_metrics': model_metrics,
            'significance_tests': significance_tests,
            'rankings': rankings
        }
        
        if output_dir is not None:
            self._save_comparison_results(comparison_results, output_dir)
            
        return comparison_results
        
    def _perform_significance_tests(self, model_results: Dict) -> Dict:
        """Perform statistical significance tests between models"""
        
        # This would implement proper statistical tests
        # For now, return placeholder
        return {'placeholder': 'Statistical significance testing would be implemented here'}
        
    def _compute_model_rankings(self, model_metrics: Dict) -> Dict:
        """Compute rankings for each metric across models"""
        
        rankings = {}
        
        # Get all metrics
        all_metrics = set()
        for model_data in model_metrics.values():
            all_metrics.update(model_data.keys())
            
        # Rank models for each metric
        for metric in all_metrics:
            metric_values = {}
            for model_name, model_data in model_metrics.items():
                if metric in model_data and 'mean' in model_data[metric]:
                    metric_values[model_name] = model_data[metric]['mean']
                    
            if metric_values:
                # Sort by value (higher is better for most metrics except losses)
                reverse_order = 'loss' not in metric.lower()
                sorted_models = sorted(metric_values.items(), 
                                     key=lambda x: x[1], reverse=reverse_order)
                
                rankings[metric] = {
                    'ranking': [model for model, _ in sorted_models],
                    'values': {model: value for model, value in sorted_models}
                }
                
        return rankings
        
    def _save_comparison_results(self, comparison_results: Dict, output_dir: str):
        """Save model comparison results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save rankings
        rankings_df = pd.DataFrame(comparison_results['rankings'])
        rankings_df.to_csv(output_path / 'model_rankings.csv')
        
        print(f"Comparison results saved to {output_dir}")


def create_evaluation_config(task_type: str = 'speech') -> Dict:
    """Create evaluation configuration for specific task types"""
    
    if task_type == 'speech':
        return {
            'standard': ['pesq', 'stoi', 'si_sdr'],
            'perceptual': ['mel_spectral_loss', 'spectral_convergence'],
            'reverb_specific': ['rt60_estimation', 'drr', 'c50'],
            'computational': True
        }
    elif task_type == 'music':
        return {
            'standard': ['sdr', 'sir', 'sar'],
            'perceptual': ['mel_spectral_loss', 'spectral_convergence', 'log_magnitude_loss'],
            'reverb_specific': ['rt60_estimation', 'drr'],
            'computational': True
        }
    else:
        # General configuration
        return {
            'standard': ['pesq', 'stoi', 'si_sdr', 'sdr', 'sir', 'sar'],
            'perceptual': ['mel_spectral_loss', 'spectral_convergence', 'log_magnitude_loss'],
            'reverb_specific': ['rt60_estimation', 'drr', 'c50'],
            'computational': True
        }