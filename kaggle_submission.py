#!/usr/bin/env python3
"""
Kaggle Submission Script for Dereverberation Challenge

This script provides a standardized interface for dereverberation inference
compatible with Kaggle competition format. It includes:
- Ensemble inference with multiple models
- Adaptive sampling strategies  
- Model complexity monitoring (GMAC/s constraint)
- Standardized CSV output format
- Support for both speech and music signals
"""

import os
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def calculate_model_complexity(model: nn.Module, input_shape: Tuple[int, ...]) -> float:
    """
    Calculate model complexity in GMAC/s (Giga Multiply-Accumulate operations per second)
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (B, C, H, W) or (B, C, T)
        
    Returns:
        GMAC/s complexity measure
    """
    try:
        from torchinfo import summary
        model_stats = summary(model, input_shape, verbose=0)
        # Extract multiply-adds from model summary
        multiply_adds = model_stats.total_mult_adds
        # Convert to GMAC/s (assuming 1 second processing time)
        gmac_per_second = multiply_adds / 1e9
        return gmac_per_second
    except ImportError:
        # Fallback estimation based on model parameters
        total_params = sum(p.numel() for p in model.parameters())
        # Rough estimation: 2 operations per parameter
        estimated_gmac = (total_params * 2) / 1e9
        return estimated_gmac

class AdaptiveSampler:
    """
    Adaptive sampling strategy that adjusts sampling parameters based on input characteristics
    """
    
    def __init__(self, base_steps: int = 50, max_steps: int = 100, min_steps: int = 20):
        self.base_steps = base_steps
        self.max_steps = max_steps
        self.min_steps = min_steps
        
    def get_optimal_steps(self, input_tensor: torch.Tensor, snr_estimate: float = None) -> int:
        """
        Determine optimal number of sampling steps based on input characteristics
        
        Args:
            input_tensor: Input audio tensor
            snr_estimate: Estimated SNR of the input
            
        Returns:
            Optimal number of sampling steps
        """
        # Calculate input complexity metrics
        energy = torch.mean(input_tensor ** 2).item()
        dynamic_range = torch.max(input_tensor).item() - torch.min(input_tensor).item()
        
        # Adaptive step calculation
        if snr_estimate is not None and snr_estimate < -10:  # Very noisy
            steps = self.max_steps
        elif energy < 0.01:  # Low energy signal
            steps = max(self.min_steps, int(self.base_steps * 0.7))
        elif dynamic_range > 0.8:  # High dynamic range
            steps = min(self.max_steps, int(self.base_steps * 1.3))
        else:
            steps = self.base_steps
            
        return steps
    
    def get_optimal_snr(self, input_tensor: torch.Tensor) -> float:
        """
        Determine optimal SNR parameter for sampling
        
        Args:
            input_tensor: Input audio tensor
            
        Returns:
            Optimal SNR value
        """
        # Estimate input quality
        energy = torch.mean(input_tensor ** 2).item()
        
        if energy < 0.001:  # Very low energy
            return 0.1
        elif energy > 0.1:  # High energy
            return 0.5
        else:
            return 0.33  # Default value

class EnsembleInference:
    """
    Ensemble inference combining multiple models and sampling strategies
    """
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        self.model_paths = model_paths
        self.weights = weights or [1.0 / len(model_paths)] * len(model_paths)
        self.models = []
        self.adaptive_sampler = AdaptiveSampler()
        
    def load_models(self):
        """Load all ensemble models"""
        try:
            from sgmse.model import ScoreModel
            
            for path in self.model_paths:
                if os.path.exists(path):
                    model = ScoreModel.load_from_checkpoint(path)
                    model.eval()
                    self.models.append(model)
                else:
                    print(f"Warning: Model checkpoint not found: {path}")
        except ImportError as e:
            print(f"Error loading models: {e}")
            print("Please ensure all dependencies are installed")
    
    def enhance_single(self, model, input_tensor: torch.Tensor, sampling_params: Dict) -> torch.Tensor:
        """
        Enhance audio using a single model
        
        Args:
            model: Loaded SGMSE model
            input_tensor: Input audio tensor
            sampling_params: Sampling parameters
            
        Returns:
            Enhanced audio tensor
        """
        with torch.no_grad():
            # Use model's enhance method with adaptive parameters
            enhanced = model.enhance(
                input_tensor,
                sampler_type=sampling_params.get('sampler_type', 'pc'),
                N=sampling_params.get('N', 50),
                snr=sampling_params.get('snr', 0.33),
                corrector=sampling_params.get('corrector', 'ald'),
                corrector_steps=sampling_params.get('corrector_steps', 1)
            )
        return enhanced
    
    def enhance_ensemble(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Enhance audio using ensemble of models
        
        Args:
            input_tensor: Input audio tensor
            
        Returns:
            Enhanced audio tensor from ensemble
        """
        if not self.models:
            raise ValueError("No models loaded for ensemble inference")
        
        # Adaptive sampling parameters
        optimal_steps = self.adaptive_sampler.get_optimal_steps(input_tensor)
        optimal_snr = self.adaptive_sampler.get_optimal_snr(input_tensor)
        
        sampling_params = {
            'sampler_type': 'pc',
            'N': optimal_steps,
            'snr': optimal_snr,
            'corrector': 'ald',
            'corrector_steps': 1
        }
        
        enhanced_outputs = []
        
        # Run inference with each model
        for i, model in enumerate(self.models):
            try:
                enhanced = self.enhance_single(model, input_tensor, sampling_params)
                enhanced_outputs.append(enhanced * self.weights[i])
            except Exception as e:
                print(f"Error with model {i}: {e}")
                continue
        
        if not enhanced_outputs:
            raise RuntimeError("All models failed during inference")
        
        # Weighted ensemble combination
        ensemble_output = torch.sum(torch.stack(enhanced_outputs), dim=0)
        
        return ensemble_output

class KaggleSubmission:
    """
    Main class for Kaggle submission handling
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.ensemble = None
        self.results = []
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "model_paths": ["checkpoints/best_model.ckpt"],
            "ensemble_weights": None,
            "output_format": "kaggle",
            "max_complexity_gmac": 50.0,
            "sampling_strategy": "adaptive",
            "batch_size": 1,
            "device": "auto"
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config
    
    def setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config["device"] == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config["device"])
        
        print(f"Using device: {device}")
        return device
    
    def initialize_ensemble(self):
        """Initialize ensemble inference"""
        self.ensemble = EnsembleInference(
            model_paths=self.config["model_paths"],
            weights=self.config["ensemble_weights"]
        )
        self.ensemble.load_models()
    
    def process_audio_file(self, input_path: str, output_path: str) -> Dict:
        """
        Process a single audio file
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            
        Returns:
            Dictionary with processing results and metrics
        """
        try:
            from torchaudio import load, save
            
            # Load audio
            waveform, sample_rate = load(input_path)
            
            # Ensure correct format (mono, 16kHz if needed)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Track processing time
            start_time = time.time()
            
            # Enhance audio using ensemble
            enhanced = self.ensemble.enhance_ensemble(waveform)
            
            processing_time = time.time() - start_time
            
            # Calculate model complexity
            if self.ensemble.models:
                complexity = calculate_model_complexity(
                    self.ensemble.models[0], 
                    waveform.shape
                )
            else:
                complexity = 0.0
            
            # Save enhanced audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save(output_path, enhanced, sample_rate)
            
            # Calculate basic metrics
            input_energy = torch.mean(waveform ** 2).item()
            output_energy = torch.mean(enhanced ** 2).item()
            energy_ratio = output_energy / (input_energy + 1e-10)
            
            result = {
                "input_file": input_path,
                "output_file": output_path,
                "processing_time": processing_time,
                "model_complexity_gmac": complexity,
                "energy_ratio": energy_ratio,
                "sample_rate": sample_rate,
                "duration": waveform.shape[-1] / sample_rate
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return {
                "input_file": input_path,
                "output_file": output_path,
                "error": str(e)
            }
    
    def process_dataset(self, input_dir: str, output_dir: str) -> pd.DataFrame:
        """
        Process entire dataset
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save enhanced audio files
            
        Returns:
            DataFrame with processing results
        """
        # Find all audio files
        audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).rglob(f'*{ext}'))
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process files
        results = []
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            # Maintain directory structure
            rel_path = audio_file.relative_to(input_dir)
            output_path = Path(output_dir) / rel_path.with_suffix('.wav')
            
            result = self.process_audio_file(str(audio_file), str(output_path))
            results.append(result)
        
        return pd.DataFrame(results)
    
    def calculate_detailed_metrics(self, results_df: pd.DataFrame, 
                                 clean_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate detailed evaluation metrics
        
        Args:
            results_df: DataFrame with processing results
            clean_dir: Directory with clean reference files (if available)
            
        Returns:
            DataFrame with detailed metrics
        """
        if clean_dir is None:
            print("No clean reference directory provided, skipping detailed metrics")
            return results_df
        
        try:
            from calc_metrics import calculate_metrics  # Use existing metrics calculation
            
            # This would integrate with existing calc_metrics.py
            # For now, return the basic results
            print("Detailed metrics calculation would be implemented here")
            return results_df
            
        except ImportError:
            print("Metrics calculation module not available")
            return results_df
    
    def save_kaggle_submission(self, results_df: pd.DataFrame, output_path: str):
        """
        Save results in Kaggle submission format
        
        Args:
            results_df: DataFrame with results
            output_path: Path to save submission CSV
        """
        # Create Kaggle-compatible submission format
        submission_data = []
        
        for _, row in results_df.iterrows():
            if 'error' not in row:
                submission_data.append({
                    'filename': os.path.basename(row['input_file']),
                    'enhanced_file': row['output_file'],
                    'processing_time': row['processing_time'],
                    'model_complexity': row['model_complexity_gmac'],
                    'energy_ratio': row['energy_ratio']
                })
        
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)
        print(f"Kaggle submission saved to: {output_path}")
        
        # Save detailed results as well
        detailed_path = output_path.replace('.csv', '_detailed.csv')
        results_df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to: {detailed_path}")

def main():
    parser = argparse.ArgumentParser(description="Kaggle Dereverberation Submission")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save enhanced audio files")
    parser.add_argument("--clean_dir", type=str, default=None,
                       help="Directory with clean reference files for evaluation")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration JSON file")
    parser.add_argument("--submission_csv", type=str, default="submission.csv",
                       help="Path to save Kaggle submission CSV")
    
    args = parser.parse_args()
    
    # Initialize submission handler
    submission = KaggleSubmission(config_path=args.config)
    
    # Setup device and models
    device = submission.setup_device()
    submission.initialize_ensemble()
    
    # Process dataset
    print("Starting dataset processing...")
    results_df = submission.process_dataset(args.input_dir, args.output_dir)
    
    # Calculate detailed metrics if clean references available
    if args.clean_dir:
        results_df = submission.calculate_detailed_metrics(results_df, args.clean_dir)
    
    # Save Kaggle submission
    submission.save_kaggle_submission(results_df, args.submission_csv)
    
    # Print summary
    successful = len(results_df[~results_df.get('error', pd.Series()).notna()])
    total = len(results_df)
    avg_complexity = results_df['model_complexity_gmac'].mean()
    avg_processing_time = results_df['processing_time'].mean()
    
    print(f"\nProcessing Summary:")
    print(f"Successfully processed: {successful}/{total} files")
    print(f"Average model complexity: {avg_complexity:.2f} GMAC/s")
    print(f"Average processing time: {avg_processing_time:.2f} seconds")
    
    if avg_complexity > 50.0:
        print(f"⚠️  WARNING: Model complexity ({avg_complexity:.2f}) exceeds 50 GMAC/s limit!")
    else:
        print(f"✅ Model complexity within 50 GMAC/s limit")

if __name__ == "__main__":
    main()