#!/usr/bin/env python3
"""
Enhanced SGMSE+ Inference Script for De-reverberation Hackathon
==============================================================

This script implements the complete inference pipeline with ensemble strategies
and test-time augmentation for SOTA performance.

Usage:
    python hackathon_inference.py --input_dir <reverb_audio_dir> --output_dir <enhanced_audio_dir> --models <model_checkpoints>

Author: SOTA Hackathon Team
"""

import os
import sys
import yaml
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import warnings

# Add hackathon solution to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_model import EnhancedScoreModel
from ensemble_inference import EnsembleInference, TestTimeAugmentation
from evaluation_framework import ComprehensiveEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_audio_files(input_dir: str) -> list:
    """Find all audio files in input directory"""
    audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(input_dir).glob(f'**/*{ext}'))
    
    return sorted([str(f) for f in audio_files])


def enhance_single_file(ensemble_system: EnsembleInference,
                       input_file: str,
                       output_file: str,
                       strategy: str = 'uncertainty_weighted',
                       sample_rate: int = 16000) -> dict:
    """Enhance a single audio file"""
    
    try:
        # Load audio
        audio, sr = torchaudio.load(input_file)
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
            sr = sample_rate
        
        # Convert to spectrogram
        spec = torch.stft(
            audio.squeeze(),
            n_fft=510,
            hop_length=256,
            window=torch.hann_window(510),
            return_complex=True
        ).unsqueeze(0)
        
        # Enhance using ensemble
        with torch.no_grad():
            enhanced_spec = ensemble_system.enhance_audio(
                spec.to(ensemble_system.models[0].device),
                strategy=strategy
            )
        
        # Convert back to audio
        enhanced_audio = torch.istft(
            enhanced_spec.squeeze().cpu(),
            n_fft=510,
            hop_length=256,
            window=torch.hann_window(510)
        )
        
        # Save enhanced audio
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torchaudio.save(output_file, enhanced_audio.unsqueeze(0), sr)
        
        return {
            'status': 'success',
            'input_file': input_file,
            'output_file': output_file,
            'duration': enhanced_audio.shape[0] / sr
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'input_file': input_file,
            'output_file': output_file,
            'error': str(e)
        }


def batch_enhance(ensemble_system: EnsembleInference,
                 input_files: list,
                 output_dir: str,
                 strategy: str = 'uncertainty_weighted',
                 show_progress: bool = True) -> dict:
    """Enhance a batch of audio files"""
    
    results = {
        'successful': [],
        'failed': [],
        'total_files': len(input_files),
        'total_duration': 0.0
    }
    
    iterator = tqdm(input_files, desc=f"Enhancing with {strategy}") if show_progress else input_files
    
    for input_file in iterator:
        # Create output filename
        rel_path = os.path.relpath(input_file, os.path.commonpath(input_files))
        output_file = os.path.join(output_dir, f"enhanced_{rel_path}")
        
        # Ensure output file has .wav extension
        output_file = os.path.splitext(output_file)[0] + '.wav'
        
        # Enhance file
        result = enhance_single_file(
            ensemble_system=ensemble_system,
            input_file=input_file,
            output_file=output_file,
            strategy=strategy
        )
        
        if result['status'] == 'success':
            results['successful'].append(result)
            results['total_duration'] += result['duration']
        else:
            results['failed'].append(result)
            print(f"❌ Failed to process {input_file}: {result['error']}")
    
    return results


def evaluate_enhancement(clean_files: list,
                        enhanced_files: list,
                        noisy_files: list = None,
                        output_dir: str = None) -> dict:
    """Evaluate enhancement performance"""
    
    print("📊 Starting comprehensive evaluation...")
    
    evaluator = ComprehensiveEvaluator(
        sample_rate=16000,
        metrics_config={
            'standard': ['pesq', 'stoi', 'si_sdr'],
            'perceptual': ['mel_spectral_loss', 'spectral_convergence'],
            'reverb_specific': ['rt60_estimation', 'drr', 'c50'],
            'computational': True
        }
    )
    
    results = evaluator.evaluate_dataset(
        clean_files=clean_files,
        enhanced_files=enhanced_files,
        noisy_files=noisy_files,
        output_dir=output_dir
    )
    
    return results


def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(description='Enhanced SGMSE+ Inference')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing reverberant audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save enhanced audio files')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints for ensemble')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--strategy', type=str, default='uncertainty_weighted',
                       choices=['simple_average', 'uncertainty_weighted', 'progressive_denoising', 'tta'],
                       help='Ensemble strategy to use')
    parser.add_argument('--clean_dir', type=str, default=None,
                       help='Directory with clean audio for evaluation (optional)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Perform comprehensive evaluation (requires --clean_dir)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"⚠️ Config file {args.config} not found, using defaults")
        config = {'model': {'sr': 16000}}
    
    print("\n" + "="*60)
    print("🎯 ENHANCED SGMSE+ DE-REVERBERATION INFERENCE")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Verify model checkpoints
    print(f"\n🤖 Loading ensemble models:")
    valid_models = []
    for model_path in args.models:
        if os.path.exists(model_path):
            valid_models.append(model_path)
            print(f"  ✅ {os.path.basename(model_path)}")
        else:
            print(f"  ❌ {model_path} not found")
    
    if not valid_models:
        print("❌ No valid model checkpoints found!")
        return
    
    # Find input files
    print(f"\n📁 Searching for audio files in: {args.input_dir}")
    input_files = find_audio_files(args.input_dir)
    print(f"🎵 Found {len(input_files)} audio files")
    
    if not input_files:
        print("❌ No audio files found in input directory!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize ensemble system
    print(f"\n🚀 Initializing ensemble inference system...")
    ensemble_system = EnsembleInference(
        model_paths=valid_models,
        ensemble_strategies=[args.strategy],
        device=str(device),
        enable_tta=True,
        enable_uncertainty=True
    )
    
    print(f"🎯 Using strategy: {args.strategy.upper()}")
    print("🎪 Ensemble features active:")
    print("  ✅ Test-Time Augmentation")
    print("  ✅ Uncertainty Estimation")
    print("  ✅ Multiple Sampling Strategies")
    
    # Process files
    print(f"\n🔄 Processing {len(input_files)} files...")
    results = batch_enhance(
        ensemble_system=ensemble_system,
        input_files=input_files,
        output_dir=args.output_dir,
        strategy=args.strategy,
        show_progress=True
    )
    
    # Print results summary
    print(f"\n📊 Processing Summary:")
    print(f"  ✅ Successful: {len(results['successful'])}")
    print(f"  ❌ Failed: {len(results['failed'])}")
    print(f"  ⏱️ Total duration: {results['total_duration']:.1f} seconds")
    print(f"  📁 Enhanced files saved in: {args.output_dir}")
    
    # Evaluation (if requested)
    if args.evaluate and args.clean_dir:
        print(f"\n📈 Starting evaluation...")
        
        # Find clean files
        clean_files = find_audio_files(args.clean_dir)
        
        # Find corresponding enhanced files
        enhanced_files = []
        for clean_file in clean_files:
            rel_path = os.path.relpath(clean_file, args.clean_dir)
            enhanced_file = os.path.join(args.output_dir, f"enhanced_{rel_path}")
            enhanced_file = os.path.splitext(enhanced_file)[0] + '.wav'
            
            if os.path.exists(enhanced_file):
                enhanced_files.append(enhanced_file)
            else:
                enhanced_files.append(None)
        
        # Filter valid pairs
        valid_pairs = [(c, e) for c, e in zip(clean_files, enhanced_files) if e is not None]
        clean_files_valid = [c for c, e in valid_pairs]
        enhanced_files_valid = [e for c, e in valid_pairs]
        
        if valid_pairs:
            eval_results = evaluate_enhancement(
                clean_files=clean_files_valid,
                enhanced_files=enhanced_files_valid,
                output_dir=os.path.join(args.output_dir, 'evaluation')
            )
            
            print(f"\n🏆 Evaluation Results:")
            if 'summary_statistics' in eval_results:
                for metric, stats in eval_results['summary_statistics'].items():
                    if metric in ['pesq', 'stoi', 'si_sdr']:
                        mean_val = stats['mean']
                        std_val = stats['std']
                        print(f"  {metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")
        else:
            print("⚠️ No matching clean-enhanced file pairs found for evaluation")
    
    print(f"\n🎉 Inference completed!")
    print(f"📁 Enhanced audio files: {args.output_dir}")
    if args.evaluate:
        print(f"📊 Evaluation results: {args.output_dir}/evaluation")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    main()