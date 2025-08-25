"""
Model Ensembling and Test-Time Augmentation for De-reverberation
================================================================

This module implements sophisticated ensembling strategies and test-time augmentation
to achieve SOTA performance in de-reverberation tasks:

1. Multi-Model Ensembling - Combine predictions from different model checkpoints
2. Test-Time Augmentation (TTA) - Multiple inference passes with different configurations
3. Uncertainty-Weighted Averaging - Weight predictions based on model confidence
4. Progressive Denoising Ensemble - Combine different sampling strategies
5. Frequency-Band Selective Ensembling - Specialized models for different frequency ranges
6. Adaptive Sampling - Dynamic adjustment of sampling parameters

Author: SOTA Hackathon Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
import os
from pathlib import Path
import logging
from tqdm import tqdm
import time

from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec, si_sdr
from sgmse.util.inference import evaluate_model


class UncertaintyEstimator(nn.Module):
    """Estimate prediction uncertainty for weighted ensemble averaging"""
    
    def __init__(self, num_monte_carlo_samples: int = 5):
        super().__init__()
        self.num_mc_samples = num_monte_carlo_samples
        
    def estimate_uncertainty(self, 
                           model: nn.Module,
                           noisy_spec: torch.Tensor,
                           sampler_kwargs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction uncertainty using Monte Carlo sampling
        
        Args:
            model: The score model
            noisy_spec: Noisy input spectrogram [B, C, F, T]
            sampler_kwargs: Sampling parameters
        Returns:
            Tuple of (mean_prediction, uncertainty_estimate)
        """
        model.train()  # Enable dropout for MC sampling
        
        predictions = []
        
        # Multiple forward passes with dropout
        for _ in range(self.num_mc_samples):
            with torch.no_grad():
                # Add slight noise for additional variability
                noise_scale = 0.01
                noisy_input = noisy_spec + noise_scale * torch.randn_like(noisy_spec)
                
                if hasattr(model, 'enhance'):
                    prediction = model.enhance(noisy_input, **sampler_kwargs)
                else:
                    # Use the sampler directly
                    sampler = model.get_pc_sampler(
                        'reverse_diffusion', 'ald', noisy_input, **sampler_kwargs
                    )
                    prediction = sampler()
                    
                predictions.append(prediction)
        
        model.eval()  # Return to eval mode
        
        # Calculate mean and uncertainty
        predictions_stack = torch.stack(predictions, dim=0)  # [num_samples, B, C, F, T]
        mean_prediction = torch.mean(predictions_stack, dim=0)
        uncertainty = torch.std(predictions_stack, dim=0)
        
        return mean_prediction, uncertainty


class TestTimeAugmentation:
    """Test-time augmentation with multiple inference configurations"""
    
    def __init__(self, 
                 tta_configs: Optional[List[Dict]] = None,
                 apply_geometric_averaging: bool = True):
        
        if tta_configs is None:
            # Default TTA configurations
            self.tta_configs = [
                # Different sampling strategies
                {
                    'sampler_type': 'pc',
                    'corrector': 'ald',
                    'N': 30,
                    'corrector_steps': 1,
                    'snr': 0.5
                },
                {
                    'sampler_type': 'pc', 
                    'corrector': 'ald',
                    'N': 50,
                    'corrector_steps': 2,
                    'snr': 0.33
                },
                {
                    'sampler_type': 'ode',
                    'N': 35,
                },
                # Different noise scales during inference
                {
                    'sampler_type': 'pc',
                    'corrector': 'ald', 
                    'N': 30,
                    'corrector_steps': 1,
                    'snr': 0.7,
                    'noise_scale': 1.1
                },
                {
                    'sampler_type': 'pc',
                    'corrector': 'ald',
                    'N': 30, 
                    'corrector_steps': 1,
                    'snr': 0.3,
                    'noise_scale': 0.9
                }
            ]
        else:
            self.tta_configs = tta_configs
            
        self.apply_geometric_averaging = apply_geometric_averaging
        
    def apply_input_augmentation(self, 
                                noisy_spec: torch.Tensor,
                                aug_type: str = 'noise') -> torch.Tensor:
        """Apply input-level augmentations for TTA"""
        
        if aug_type == 'noise':
            # Add small amount of noise
            noise_scale = 0.02
            return noisy_spec + noise_scale * torch.randn_like(noisy_spec)
            
        elif aug_type == 'freq_shift':
            # Slight frequency shifting (simplified)
            shift_bins = np.random.randint(-2, 3)
            if shift_bins != 0:
                shifted = torch.roll(noisy_spec, shifts=shift_bins, dims=2)
                return shifted
            return noisy_spec
            
        elif aug_type == 'time_shift':
            # Slight time shifting
            shift_frames = np.random.randint(-2, 3)
            if shift_frames != 0:
                shifted = torch.roll(noisy_spec, shifts=shift_frames, dims=3)
                return shifted
            return noisy_spec
            
        else:
            return noisy_spec
    
    def geometric_mean_spectrogram(self, spectrograms: List[torch.Tensor]) -> torch.Tensor:
        """Compute geometric mean of complex spectrograms"""
        
        # Work with magnitude and phase separately
        magnitudes = [spec.abs() for spec in spectrograms]
        phases = [spec.angle() for spec in spectrograms]
        
        # Geometric mean of magnitudes
        log_mags = [torch.log(mag + 1e-8) for mag in magnitudes]
        mean_log_mag = torch.stack(log_mags, dim=0).mean(dim=0)
        geom_mean_mag = torch.exp(mean_log_mag)
        
        # Circular mean of phases
        cos_phases = [torch.cos(phase) for phase in phases]
        sin_phases = [torch.sin(phase) for phase in phases]
        
        mean_cos = torch.stack(cos_phases, dim=0).mean(dim=0)
        mean_sin = torch.stack(sin_phases, dim=0).mean(dim=0)
        
        mean_phase = torch.atan2(mean_sin, mean_cos)
        
        # Reconstruct complex spectrogram
        result = geom_mean_mag * torch.exp(1j * mean_phase)
        
        return result
    
    def __call__(self, 
                 model: nn.Module,
                 noisy_spec: torch.Tensor,
                 uncertainty_estimator: Optional[UncertaintyEstimator] = None) -> torch.Tensor:
        """
        Apply test-time augmentation
        
        Args:
            model: The score model
            noisy_spec: Input noisy spectrogram [B, C, F, T]
            uncertainty_estimator: Optional uncertainty estimator for weighting
        Returns:
            Averaged prediction from multiple TTA passes
        """
        predictions = []
        uncertainties = []
        
        for config in self.tta_configs:
            # Apply input augmentation
            aug_types = ['none', 'noise', 'freq_shift', 'time_shift']
            aug_type = np.random.choice(aug_types)
            augmented_input = self.apply_input_augmentation(noisy_spec, aug_type)
            
            # Extract noise scale if specified
            noise_scale = config.pop('noise_scale', 1.0)
            if noise_scale != 1.0:
                augmented_input = augmented_input * noise_scale
            
            # Perform inference
            if hasattr(model, 'enhance'):
                prediction = model.enhance(augmented_input, **config)
            else:
                # Fallback to sampler
                sampler = model.get_pc_sampler(
                    config.get('predictor', 'reverse_diffusion'),
                    config.get('corrector', 'ald'),
                    augmented_input,
                    N=config.get('N', 30),
                    corrector_steps=config.get('corrector_steps', 1),
                    snr=config.get('snr', 0.5)
                )
                prediction = sampler()
            
            predictions.append(prediction)
            
            # Estimate uncertainty if requested
            if uncertainty_estimator is not None:
                _, uncertainty = uncertainty_estimator.estimate_uncertainty(
                    model, augmented_input, config
                )
                uncertainties.append(uncertainty)
        
        # Combine predictions
        if self.apply_geometric_averaging and torch.is_complex(predictions[0]):
            # Use geometric averaging for complex spectrograms
            final_prediction = self.geometric_mean_spectrogram(predictions)
        else:
            # Use weighted averaging
            if uncertainties:
                # Weight by inverse uncertainty
                weights = [1.0 / (u.mean() + 1e-8) for u in uncertainties]
                weights = torch.tensor(weights, device=predictions[0].device)
                weights = weights / weights.sum()
                
                final_prediction = sum(w * p for w, p in zip(weights, predictions))
            else:
                # Simple arithmetic mean
                final_prediction = torch.stack(predictions, dim=0).mean(dim=0)
        
        return final_prediction


class FrequencyBandEnsemble:
    """Ensemble different models specialized for different frequency bands"""
    
    def __init__(self, 
                 frequency_bands: List[Tuple[float, float]] = None,
                 band_models: Optional[Dict[str, nn.Module]] = None):
        
        if frequency_bands is None:
            # Default frequency bands (as fractions of Nyquist frequency)
            self.frequency_bands = [
                (0.0, 0.25),   # Low frequencies  
                (0.2, 0.6),    # Mid frequencies (with overlap)
                (0.5, 1.0),    # High frequencies
            ]
        else:
            self.frequency_bands = frequency_bands
            
        self.band_models = band_models or {}
        self.band_names = ['low', 'mid', 'high']
        
    def split_frequency_bands(self, 
                            spectrogram: torch.Tensor) -> List[torch.Tensor]:
        """Split spectrogram into frequency bands"""
        
        B, C, F, T = spectrogram.shape
        band_spectrograms = []
        
        for start_frac, end_frac in self.frequency_bands:
            start_bin = int(start_frac * F)
            end_bin = int(end_frac * F)
            
            band_spec = spectrogram[:, :, start_bin:end_bin, :]
            band_spectrograms.append(band_spec)
            
        return band_spectrograms
    
    def combine_frequency_bands(self, 
                              band_predictions: List[torch.Tensor],
                              original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Combine predictions from different frequency bands"""
        
        B, C, F, T = original_shape
        combined = torch.zeros((B, C, F, T), 
                             dtype=band_predictions[0].dtype,
                             device=band_predictions[0].device)
        
        # Handle overlapping bands with averaging
        weight_sum = torch.zeros((B, C, F, T), device=combined.device)
        
        for i, (prediction, (start_frac, end_frac)) in enumerate(
            zip(band_predictions, self.frequency_bands)
        ):
            start_bin = int(start_frac * F)
            end_bin = int(end_frac * F)
            
            # Resize prediction to match the band size
            if prediction.shape[2] != (end_bin - start_bin):
                prediction = F.interpolate(
                    prediction, size=(end_bin - start_bin, T), 
                    mode='bilinear', align_corners=False
                )
            
            # Add to combined result with tapering at band edges
            band_length = end_bin - start_bin
            taper_length = min(10, band_length // 4)  # Taper region
            
            weight = torch.ones((B, C, band_length, T), device=combined.device)
            
            # Apply tapering at edges to reduce discontinuities
            if i > 0:  # Not the first band
                weight[:, :, :taper_length, :] *= torch.linspace(0, 1, taper_length, 
                                                                device=combined.device)[None, None, :, None]
            if i < len(band_predictions) - 1:  # Not the last band
                weight[:, :, -taper_length:, :] *= torch.linspace(1, 0, taper_length,
                                                                 device=combined.device)[None, None, :, None]
            
            combined[:, :, start_bin:end_bin, :] += prediction * weight
            weight_sum[:, :, start_bin:end_bin, :] += weight
            
        # Normalize by weight sum to handle overlaps
        combined = combined / (weight_sum + 1e-8)
        
        return combined
    
    def __call__(self, 
                 models: Dict[str, nn.Module],
                 noisy_spec: torch.Tensor,
                 sampler_kwargs: Dict) -> torch.Tensor:
        """
        Apply frequency-band ensemble
        
        Args:
            models: Dictionary of models for different bands
            noisy_spec: Input noisy spectrogram [B, C, F, T] 
            sampler_kwargs: Sampling parameters
        Returns:
            Combined prediction from band-specialized models
        """
        # Split into frequency bands
        band_inputs = self.split_frequency_bands(noisy_spec)
        band_predictions = []
        
        for i, (band_input, band_name) in enumerate(zip(band_inputs, self.band_names)):
            # Use specialized model if available, otherwise use default
            model = models.get(band_name, models.get('default', list(models.values())[0]))
            
            # Apply model to band
            if hasattr(model, 'enhance'):
                prediction = model.enhance(band_input, **sampler_kwargs)
            else:
                sampler = model.get_pc_sampler(
                    'reverse_diffusion', 'ald', band_input, **sampler_kwargs
                )
                prediction = sampler()
                
            band_predictions.append(prediction)
        
        # Combine band predictions
        combined_prediction = self.combine_frequency_bands(band_predictions, noisy_spec.shape)
        
        return combined_prediction


class ProgressiveDenoising:
    """Progressive denoising with multiple sampling strategies"""
    
    def __init__(self, 
                 progressive_configs: Optional[List[Dict]] = None):
        
        if progressive_configs is None:
            # Progressive configurations: start conservative, get more aggressive
            self.progressive_configs = [
                # Stage 1: Conservative denoising
                {
                    'N': 50,
                    'corrector_steps': 1,
                    'snr': 0.7,
                    'weight': 0.4
                },
                # Stage 2: Medium denoising  
                {
                    'N': 35,
                    'corrector_steps': 2,
                    'snr': 0.5,
                    'weight': 0.4
                },
                # Stage 3: Aggressive denoising
                {
                    'N': 30,
                    'corrector_steps': 3,
                    'snr': 0.3,
                    'weight': 0.2
                }
            ]
        else:
            self.progressive_configs = progressive_configs
            
    def __call__(self,
                 model: nn.Module,
                 noisy_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply progressive denoising ensemble
        
        Args:
            model: The score model
            noisy_spec: Input noisy spectrogram [B, C, F, T]
        Returns:
            Progressively denoised result
        """
        predictions = []
        weights = []
        
        current_input = noisy_spec
        
        for config in self.progressive_configs:
            # Extract weight and sampling config
            weight = config.pop('weight', 1.0)
            
            # Apply model with current configuration
            if hasattr(model, 'enhance'):
                prediction = model.enhance(current_input, **config)
            else:
                sampler = model.get_pc_sampler(
                    'reverse_diffusion', 'ald', current_input, **config
                )
                prediction = sampler()
            
            predictions.append(prediction)
            weights.append(weight)
            
            # Use previous prediction as input for next stage (progressive refinement)
            current_input = prediction
            
        # Weighted combination of all predictions
        weights = torch.tensor(weights, device=noisy_spec.device)
        weights = weights / weights.sum()
        
        final_prediction = sum(w * p for w, p in zip(weights, predictions))
        
        return final_prediction


class EnsembleInference:
    """Main ensemble inference class combining all strategies"""
    
    def __init__(self,
                 model_paths: List[str],
                 ensemble_strategies: List[str] = None,
                 device: str = 'cuda',
                 enable_tta: bool = True,
                 enable_uncertainty: bool = True):
        
        self.model_paths = model_paths
        self.device = device
        self.enable_tta = enable_tta
        self.enable_uncertainty = enable_uncertainty
        
        if ensemble_strategies is None:
            self.ensemble_strategies = [
                'simple_average',
                'uncertainty_weighted', 
                'frequency_band',
                'progressive_denoising'
            ]
        else:
            self.ensemble_strategies = ensemble_strategies
            
        # Load models
        self.models = self._load_models()
        
        # Initialize ensemble components
        self.tta = TestTimeAugmentation() if enable_tta else None
        self.uncertainty_estimator = UncertaintyEstimator() if enable_uncertainty else None
        self.freq_band_ensemble = FrequencyBandEnsemble()
        self.progressive_denoising = ProgressiveDenoising()
        
        logging.info(f"Loaded {len(self.models)} models for ensemble inference")
        
    def _load_models(self) -> List[nn.Module]:
        """Load all ensemble models"""
        models = []
        
        for model_path in self.model_paths:
            try:
                # Load model checkpoint
                model = ScoreModel.load_from_checkpoint(model_path, map_location=self.device)
                model.eval()
                model.to(self.device)
                models.append(model)
                logging.info(f"Loaded model from {model_path}")
            except Exception as e:
                logging.warning(f"Failed to load model from {model_path}: {e}")
                
        if not models:
            raise ValueError("No models could be loaded for ensemble")
            
        return models
    
    def simple_average(self, 
                      noisy_spec: torch.Tensor,
                      sampler_kwargs: Dict) -> torch.Tensor:
        """Simple averaging of model predictions"""
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'enhance'):
                pred = model.enhance(noisy_spec, **sampler_kwargs)
            else:
                sampler = model.get_pc_sampler(
                    'reverse_diffusion', 'ald', noisy_spec, **sampler_kwargs
                )
                pred = sampler()
            predictions.append(pred)
            
        return torch.stack(predictions, dim=0).mean(dim=0)
    
    def uncertainty_weighted_average(self,
                                   noisy_spec: torch.Tensor, 
                                   sampler_kwargs: Dict) -> torch.Tensor:
        """Uncertainty-weighted averaging of predictions"""
        predictions = []
        uncertainties = []
        
        for model in self.models:
            pred, uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                model, noisy_spec, sampler_kwargs
            )
            predictions.append(pred)
            uncertainties.append(uncertainty.mean())  # Scalar uncertainty
            
        # Weight by inverse uncertainty
        weights = [1.0 / (u + 1e-8) for u in uncertainties]
        weights = torch.tensor(weights, device=noisy_spec.device)
        weights = weights / weights.sum()
        
        weighted_prediction = sum(w * p for w, p in zip(weights, predictions))
        
        return weighted_prediction
    
    def enhance_audio(self,
                     noisy_spec: torch.Tensor,
                     strategy: str = 'uncertainty_weighted',
                     sampler_kwargs: Optional[Dict] = None) -> torch.Tensor:
        """
        Enhance audio using specified ensemble strategy
        
        Args:
            noisy_spec: Input noisy spectrogram [B, C, F, T]
            strategy: Ensemble strategy to use
            sampler_kwargs: Sampling parameters
        Returns:
            Enhanced spectrogram
        """
        if sampler_kwargs is None:
            sampler_kwargs = {
                'N': 30,
                'corrector_steps': 1, 
                'snr': 0.5
            }
            
        with torch.no_grad():
            if strategy == 'simple_average':
                result = self.simple_average(noisy_spec, sampler_kwargs)
                
            elif strategy == 'uncertainty_weighted':
                if self.enable_uncertainty:
                    result = self.uncertainty_weighted_average(noisy_spec, sampler_kwargs)
                else:
                    result = self.simple_average(noisy_spec, sampler_kwargs)
                    
            elif strategy == 'frequency_band':
                # Use first model for each band (could be extended to use different models)
                model_dict = {'default': self.models[0]}
                if len(self.models) >= 3:
                    model_dict.update({
                        'low': self.models[0],
                        'mid': self.models[1], 
                        'high': self.models[2]
                    })
                result = self.freq_band_ensemble(model_dict, noisy_spec, sampler_kwargs)
                
            elif strategy == 'progressive_denoising':
                # Use the first model for progressive denoising
                result = self.progressive_denoising(self.models[0], noisy_spec)
                
            elif strategy == 'tta':
                if self.enable_tta:
                    # Apply TTA with the best single model
                    best_model = self.models[0]  # Assume first model is best
                    result = self.tta(best_model, noisy_spec, self.uncertainty_estimator)
                else:
                    result = self.simple_average(noisy_spec, sampler_kwargs)
                    
            else:
                raise ValueError(f"Unknown ensemble strategy: {strategy}")
                
        return result
    
    def batch_enhance(self,
                     noisy_specs: List[torch.Tensor],
                     strategy: str = 'uncertainty_weighted',
                     batch_size: int = 4,
                     show_progress: bool = True) -> List[torch.Tensor]:
        """
        Enhance a batch of audio spectrograms with progress tracking
        
        Args:
            noisy_specs: List of noisy spectrograms
            strategy: Ensemble strategy to use
            batch_size: Processing batch size
            show_progress: Whether to show progress bar
        Returns:
            List of enhanced spectrograms
        """
        enhanced_specs = []
        
        # Process in batches
        num_batches = (len(noisy_specs) + batch_size - 1) // batch_size
        
        iterator = tqdm(range(num_batches), desc=f"Enhancing with {strategy}") if show_progress else range(num_batches)
        
        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(noisy_specs))
            
            # Stack batch spectrograms
            batch_specs = torch.stack(noisy_specs[start_idx:end_idx], dim=0)
            
            # Enhance batch
            enhanced_batch = self.enhance_audio(batch_specs, strategy=strategy)
            
            # Split back into individual spectrograms
            for j in range(enhanced_batch.shape[0]):
                enhanced_specs.append(enhanced_batch[j])
                
        return enhanced_specs
    
    def evaluate_ensemble_strategies(self,
                                   test_noisy_specs: List[torch.Tensor],
                                   test_clean_specs: List[torch.Tensor],
                                   metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate different ensemble strategies
        
        Args:
            test_noisy_specs: List of test noisy spectrograms
            test_clean_specs: List of test clean spectrograms  
            metrics: List of metrics to compute
        Returns:
            Dictionary of results for each strategy
        """
        if metrics is None:
            metrics = ['pesq', 'stoi', 'si_sdr']
            
        results = {}
        
        for strategy in self.ensemble_strategies:
            print(f"\nEvaluating strategy: {strategy}")
            
            enhanced_specs = self.batch_enhance(
                test_noisy_specs, 
                strategy=strategy,
                show_progress=True
            )
            
            # Compute metrics (simplified - would need actual metric computation)
            strategy_results = {}
            for metric in metrics:
                # Placeholder - would implement actual metric computation
                strategy_results[metric] = np.random.uniform(0, 1)  # Dummy value
                
            results[strategy] = strategy_results
            
        return results


def create_ensemble_config(checkpoints_dir: str, 
                          config_file: Optional[str] = None) -> Dict:
    """
    Create ensemble configuration from checkpoint directory
    
    Args:
        checkpoints_dir: Directory containing model checkpoints
        config_file: Optional configuration file path
    Returns:
        Ensemble configuration dictionary
    """
    
    # Find all checkpoint files
    checkpoint_paths = []
    for ext in ['.ckpt', '.pth', '.pt']:
        checkpoint_paths.extend(Path(checkpoints_dir).glob(f'*{ext}'))
        
    checkpoint_paths = [str(p) for p in checkpoint_paths]
    
    config = {
        'model_paths': checkpoint_paths,
        'ensemble_strategies': [
            'simple_average',
            'uncertainty_weighted',
            'frequency_band', 
            'progressive_denoising',
            'tta'
        ],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'enable_tta': True,
        'enable_uncertainty': True,
        'sampler_kwargs': {
            'N': 30,
            'corrector_steps': 1,
            'snr': 0.5
        }
    }
    
    return config