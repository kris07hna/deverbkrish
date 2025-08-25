"""
Advanced Data Augmentation for De-reverberation Hackathon
=========================================================

This module implements sophisticated data augmentation techniques specifically designed
for improving de-reverberation performance:

1. Spectral Augmentation - Time and frequency masking with reverb-aware strategies
2. Dynamic Range Compression/Expansion - Simulate different acoustic conditions
3. Room Impulse Response (RIR) Simulation - Generate diverse reverberant conditions  
4. Multi-resolution Processing - Data augmentation at different time scales
5. Adaptive Noise Addition - Context-aware noise injection
6. Spectral Morphing - Interpolation between different acoustic signatures

Author: SOTA Hackathon Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import random
from typing import Tuple, Optional, List, Dict, Union
import math
from scipy import signal
from scipy.signal import fftconvolve
import librosa


class SpectralAugmentation(nn.Module):
    """Advanced spectral augmentation with reverb-aware masking strategies"""
    
    def __init__(self, 
                 freq_mask_param: int = 40,
                 time_mask_param: int = 40,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2,
                 mask_prob: float = 0.5,
                 reverb_aware: bool = True):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_prob = mask_prob
        self.reverb_aware = reverb_aware
        
        # Frequency bands more susceptible to reverberation artifacts
        self.critical_freq_bands = [
            (0, 0.1),      # Low frequencies - room modes
            (0.3, 0.7),    # Mid frequencies - early reflections
            (0.8, 1.0),    # High frequencies - late reverberation
        ]
        
    def forward(self, spectrogram: torch.Tensor, 
                is_reverberant: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply spectral augmentation with optional reverb-aware masking
        
        Args:
            spectrogram: Input spectrogram [B, C, F, T]
            is_reverberant: Boolean tensor indicating reverberant samples [B]
        Returns:
            Augmented spectrogram
        """
        if random.random() > self.mask_prob:
            return spectrogram
            
        B, C, F, T = spectrogram.shape
        augmented = spectrogram.clone()
        
        for b in range(B):
            # Apply frequency masking
            for _ in range(self.num_freq_masks):
                if self.reverb_aware and is_reverberant is not None and is_reverberant[b]:
                    # Focus masking on critical frequency bands for reverberant signals
                    band_start, band_end = random.choice(self.critical_freq_bands)
                    f_start = int(band_start * F)
                    f_end = int(band_end * F)
                    mask_width = min(self.freq_mask_param, f_end - f_start)
                    if mask_width > 0:
                        mask_start = random.randint(f_start, max(f_start, f_end - mask_width))
                        mask_end = mask_start + mask_width
                else:
                    # Standard frequency masking
                    mask_width = random.randint(0, self.freq_mask_param)
                    if mask_width > 0:
                        mask_start = random.randint(0, F - mask_width)
                        mask_end = mask_start + mask_width
                
                if mask_width > 0:
                    augmented[b, :, mask_start:mask_end, :] = 0
            
            # Apply time masking with reverb considerations
            for _ in range(self.num_time_masks):
                mask_width = random.randint(0, self.time_mask_param)
                if mask_width > 0:
                    if self.reverb_aware and is_reverberant is not None and is_reverberant[b]:
                        # Avoid masking the beginning for reverberant signals (preserve direct sound)
                        safe_start = min(T // 4, 10)  # Preserve first 25% or 10 frames
                        if T - safe_start > mask_width:
                            mask_start = random.randint(safe_start, T - mask_width)
                            mask_end = mask_start + mask_width
                            augmented[b, :, :, mask_start:mask_end] = 0
                    else:
                        mask_start = random.randint(0, T - mask_width)
                        mask_end = mask_start + mask_width
                        augmented[b, :, :, mask_start:mask_end] = 0
        
        return augmented


class DynamicRangeAugmentation(nn.Module):
    """Dynamic range compression/expansion to simulate different acoustic conditions"""
    
    def __init__(self, 
                 compression_range: Tuple[float, float] = (0.5, 2.0),
                 expansion_range: Tuple[float, float] = (1.0, 3.0),
                 apply_prob: float = 0.3):
        super().__init__()
        self.compression_range = compression_range
        self.expansion_range = expansion_range
        self.apply_prob = apply_prob
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic range modification
        
        Args:
            spectrogram: Input complex spectrogram [B, C, F, T]
        Returns:
            Modified spectrogram with altered dynamic range
        """
        if random.random() > self.apply_prob:
            return spectrogram
            
        # Work with magnitude spectrogram
        magnitude = spectrogram.abs()
        phase = spectrogram.angle()
        
        # Apply logarithmic compression/expansion
        if random.random() < 0.5:
            # Compression (reduce dynamic range)
            factor = random.uniform(*self.compression_range)
            modified_mag = magnitude.pow(factor)
        else:
            # Expansion (increase dynamic range)  
            factor = random.uniform(*self.expansion_range)
            # Use log domain for expansion to avoid extreme values
            log_mag = torch.log(magnitude + 1e-8)
            modified_log_mag = log_mag * factor
            modified_mag = torch.exp(modified_log_mag)
            
        # Reconstruct complex spectrogram
        modified_spectrogram = modified_mag * torch.exp(1j * phase)
        
        return modified_spectrogram


class RIRSimulation(nn.Module):
    """Room Impulse Response simulation for diverse reverberant conditions"""
    
    def __init__(self, 
                 room_sizes: List[Tuple[float, float, float]] = None,
                 rt60_range: Tuple[float, float] = (0.2, 1.5),
                 absorption_range: Tuple[float, float] = (0.1, 0.9),
                 apply_prob: float = 0.4):
        super().__init__()
        
        if room_sizes is None:
            # Default room configurations (length, width, height in meters)
            self.room_sizes = [
                (3, 3, 2.5),    # Small room
                (5, 4, 3),      # Medium room  
                (8, 6, 3.5),    # Large room
                (12, 10, 4),    # Very large room
                (20, 15, 5),    # Hall
            ]
        else:
            self.room_sizes = room_sizes
            
        self.rt60_range = rt60_range
        self.absorption_range = absorption_range
        self.apply_prob = apply_prob
        
    def generate_rir(self, 
                     room_size: Tuple[float, float, float],
                     rt60: float,
                     sample_rate: int = 16000,
                     rir_length: float = 1.0) -> torch.Tensor:
        """
        Generate a synthetic room impulse response
        
        Args:
            room_size: Room dimensions (L, W, H) in meters
            rt60: Reverberation time in seconds
            sample_rate: Audio sample rate
            rir_length: Length of RIR in seconds
        Returns:
            Generated RIR as tensor [samples]
        """
        L, W, H = room_size
        room_volume = L * W * H
        
        # Calculate absorption coefficient from RT60
        # Sabine's formula: RT60 = 0.161 * V / A
        # where A = α * S (total absorption)
        surface_area = 2 * (L*W + L*H + W*H)
        alpha = 0.161 * room_volume / (rt60 * surface_area)
        alpha = np.clip(alpha, 0.01, 0.99)
        
        # Generate early reflections using image source method (simplified)
        num_samples = int(rir_length * sample_rate)
        rir = torch.zeros(num_samples)
        
        # Direct path
        rir[0] = 1.0
        
        # Early reflections (simplified model)
        max_order = 3  # Reflection order
        reflection_times = []
        reflection_gains = []
        
        for nx in range(-max_order, max_order + 1):
            for ny in range(-max_order, max_order + 1):
                for nz in range(-max_order, max_order + 1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue  # Skip direct path
                        
                    # Calculate reflection time (simplified)
                    distance = np.sqrt((nx * L)**2 + (ny * W)**2 + (nz * H)**2)
                    time_samples = int(distance / 343 * sample_rate)  # Speed of sound = 343 m/s
                    
                    if time_samples < num_samples:
                        # Calculate gain with absorption
                        num_reflections = abs(nx) + abs(ny) + abs(nz)
                        gain = (1 - alpha) ** num_reflections
                        
                        reflection_times.append(time_samples)
                        reflection_gains.append(gain)
        
        # Add early reflections
        for time_idx, gain in zip(reflection_times, reflection_gains):
            if time_idx < num_samples:
                rir[time_idx] += gain
                
        # Add late reverberation (exponential decay with noise)
        late_start = min(len(reflection_times) * 2, num_samples // 4)
        if late_start < num_samples:
            # Generate decaying noise for late reverberation
            decay_rate = -6.9 / (rt60 * sample_rate)  # -60dB decay
            time_indices = torch.arange(late_start, num_samples, dtype=torch.float32)
            decay_envelope = torch.exp(decay_rate * time_indices)
            
            # Add random noise component
            late_reverb = torch.randn(num_samples - late_start) * decay_envelope
            rir[late_start:] += late_reverb * 0.1  # Scale factor for late reverb
            
        return rir
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply RIR simulation to clean audio
        
        Args:
            audio: Clean audio tensor [B, samples] or [B, C, samples]
        Returns:
            Reverberant audio
        """
        if random.random() > self.apply_prob:
            return audio
            
        if len(audio.shape) == 2:
            B, samples = audio.shape
            channels = 1
            audio = audio.unsqueeze(1)  # Add channel dimension
        else:
            B, channels, samples = audio.shape
            
        reverberant_audio = torch.zeros_like(audio)
        
        for b in range(B):
            # Random room configuration
            room_size = random.choice(self.room_sizes)
            rt60 = random.uniform(*self.rt60_range)
            
            # Generate RIR
            rir = self.generate_rir(room_size, rt60, sample_rate=16000)
            
            # Apply convolution for each channel
            for c in range(channels):
                # Convolve with RIR (using torchaudio's implementation would be better in practice)
                audio_np = audio[b, c].detach().cpu().numpy()
                rir_np = rir.detach().cpu().numpy()
                
                # Convolution
                reverberant_np = fftconvolve(audio_np, rir_np, mode='same')
                reverberant_audio[b, c] = torch.from_numpy(reverberant_np).to(audio.device)
                
        if len(audio.shape) == 2:
            reverberant_audio = reverberant_audio.squeeze(1)  # Remove channel dimension
            
        return reverberant_audio


class AdaptiveNoiseAugmentation(nn.Module):
    """Context-aware noise injection based on signal characteristics"""
    
    def __init__(self,
                 noise_types: List[str] = None,
                 snr_range: Tuple[float, float] = (10, 30),
                 apply_prob: float = 0.3):
        super().__init__()
        
        if noise_types is None:
            self.noise_types = ['white', 'pink', 'brown', 'babble', 'ambient']
        else:
            self.noise_types = noise_types
            
        self.snr_range = snr_range
        self.apply_prob = apply_prob
        
    def generate_noise(self, 
                       noise_type: str, 
                       shape: Tuple[int, ...], 
                       device: torch.device) -> torch.Tensor:
        """Generate different types of noise"""
        
        if noise_type == 'white':
            return torch.randn(shape, device=device)
            
        elif noise_type == 'pink':
            # Pink noise (1/f noise)
            white_noise = torch.randn(shape, device=device)
            # Apply 1/f filter in frequency domain (simplified)
            return white_noise * 0.7  # Placeholder - proper pink noise is more complex
            
        elif noise_type == 'brown':
            # Brown noise (1/f^2)
            white_noise = torch.randn(shape, device=device)
            return white_noise * 0.5  # Placeholder
            
        elif noise_type == 'babble':
            # Simulated babble noise (multiple sinusoids)
            t = torch.linspace(0, 1, shape[-1], device=device)
            noise = torch.zeros(shape, device=device)
            for freq in [100, 200, 300, 500, 800]:  # Typical speech formants
                noise += torch.sin(2 * np.pi * freq * t) * random.uniform(0.1, 0.3)
            return noise
            
        elif noise_type == 'ambient':
            # Ambient noise (low-frequency emphasis)
            white_noise = torch.randn(shape, device=device)
            # Simple low-pass filtering
            return white_noise * 0.6
            
        else:
            return torch.randn(shape, device=device)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive noise based on audio characteristics
        
        Args:
            audio: Input audio tensor [B, samples] or [B, C, samples]
        Returns:
            Noisy audio
        """
        if random.random() > self.apply_prob:
            return audio
            
        # Analyze audio characteristics
        rms_level = torch.sqrt(torch.mean(audio**2, dim=-1, keepdim=True))
        
        # Choose noise type based on audio characteristics
        if rms_level.mean() > 0.1:  # High energy signal
            noise_type = random.choice(['white', 'ambient'])
        else:  # Low energy signal
            noise_type = random.choice(['pink', 'brown', 'babble'])
            
        # Generate noise
        noise = self.generate_noise(noise_type, audio.shape, audio.device)
        
        # Adaptive SNR based on signal level
        base_snr = random.uniform(*self.snr_range)
        # Adjust SNR based on signal characteristics
        adaptive_snr = base_snr + 5 * torch.log10(rms_level.mean() + 1e-8)
        adaptive_snr = torch.clamp(adaptive_snr, self.snr_range[0], self.snr_range[1] + 10)
        
        # Calculate noise scaling
        signal_power = torch.mean(audio**2)
        noise_power = torch.mean(noise**2)
        snr_linear = 10**(adaptive_snr / 10)
        noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        
        # Apply noise
        noisy_audio = audio + noise_scale * noise
        
        return noisy_audio


class SpectralMorphing(nn.Module):
    """Spectral interpolation between different acoustic signatures"""
    
    def __init__(self, morph_prob: float = 0.2, alpha_range: Tuple[float, float] = (0.1, 0.5)):
        super().__init__()
        self.morph_prob = morph_prob
        self.alpha_range = alpha_range
        
    def forward(self, 
                spec1: torch.Tensor, 
                spec2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform spectral morphing between two spectrograms
        
        Args:
            spec1: First spectrogram [B, C, F, T]
            spec2: Second spectrogram [B, C, F, T] 
        Returns:
            Tuple of morphed spectrograms
        """
        if random.random() > self.morph_prob:
            return spec1, spec2
            
        alpha = random.uniform(*self.alpha_range)
        
        # Morph in magnitude and phase separately
        mag1, phase1 = spec1.abs(), spec1.angle()
        mag2, phase2 = spec2.abs(), spec2.angle()
        
        # Magnitude morphing (linear interpolation)
        morphed_mag1 = (1 - alpha) * mag1 + alpha * mag2
        morphed_mag2 = alpha * mag1 + (1 - alpha) * mag2
        
        # Phase morphing (circular interpolation)
        # Handle phase wrapping
        phase_diff = phase2 - phase1
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        morphed_phase1 = phase1 + alpha * phase_diff
        morphed_phase2 = phase2 - alpha * phase_diff
        
        # Reconstruct complex spectrograms
        morphed_spec1 = morphed_mag1 * torch.exp(1j * morphed_phase1)
        morphed_spec2 = morphed_mag2 * torch.exp(1j * morphed_phase2)
        
        return morphed_spec1, morphed_spec2


class AdvancedDataAugmentation(nn.Module):
    """Combined advanced data augmentation pipeline"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 training_mode: bool = True):
        super().__init__()
        
        # Default configuration
        default_config = {
            'spectral_aug': {
                'freq_mask_param': 40,
                'time_mask_param': 40,
                'num_freq_masks': 2,
                'num_time_masks': 2,
                'mask_prob': 0.5,
                'reverb_aware': True
            },
            'dynamic_range': {
                'compression_range': (0.5, 2.0),
                'expansion_range': (1.0, 3.0),
                'apply_prob': 0.3
            },
            'rir_simulation': {
                'rt60_range': (0.2, 1.5),
                'apply_prob': 0.4
            },
            'noise_aug': {
                'snr_range': (10, 30),
                'apply_prob': 0.3
            },
            'spectral_morph': {
                'morph_prob': 0.2,
                'alpha_range': (0.1, 0.5)
            }
        }
        
        self.config = config if config is not None else default_config
        self.training_mode = training_mode
        
        # Initialize augmentation modules
        self.spectral_aug = SpectralAugmentation(**self.config['spectral_aug'])
        self.dynamic_range = DynamicRangeAugmentation(**self.config['dynamic_range'])
        self.rir_simulation = RIRSimulation(**self.config['rir_simulation'])
        self.noise_aug = AdaptiveNoiseAugmentation(**self.config['noise_aug'])
        self.spectral_morph = SpectralMorphing(**self.config['spectral_morph'])
        
    def forward(self, 
                clean_audio: torch.Tensor,
                reverberant_audio: torch.Tensor,
                clean_spec: torch.Tensor,
                reverberant_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply comprehensive data augmentation pipeline
        
        Args:
            clean_audio: Clean audio signals [B, samples]
            reverberant_audio: Reverberant audio signals [B, samples]  
            clean_spec: Clean spectrograms [B, C, F, T]
            reverberant_spec: Reverberant spectrograms [B, C, F, T]
        Returns:
            Tuple of augmented (clean_audio, reverberant_audio, clean_spec, reverberant_spec)
        """
        if not self.training_mode:
            return clean_audio, reverberant_audio, clean_spec, reverberant_spec
            
        # Audio-domain augmentations
        
        # 1. Apply RIR simulation to clean audio (create additional reverberation)
        if random.random() < 0.3:  # Sometimes add extra reverberation
            clean_audio = self.rir_simulation(clean_audio)
            
        # 2. Apply adaptive noise to both clean and reverberant audio
        clean_audio = self.noise_aug(clean_audio)
        reverberant_audio = self.noise_aug(reverberant_audio)
        
        # Spectrogram-domain augmentations
        
        # 3. Apply dynamic range modification
        clean_spec = self.dynamic_range(clean_spec)
        reverberant_spec = self.dynamic_range(reverberant_spec)
        
        # 4. Apply spectral augmentation with reverb awareness
        is_reverberant = torch.ones(clean_spec.shape[0], dtype=torch.bool, device=clean_spec.device)
        clean_spec = self.spectral_aug(clean_spec, ~is_reverberant)  # Clean signals
        reverberant_spec = self.spectral_aug(reverberant_spec, is_reverberant)  # Reverberant signals
        
        # 5. Apply spectral morphing between pairs (creates diverse acoustic conditions)
        if random.random() < self.config['spectral_morph']['morph_prob']:
            # Randomly pair samples for morphing
            indices = torch.randperm(clean_spec.shape[0])
            clean_spec_morphed, reverberant_spec_morphed = self.spectral_morph(
                clean_spec, reverberant_spec[indices]
            )
            clean_spec = clean_spec_morphed
            reverberant_spec = reverberant_spec_morphed
            
        return clean_audio, reverberant_audio, clean_spec, reverberant_spec
    
    def set_training_mode(self, training: bool):
        """Set training mode for augmentation"""
        self.training_mode = training
        return self
    
    def get_augmentation_stats(self) -> Dict[str, float]:
        """Get statistics about applied augmentations"""
        return {
            'spectral_mask_prob': self.config['spectral_aug']['mask_prob'],
            'dynamic_range_prob': self.config['dynamic_range']['apply_prob'],
            'rir_simulation_prob': self.config['rir_simulation']['apply_prob'],
            'noise_aug_prob': self.config['noise_aug']['apply_prob'],
            'spectral_morph_prob': self.config['spectral_morph']['morph_prob']
        }