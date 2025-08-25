"""
Advanced Data Augmentation for Dereverberation

This module provides sophisticated data augmentation techniques specifically
designed for dereverberation tasks:
1. Synthetic reverb generation with realistic parameters
2. Multi-room impulse response simulation
3. Dynamic range and spectral augmentation
4. Adversarial augmentation for robustness
5. Frequency-domain augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Optional, Dict
import math

class SyntheticReverbGenerator:
    """
    Generates synthetic reverb using physically-inspired models
    """
    
    def __init__(self, sample_rate: int = 16000, max_rir_length: float = 2.0):
        self.sample_rate = sample_rate
        self.max_rir_length = max_rir_length
        self.max_samples = int(max_rir_length * sample_rate)
        
    def generate_exponential_decay_rir(self, rt60: float, room_size: float = 1.0,
                                     direct_path_delay: float = 0.001) -> np.ndarray:
        """
        Generate RIR with exponential decay model
        
        Args:
            rt60: Reverberation time (T60) in seconds
            room_size: Room size factor (affects early reflections)
            direct_path_delay: Direct path delay in seconds
            
        Returns:
            Room impulse response
        """
        # Calculate decay constant
        decay_constant = -3 * np.log(10) / rt60
        
        # Time vector
        t = np.linspace(0, self.max_rir_length, self.max_samples)
        
        # Direct path impulse
        direct_delay_samples = int(direct_path_delay * self.sample_rate)
        direct_path = np.zeros(self.max_samples)
        direct_path[direct_delay_samples] = 1.0
        
        # Early reflections (first 50ms)
        early_reflection_time = 0.05
        early_samples = int(early_reflection_time * self.sample_rate)
        
        # Generate early reflections
        num_reflections = int(room_size * 20)  # More reflections for larger rooms
        early_reflections = np.zeros(self.max_samples)
        
        for i in range(num_reflections):
            delay = np.random.uniform(0.005, early_reflection_time)
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < self.max_samples:
                amplitude = np.random.uniform(0.1, 0.5) * np.exp(decay_constant * delay)
                early_reflections[delay_samples] += amplitude
        
        # Late reverberation (exponential decay with noise)
        late_reverb = np.random.randn(self.max_samples) * 0.1
        late_reverb[:early_samples] = 0  # No late reverb in early reflection period
        
        # Apply exponential decay
        decay_envelope = np.exp(decay_constant * t)
        late_reverb *= decay_envelope
        
        # Combine components
        rir = direct_path + early_reflections + late_reverb
        
        # Normalize
        rir = rir / np.max(np.abs(rir))
        
        return rir.astype(np.float32)
    
    def generate_parametric_rir(self, room_dimensions: Tuple[float, float, float],
                              source_pos: Tuple[float, float, float],
                              mic_pos: Tuple[float, float, float],
                              absorption_coeff: float = 0.2) -> np.ndarray:
        """
        Generate RIR using image source method (simplified)
        
        Args:
            room_dimensions: (width, height, depth) in meters
            source_pos: Source position (x, y, z) in meters
            mic_pos: Microphone position (x, y, z) in meters
            absorption_coeff: Wall absorption coefficient
            
        Returns:
            Room impulse response
        """
        width, height, depth = room_dimensions
        sx, sy, sz = source_pos
        mx, my, mz = mic_pos
        
        # Speed of sound
        c = 343.0  # m/s
        
        rir = np.zeros(self.max_samples)
        
        # Image source positions (limited to first few reflections for efficiency)
        max_order = 3
        
        for nx in range(-max_order, max_order + 1):
            for ny in range(-max_order, max_order + 1):
                for nz in range(-max_order, max_order + 1):
                    # Image source position
                    if nx % 2 == 0:
                        isx = sx + nx * width
                    else:
                        isx = width - sx + nx * width
                    
                    if ny % 2 == 0:
                        isy = sy + ny * height
                    else:
                        isy = height - sy + ny * height
                    
                    if nz % 2 == 0:
                        isz = sz + nz * depth
                    else:
                        isz = depth - sz + nz * depth
                    
                    # Distance from image source to microphone
                    distance = np.sqrt((isx - mx)**2 + (isy - my)**2 + (isz - mz)**2)
                    
                    # Time delay
                    delay = distance / c
                    delay_samples = int(delay * self.sample_rate)
                    
                    if delay_samples < self.max_samples:
                        # Amplitude considering distance and wall reflections
                        num_reflections = abs(nx) + abs(ny) + abs(nz)
                        amplitude = (1 - absorption_coeff) ** num_reflections / distance
                        
                        rir[delay_samples] += amplitude
        
        # Normalize
        if np.max(np.abs(rir)) > 0:
            rir = rir / np.max(np.abs(rir))
        
        return rir.astype(np.float32)

class MultiRoomSimulator:
    """
    Simulates multiple room acoustics for diverse training data
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.reverb_generator = SyntheticReverbGenerator(sample_rate)
        
        # Predefined room types with typical parameters
        self.room_types = {
            'small_room': {
                'dimensions': [(3, 3, 3), (4, 3, 3), (3, 4, 3)],
                'rt60_range': (0.2, 0.6),
                'absorption_range': (0.15, 0.4)
            },
            'medium_room': {
                'dimensions': [(6, 5, 3), (7, 6, 3), (8, 5, 3)],
                'rt60_range': (0.4, 1.2),
                'absorption_range': (0.1, 0.3)
            },
            'large_room': {
                'dimensions': [(10, 8, 4), (12, 10, 4), (15, 12, 5)],
                'rt60_range': (0.8, 2.0),
                'absorption_range': (0.05, 0.2)
            },
            'hall': {
                'dimensions': [(20, 15, 8), (25, 20, 10), (30, 25, 12)],
                'rt60_range': (1.5, 3.0),
                'absorption_range': (0.02, 0.1)
            }
        }
    
    def generate_random_room_rir(self, room_type: Optional[str] = None) -> np.ndarray:
        """
        Generate RIR for a random room configuration
        
        Args:
            room_type: Type of room ('small_room', 'medium_room', etc.)
                      If None, randomly selected
                      
        Returns:
            Room impulse response
        """
        if room_type is None:
            room_type = random.choice(list(self.room_types.keys()))
        
        room_config = self.room_types[room_type]
        
        # Random room dimensions
        dimensions = random.choice(room_config['dimensions'])
        
        # Random source and microphone positions
        w, h, d = dimensions
        source_pos = (
            random.uniform(0.5, w - 0.5),
            random.uniform(0.5, h - 0.5),
            random.uniform(0.5, d - 0.5)
        )
        mic_pos = (
            random.uniform(0.5, w - 0.5),
            random.uniform(0.5, h - 0.5),
            random.uniform(0.5, d - 0.5)
        )
        
        # Random acoustic parameters
        rt60_min, rt60_max = room_config['rt60_range']
        rt60 = random.uniform(rt60_min, rt60_max)
        
        abs_min, abs_max = room_config['absorption_range']
        absorption = random.uniform(abs_min, abs_max)
        
        # Generate RIR
        if random.random() < 0.5:  # Use parametric model
            rir = self.reverb_generator.generate_parametric_rir(
                dimensions, source_pos, mic_pos, absorption
            )
        else:  # Use exponential decay model
            room_size = (w * h * d) / 50.0  # Normalize room size
            rir = self.reverb_generator.generate_exponential_decay_rir(
                rt60, room_size
            )
        
        return rir

class FrequencyDomainAugmentation:
    """
    Frequency-domain specific augmentations
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def spectral_masking(self, spectrogram: torch.Tensor, 
                        freq_mask_ratio: float = 0.1,
                        time_mask_ratio: float = 0.1) -> torch.Tensor:
        """
        Apply spectral masking similar to SpecAugment
        
        Args:
            spectrogram: Input spectrogram (B, C, F, T)
            freq_mask_ratio: Ratio of frequency bins to mask
            time_mask_ratio: Ratio of time frames to mask
            
        Returns:
            Masked spectrogram
        """
        B, C, F, T = spectrogram.shape
        masked_spec = spectrogram.clone()
        
        for b in range(B):
            # Frequency masking
            freq_mask_size = int(F * freq_mask_ratio)
            if freq_mask_size > 0:
                freq_start = random.randint(0, F - freq_mask_size)
                masked_spec[b, :, freq_start:freq_start + freq_mask_size, :] = 0
            
            # Time masking
            time_mask_size = int(T * time_mask_ratio)
            if time_mask_size > 0:
                time_start = random.randint(0, T - time_mask_size)
                masked_spec[b, :, :, time_start:time_start + time_mask_size] = 0
        
        return masked_spec
    
    def frequency_shift(self, spectrogram: torch.Tensor, 
                       max_shift_bins: int = 5) -> torch.Tensor:
        """
        Apply random frequency shifting
        
        Args:
            spectrogram: Input spectrogram (B, C, F, T)
            max_shift_bins: Maximum frequency bins to shift
            
        Returns:
            Frequency-shifted spectrogram
        """
        B, C, F, T = spectrogram.shape
        shifted_spec = torch.zeros_like(spectrogram)
        
        for b in range(B):
            shift = random.randint(-max_shift_bins, max_shift_bins)
            
            if shift > 0:
                shifted_spec[b, :, shift:, :] = spectrogram[b, :, :-shift, :]
            elif shift < 0:
                shifted_spec[b, :, :shift, :] = spectrogram[b, :, -shift:, :]
            else:
                shifted_spec[b] = spectrogram[b]
        
        return shifted_spec
    
    def spectral_dropout(self, spectrogram: torch.Tensor, 
                        dropout_prob: float = 0.1) -> torch.Tensor:
        """
        Apply spectral dropout (randomly zero out frequency bins)
        
        Args:
            spectrogram: Input spectrogram (B, C, F, T)
            dropout_prob: Probability of dropping each frequency bin
            
        Returns:
            Spectrogram with spectral dropout
        """
        if self.training:
            mask = torch.rand_like(spectrogram) > dropout_prob
            return spectrogram * mask
        return spectrogram

class DynamicRangeAugmentation:
    """
    Dynamic range and amplitude augmentations
    """
    
    def __init__(self):
        pass
    
    def dynamic_range_compression(self, audio: torch.Tensor, 
                                ratio: float = 4.0, 
                                threshold: float = -20.0) -> torch.Tensor:
        """
        Apply dynamic range compression
        
        Args:
            audio: Input audio tensor
            ratio: Compression ratio
            threshold: Compression threshold in dB
            
        Returns:
            Compressed audio
        """
        # Convert to dB
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-8)
        
        # Apply compression above threshold
        compressed_db = torch.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear scale
        compressed_audio = torch.sign(audio) * torch.pow(10, compressed_db / 20)
        
        return compressed_audio
    
    def random_gain(self, audio: torch.Tensor, 
                   gain_range: Tuple[float, float] = (0.5, 2.0)) -> torch.Tensor:
        """
        Apply random gain
        
        Args:
            audio: Input audio tensor
            gain_range: Range of gain values
            
        Returns:
            Audio with random gain applied
        """
        gain = random.uniform(*gain_range)
        return audio * gain
    
    def clipping_simulation(self, audio: torch.Tensor, 
                          clip_threshold: float = 0.95) -> torch.Tensor:
        """
        Simulate audio clipping
        
        Args:
            audio: Input audio tensor
            clip_threshold: Clipping threshold
            
        Returns:
            Clipped audio
        """
        return torch.clamp(audio, -clip_threshold, clip_threshold)

class AdversarialAugmentation:
    """
    Adversarial augmentation for robustness
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 0.01):
        self.model = model
        self.epsilon = epsilon
    
    def fgsm_attack(self, input_tensor: torch.Tensor, 
                   target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Fast Gradient Sign Method attack for adversarial training
        
        Args:
            input_tensor: Input tensor
            target_tensor: Target tensor
            
        Returns:
            Adversarially perturbed input
        """
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        loss = F.mse_loss(output, target_tensor)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        data_grad = input_tensor.grad.data
        perturbed_input = input_tensor + self.epsilon * data_grad.sign()
        
        return perturbed_input.detach()

class AdvancedAugmentationPipeline:
    """
    Main augmentation pipeline combining all techniques
    """
    
    def __init__(self, sample_rate: int = 16000, config: Optional[Dict] = None):
        self.sample_rate = sample_rate
        self.config = config or {}
        
        # Initialize augmentation modules
        self.room_simulator = MultiRoomSimulator(sample_rate)
        self.freq_aug = FrequencyDomainAugmentation(sample_rate)
        self.dynamic_aug = DynamicRangeAugmentation()
        
        # Augmentation probabilities
        self.aug_probs = self.config.get('augmentation_probabilities', {
            'add_reverb': 0.8,
            'spectral_masking': 0.3,
            'frequency_shift': 0.2,
            'dynamic_compression': 0.3,
            'random_gain': 0.5,
            'clipping': 0.1
        })
    
    def apply_reverb_augmentation(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """
        Apply reverb augmentation to clean audio
        
        Args:
            clean_audio: Clean audio tensor (B, C, T)
            
        Returns:
            Reverberant audio tensor
        """
        B, C, T = clean_audio.shape
        reverberant_audio = clean_audio.clone()
        
        for b in range(B):
            if random.random() < self.aug_probs['add_reverb']:
                # Generate random RIR
                rir = self.room_simulator.generate_random_room_rir()
                rir_tensor = torch.from_numpy(rir).to(clean_audio.device)
                
                # Convolve with RIR
                for c in range(C):
                    # Pad and convolve
                    audio_padded = F.pad(clean_audio[b, c], (len(rir) - 1, 0))
                    reverberant = F.conv1d(
                        audio_padded.unsqueeze(0).unsqueeze(0),
                        rir_tensor.flip(0).unsqueeze(0).unsqueeze(0)
                    ).squeeze()
                    
                    # Trim to original length
                    reverberant_audio[b, c] = reverberant[:T]
        
        return reverberant_audio
    
    def apply_spectral_augmentations(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral domain augmentations
        
        Args:
            spectrogram: Input spectrogram (B, C, F, T)
            
        Returns:
            Augmented spectrogram
        """
        augmented_spec = spectrogram
        
        if random.random() < self.aug_probs['spectral_masking']:
            augmented_spec = self.freq_aug.spectral_masking(augmented_spec)
        
        if random.random() < self.aug_probs['frequency_shift']:
            augmented_spec = self.freq_aug.frequency_shift(augmented_spec)
        
        return augmented_spec
    
    def apply_dynamic_augmentations(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic range augmentations
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Augmented audio tensor
        """
        augmented_audio = audio
        
        if random.random() < self.aug_probs['dynamic_compression']:
            augmented_audio = self.dynamic_aug.dynamic_range_compression(augmented_audio)
        
        if random.random() < self.aug_probs['random_gain']:
            augmented_audio = self.dynamic_aug.random_gain(augmented_audio)
        
        if random.random() < self.aug_probs['clipping']:
            augmented_audio = self.dynamic_aug.clipping_simulation(augmented_audio)
        
        return augmented_audio
    
    def __call__(self, clean_audio: torch.Tensor, 
                 mode: str = 'training') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply full augmentation pipeline
        
        Args:
            clean_audio: Clean audio tensor (B, C, T)
            mode: 'training' or 'inference'
            
        Returns:
            Tuple of (clean_audio, reverberant_audio)
        """
        if mode != 'training':
            return clean_audio, clean_audio
        
        # Apply reverb augmentation
        reverberant_audio = self.apply_reverb_augmentation(clean_audio)
        
        # Apply dynamic range augmentations
        clean_audio = self.apply_dynamic_augmentations(clean_audio)
        reverberant_audio = self.apply_dynamic_augmentations(reverberant_audio)
        
        return clean_audio, reverberant_audio

def create_augmentation_pipeline(config: Dict) -> AdvancedAugmentationPipeline:
    """
    Factory function to create augmentation pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured augmentation pipeline
    """
    return AdvancedAugmentationPipeline(
        sample_rate=config.get('sample_rate', 16000),
        config=config
    )