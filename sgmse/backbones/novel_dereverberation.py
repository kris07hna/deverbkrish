"""
Novel Dereverberation Architecture

This module implements advanced architectural improvements for dereverberation:
1. Multi-Scale Attention Mechanism
2. Frequency-Aware Cross-Attention
3. Adaptive Spectral Gating
4. Room Impulse Response (RIR) Estimation Module
5. Perceptual Loss Integration
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism that processes different frequency bands
    with specialized attention patterns
    """
    
    def __init__(self, channels: int, num_scales: int = 3, reduction: int = 16):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        
        # Different kernel sizes for different scales
        self.scales = [3, 7, 15][:num_scales]
        
        # Scale-specific convolutions
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // num_scales, kernel_size=(1, k), 
                     padding=(0, k//2), groups=channels // num_scales)
            for k in self.scales
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        
        # Split channels for different scales
        scale_features = []
        chunk_size = C // self.num_scales
        
        for i, (conv, scale) in enumerate(zip(self.scale_convs, self.scales)):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_scales - 1 else C
            
            x_scale = x[:, start_idx:end_idx]
            feat = conv(x_scale)
            scale_features.append(feat)
        
        # Concatenate scale features
        multi_scale_feat = torch.cat(scale_features, dim=1)
        
        # Apply attention
        attention_weights = self.attention(multi_scale_feat)
        attended_feat = multi_scale_feat * attention_weights
        
        # Residual connection and fusion
        output = self.fusion(attended_feat) + x
        
        return output

class FrequencyAwareCrossAttention(nn.Module):
    """
    Cross-attention mechanism that leverages frequency domain knowledge
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        # Frequency positional encoding
        self.freq_embed = nn.Parameter(torch.randn(1, 1, 513, channels))  # For 1024 FFT
        
        # Output projection
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query_feat: torch.Tensor, key_feat: torch.Tensor, 
                value_feat: torch.Tensor) -> torch.Tensor:
        B, C, F, T = query_feat.shape
        
        # Reshape for attention computation
        q = query_feat.permute(0, 2, 3, 1).reshape(B * F, T, C)  # (B*F, T, C)
        k = key_feat.permute(0, 2, 3, 1).reshape(B * F, T, C)
        v = value_feat.permute(0, 2, 3, 1).reshape(B * F, T, C)
        
        # Add frequency positional encoding
        freq_pos = self.freq_embed[:, :, :F].permute(0, 2, 1, 3).reshape(1, F, 1, C)
        freq_pos = freq_pos.expand(B, -1, T, -1).reshape(B * F, T, C)
        
        q = q + freq_pos
        k = k + freq_pos
        v = v + freq_pos
        
        # Multi-head attention
        q = self.query(q).view(B * F, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(k).view(B * F, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(v).view(B * F, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B * F, T, C)
        
        # Output projection
        output = self.proj(attn_output)
        
        # Reshape back to (B, C, F, T)
        output = output.reshape(B, F, T, C).permute(0, 3, 1, 2)
        
        return output

class AdaptiveSpectralGating(nn.Module):
    """
    Adaptive spectral gating for frequency-specific processing
    """
    
    def __init__(self, channels: int, num_freq_bands: int = 4):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        
        # Frequency band processors
        self.band_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(num_freq_bands)
        ])
        
        # Band combination
        self.band_combiner = nn.Conv2d(channels * num_freq_bands, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        
        # Split frequency dimension into bands
        band_size = F // self.num_freq_bands
        band_outputs = []
        
        for i, processor in enumerate(self.band_processors):
            start_freq = i * band_size
            end_freq = start_freq + band_size if i < self.num_freq_bands - 1 else F
            
            band_input = x[:, :, start_freq:end_freq, :]
            
            # Apply adaptive gating
            gate = processor(band_input)
            gated_output = band_input * gate
            
            # Pad to original frequency size for concatenation
            if gated_output.shape[2] < band_size:
                padding = band_size - gated_output.shape[2]
                gated_output = F.pad(gated_output, (0, 0, 0, padding))
            
            band_outputs.append(gated_output)
        
        # Concatenate bands
        concatenated = torch.cat(band_outputs, dim=1)
        
        # Combine bands
        combined = self.band_combiner(concatenated)
        
        # Ensure output has correct frequency dimension
        combined = combined[:, :, :F, :]
        
        return combined + x  # Residual connection

class RIREstimationModule(nn.Module):
    """
    Room Impulse Response estimation module for reverb characterization
    """
    
    def __init__(self, input_channels: int, rir_length: int = 256):
        super().__init__()
        self.rir_length = rir_length
        
        # Feature extraction for RIR estimation
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 32)),  # Reduce spatial dimensions
        )
        
        # RIR predictor
        self.rir_predictor = nn.Sequential(
            nn.Linear(64 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, rir_length),
            nn.Tanh()  # RIR values typically in [-1, 1]
        )
        
        # RIR conditioning network
        self.rir_conditioner = nn.Sequential(
            nn.Linear(rir_length, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, F, T = x.shape
        
        # Extract features for RIR estimation
        features = self.feature_extractor(x)
        features_flat = features.view(B, -1)
        
        # Predict RIR
        predicted_rir = self.rir_predictor(features_flat)
        
        # Generate conditioning signal
        conditioning = self.rir_conditioner(predicted_rir)
        conditioning = conditioning.view(B, C, 1, 1)
        
        # Apply conditioning to input
        conditioned_x = x * conditioning
        
        return conditioned_x, predicted_rir

class NovelDereverbModel(nn.Module):
    """
    Novel dereverberation model combining all advanced components
    """
    
    def __init__(self, input_channels: int = 1, hidden_channels: int = 64,
                 num_layers: int = 4, time_embedding_dim: int = 128):
        super().__init__()
        
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Initial convolution
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        
        # RIR estimation module
        self.rir_estimator = RIREstimationModule(hidden_channels)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                MultiScaleAttention(hidden_channels),
                AdaptiveSpectralGating(hidden_channels),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU()
            ) for _ in range(num_layers // 2)
        ])
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            FrequencyAwareCrossAttention(hidden_channels)
            for _ in range(2)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                MultiScaleAttention(hidden_channels),
                AdaptiveSpectralGating(hidden_channels),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU()
            ) for _ in range(num_layers // 2)
        ])
        
        # Output projection
        self.output_conv = nn.Conv2d(hidden_channels, input_channels, 3, padding=1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings
        """
        half_dim = self.time_embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Noisy input spectrogram (B, C, F, T)
            y: Conditioning spectrogram (reverberant) (B, C, F, T)
            t: Timestep tensor (B,)
            
        Returns:
            Predicted noise or clean signal
        """
        # Time embedding
        t_emb = self.timestep_embedding(t)
        t_emb = self.time_embed(t_emb)  # (B, hidden_channels)
        
        # Initial feature extraction
        h = self.input_conv(x)
        
        # Add time embedding
        t_emb = t_emb.view(-1, h.shape[1], 1, 1)
        h = h + t_emb
        
        # RIR estimation and conditioning
        h, predicted_rir = self.rir_estimator(h)
        
        # Encoder
        encoder_features = []
        for layer in self.encoder_layers:
            h = layer(h)
            encoder_features.append(h)
        
        # Cross-attention with conditioning
        y_feat = self.input_conv(y) + t_emb
        for cross_attn in self.cross_attention_layers:
            h = cross_attn(h, y_feat, y_feat) + h
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            if i < len(encoder_features):
                h = h + encoder_features[-(i+1)]  # Skip connections
            h = layer(h)
        
        # Output
        output = self.output_conv(h)
        
        return output

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained features for better audio quality
    """
    
    def __init__(self, feature_layers: List[int] = [2, 4, 6]):
        super().__init__()
        self.feature_layers = feature_layers
        
        # Simple feature extractor (can be replaced with pre-trained audio features)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 0
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 1
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 5
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),  # Layer 6
        )
        
        # Freeze feature extractor if using pre-trained weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        h = x
        for i, layer in enumerate(self.feature_extractor):
            h = layer(h)
            if i in self.feature_layers:
                features.append(h)
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)

def create_novel_dereverberation_model(config: dict) -> NovelDereverbModel:
    """
    Factory function to create the novel dereverberation model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured model instance
    """
    return NovelDereverbModel(
        input_channels=config.get('input_channels', 1),
        hidden_channels=config.get('hidden_channels', 64),
        num_layers=config.get('num_layers', 4),
        time_embedding_dim=config.get('time_embedding_dim', 128)
    )