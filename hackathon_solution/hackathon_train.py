#!/usr/bin/env python3
"""
Enhanced SGMSE+ Training Script for De-reverberation Hackathon
=============================================================

This script implements the complete training pipeline for our SOTA de-reverberation solution.
It includes all novel features and architectural improvements.

Usage:
    python hackathon_train.py --config config.yaml [additional args]

Author: SOTA Hackathon Team
"""

import os
import sys
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
import warnings

# Add hackathon solution to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_model import EnhancedScoreModel
from advanced_data_augmentation import AdvancedDataAugmentation
from sgmse.data_module import SpecsDataModule


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Create necessary directories"""
    for path_key in ['output_dir', 'model_dir', 'results_dir', 'submission_dir']:
        path = config['paths'][path_key]
        os.makedirs(path, exist_ok=True)
        print(f"📁 Created directory: {path}")


def create_enhanced_model(config: dict) -> EnhancedScoreModel:
    """Create enhanced SGMSE+ model with novel features"""
    
    model_config = {
        **config['model'],
        **config['enhanced_features'],
        **config['training'],
        **config['data']
    }
    
    print("🚀 Creating Enhanced SGMSE+ Model...")
    print("🎯 Novel Features Active:")
    for feature, enabled in config['enhanced_features'].items():
        if enabled:
            print(f"  ✅ {feature.replace('_', ' ').title()}")
    
    model = EnhancedScoreModel(
        data_module_cls=SpecsDataModule,
        **model_config
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")
    
    return model


def create_data_module(config: dict) -> SpecsDataModule:
    """Create data module with enhanced configuration"""
    
    data_config = {
        'base_dir': '/kaggle/input',  # Kaggle input directory
        'batch_size': config['training']['batch_size'],
        'num_workers': config['hardware']['num_workers'],
        **config['data']
    }
    
    print("📚 Creating Enhanced Data Module...")
    data_module = SpecsDataModule(**data_config)
    
    return data_module


def setup_callbacks(config: dict) -> list:
    """Setup training callbacks"""
    
    callbacks = [
        ModelCheckpoint(
            dirpath=config['paths']['model_dir'],
            filename='enhanced_sgmse_best_{epoch:02d}_{pesq:.3f}',
            monitor='pesq',
            mode='max',
            save_top_k=3,
            verbose=True
        ),
        ModelCheckpoint(
            dirpath=config['paths']['model_dir'],
            filename='enhanced_sgmse_last_{epoch:02d}',
            save_last=True
        ),
        EarlyStopping(
            monitor='pesq',
            patience=10,
            mode='max',
            verbose=True
        )
    ]
    
    print("📋 Training callbacks configured:")
    print("  ✅ Best model checkpointing (PESQ-based)")
    print("  ✅ Last checkpoint saving")
    print("  ✅ Early stopping (patience=10)")
    
    return callbacks


def setup_trainer(config: dict, callbacks: list) -> pl.Trainer:
    """Setup PyTorch Lightning trainer"""
    
    logger = CSVLogger(
        save_dir=config['paths']['results_dir'],
        name='enhanced_sgmse_logs'
    )
    
    trainer_config = {
        'max_epochs': config['training']['max_epochs'],
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'callbacks': callbacks,
        'logger': logger,
        'log_every_n_steps': 50,
        **config['optimizations']
    }
    
    trainer = pl.Trainer(**trainer_config)
    
    print("🏃‍♂️ Trainer configured:")
    print(f"  Max epochs: {config['training']['max_epochs']}")
    print(f"  Mixed precision: {config['optimizations']['mixed_precision']}")
    print(f"  Gradient clipping: {config['optimizations']['gradient_clipping']}")
    print(f"  Effective batch size: {config['training']['batch_size'] * config['optimizations']['accumulate_grad_batches']}")
    
    return trainer


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Enhanced SGMSE+ Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with reduced epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Test mode adjustments
    if args.test_mode:
        config['training']['max_epochs'] = 5
        config['training']['batch_size'] = 2
        print("⚠️ Running in test mode (5 epochs, batch_size=2)")
    
    # Setup directories
    setup_directories(config)
    
    print("\n" + "="*60)
    print("🎯 ENHANCED SGMSE+ DE-REVERBERATION TRAINING")
    print("="*60)
    
    # Create components
    model = create_enhanced_model(config)
    data_module = create_data_module(config)
    callbacks = setup_callbacks(config)
    trainer = setup_trainer(config, callbacks)
    
    # Print training information
    print(f"\n🎵 Dataset Configuration:")
    print(f"  Clean data: {config['paths']['clean_data']}")
    print(f"  Reverb data: {config['paths']['reverb_data']}")
    print(f"  Sample rate: {config['model']['sr']} Hz")
    
    print(f"\n🎯 Performance Targets:")
    print(f"  PESQ: {config['targets']['pesq']}")
    print(f"  STOI: {config['targets']['stoi']}")
    print(f"  SI-SDR: {config['targets']['si_sdr']} dB")
    
    print(f"\n⚡ Computational Constraints:")
    print(f"  Max GMAC/s: {config['computational']['max_gmacs']}")
    print(f"  Memory limit: {config['computational']['memory_limit_gb']} GB")
    
    # Start training
    print("\n🚀 Starting Enhanced SGMSE+ Training...")
    print("🎯 Novel Features Active:")
    print("  ✅ Adaptive Loss Weighting")
    print("  ✅ Multi-Scale Feature Fusion")
    print("  ✅ Spectral Consistency Loss")
    print("  ✅ Progressive Training")
    print("  ✅ Advanced Data Augmentation")
    
    try:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Models saved in: {config['paths']['model_dir']}")
        
        # Save final configuration
        final_config_path = os.path.join(config['paths']['model_dir'], 'final_config.yaml')
        with open(final_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"⚙️ Configuration saved: {final_config_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise e


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    main()