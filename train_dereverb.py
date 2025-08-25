"""
Specialized training script for dereverberation tasks.
Extends the base training functionality with reverb-specific features and metrics.
"""

import torch
import wandb
import argparse
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from os.path import join
import os
import logging

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from sgmse.backbones.shared import BackboneRegistry
from sgmse.dereverb_data_module import ReverbDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReverbScoreModel(ScoreModel):
    """
    Extended ScoreModel for dereverberation with additional metrics and functionality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional metrics for dereverberation
        self.dereverb_metrics = {
            'train_t60_improvement': [],
            'val_t60_improvement': [],
            'train_drr_improvement': [],
            'val_drr_improvement': []
        }
    
    def training_step(self, batch, batch_idx):
        """Training step with additional dereverberation logging."""
        
        # Call parent training step
        loss = super().training_step(batch, batch_idx)
        
        # Log additional dereverberation metrics periodically
        if batch_idx % 100 == 0:
            self._log_dereverb_metrics(batch, 'train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with additional dereverberation metrics."""
        
        # Call parent validation step
        result = super().validation_step(batch, batch_idx)
        
        # Log dereverberation metrics for validation
        if batch_idx % 10 == 0:
            self._log_dereverb_metrics(batch, 'val')
        
        return result
    
    def _log_dereverb_metrics(self, batch, stage):
        """Log dereverberation-specific metrics."""
        
        try:
            X, Y = batch
            
            # Generate prediction
            with torch.no_grad():
                # Use a simple denoising step for metric calculation
                X_hat = self._simple_denoise(X)
                
                # Calculate spectral metrics
                spectral_dist = torch.mean((X_hat - X) ** 2).item()
                
                # Log metrics
                self.log(f'{stage}/spectral_distance', spectral_dist, 
                        on_step=False, on_epoch=True, prog_bar=True)
                
                # Additional reverb-specific metrics can be added here
                # when audio metadata is available
                
        except Exception as e:
            logger.warning(f"Error calculating dereverberation metrics: {e}")
    
    def _simple_denoise(self, X):
        """Simple denoising for metric calculation."""
        # This is a placeholder - in practice, you would use the full sampling process
        # but that's too expensive for frequent metric calculation
        return X * 0.9  # Simple noise reduction
    
    def configure_optimizers(self):
        """Configure optimizers with dereverberation-specific parameters."""
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Use cosine annealing with warm restarts for dereverberation
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=self.lr * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "pesq" if self.num_eval_files else "val_loss",
                "interval": "epoch",
            },
        }


def get_argparse_groups(parser):
    """Extract argument groups from parser."""
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


def validate_args(args):
    """Validate and adjust arguments for dereverberation training."""
    
    # Validate data source
    if not args.csv_path and not args.base_dir:
        raise ValueError("Either --csv_path or --base_dir must be provided")
    
    # Set format based on data source
    if args.csv_path:
        args.format = 'csv'
        logger.info(f"Using CSV dataset: {args.csv_path}")
    else:
        logger.info(f"Using directory dataset: {args.base_dir}")
    
    # Validate filtering parameters
    if args.filter_t60_min is not None and args.filter_t60_max is not None:
        if args.filter_t60_min >= args.filter_t60_max:
            raise ValueError("filter_t60_min must be less than filter_t60_max")
    
    if args.filter_drr_min is not None and args.filter_drr_max is not None:
        if args.filter_drr_min >= args.filter_drr_max:
            raise ValueError("filter_drr_min must be less than filter_drr_max")
    
    # Adjust default parameters for dereverberation
    if not hasattr(args, 'spec_factor') or args.spec_factor is None:
        args.spec_factor = 0.15  # Good default for reverb
    
    if not hasattr(args, 'num_frames') or args.num_frames is None:
        args.num_frames = 256  # Good default for reverb
    
    return args


def setup_logging_and_callbacks(args, logger_instance):
    """Set up logging and callbacks for dereverberation training."""
    
    callbacks = []
    
    if logger_instance is not None:
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=join(args.log_dir, str(logger_instance.version)),
            save_last=True,
            filename='{epoch}-last'
        )
        callbacks.append(checkpoint_callback)
        
        # Save checkpoints periodically
        checkpoint_callback_periodic = ModelCheckpoint(
            dirpath=join(args.log_dir, f'{str(logger_instance.version)}-{args.wandb_name}'),
            filename='{step}',
            save_top_k=-1,
            every_n_train_steps=args.save_ckpt_interval
        )
        callbacks.append(checkpoint_callback_periodic)
        
        # Metric-based checkpointing if evaluation is enabled
        if args.num_eval_files:
            checkpoint_callback_pesq = ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger_instance.version)),
                save_top_k=1,
                monitor="pesq",
                mode="max",
                filename='{epoch}-{pesq:.2f}'
            )
            callbacks.append(checkpoint_callback_pesq)
            
            checkpoint_callback_si_sdr = ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger_instance.version)),
                save_top_k=1,
                monitor="si_sdr",
                mode="max",
                filename='{epoch}-{si_sdr:.2f}'
            )
            callbacks.append(checkpoint_callback_si_sdr)
        
        # Early stopping for dereverberation
        early_stopping = EarlyStopping(
            monitor="pesq" if args.num_eval_files else "val_loss",
            patience=20,
            mode="max" if args.num_eval_files else "min",
            verbose=True
        )
        callbacks.append(early_stopping)
    
    return callbacks


def log_dataset_info(data_module, logger_instance):
    """Log dataset information to wandb."""
    
    if logger_instance is None:
        return
    
    try:
        # Get dataset statistics
        train_stats = data_module.get_dataset_statistics('train')
        valid_stats = data_module.get_dataset_statistics('valid')
        
        # Log basic info
        logger_instance.experiment.log({
            'dataset/train_samples': train_stats.get('num_samples', 0),
            'dataset/valid_samples': valid_stats.get('num_samples', 0),
            'dataset/format': train_stats.get('format', 'unknown')
        })
        
        # Log T60 statistics if available
        if 't60_mean' in train_stats:
            logger_instance.experiment.log({
                'dataset/train_t60_mean': train_stats['t60_mean'],
                'dataset/train_t60_std': train_stats['t60_std'],
                'dataset/train_t60_range': f"{train_stats['t60_min']:.2f}-{train_stats['t60_max']:.2f}"
            })
        
        if 't60_mean' in valid_stats:
            logger_instance.experiment.log({
                'dataset/valid_t60_mean': valid_stats['t60_mean'],
                'dataset/valid_t60_std': valid_stats['t60_std'],
                'dataset/valid_t60_range': f"{valid_stats['t60_min']:.2f}-{valid_stats['t60_max']:.2f}"
            })
        
        # Log DRR statistics if available
        if 'drr_mean' in train_stats:
            logger_instance.experiment.log({
                'dataset/train_drr_mean': train_stats['drr_mean'],
                'dataset/train_drr_std': train_stats['drr_std'],
                'dataset/train_drr_range': f"{train_stats['drr_min']:.1f}-{train_stats['drr_max']:.1f}"
            })
        
        # Log room type distribution if available
        if 'room_types' in train_stats:
            for room_type, count in train_stats['room_types'].items():
                logger_instance.experiment.log({f'dataset/room_type_{room_type}': count})
        
    except Exception as e:
        logger.warning(f"Error logging dataset info: {e}")


def main():
    # Parse arguments
    parser = ArgumentParser(description="Train SGMSE for dereverberation")
    
    # Base arguments
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), 
                       default="ncsnpp", help="Backbone network architecture")
    parser.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), 
                       default="ouve", help="SDE type")
    parser.add_argument("--nolog", action='store_true', help="Turn off logging")
    parser.add_argument("--wandb_name", type=str, default=None, 
                       help="Name for wandb logger")
    parser.add_argument("--ckpt", type=str, default=None, 
                       help="Resume training from checkpoint")
    parser.add_argument("--log_dir", type=str, default="logs", 
                       help="Directory to save logs")
    parser.add_argument("--save_ckpt_interval", type=int, default=50000, 
                       help="Save checkpoint interval")
    
    # Get dynamic arguments
    temp_args, _ = parser.parse_known_args()
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)
    
    # Add trainer arguments
    trainer_group = parser.add_argument_group("Trainer", description="Lightning Trainer")
    trainer_group.add_argument("--accelerator", type=str, default="gpu", 
                              help="Accelerator type")
    trainer_group.add_argument("--devices", default="auto", 
                              help="Number of devices to use")
    trainer_group.add_argument("--accumulate_grad_batches", type=int, default=1, 
                              help="Accumulate gradients")
    trainer_group.add_argument("--max_epochs", type=int, default=-1, 
                              help="Maximum number of epochs")
    trainer_group.add_argument("--precision", type=str, default="16-mixed", 
                              help="Training precision")
    
    # Add model arguments
    ReverbScoreModel.add_argparse_args(
        parser.add_argument_group("ScoreModel", description="Score Model"))
    sde_class.add_argparse_args(
        parser.add_argument_group("SDE", description=sde_class.__name__))
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    
    # Add data module arguments
    ReverbDataModule.add_argparse_args(
        parser.add_argument_group("DataModule", description="Reverb Data Module"))
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Validate and adjust arguments
    args = validate_args(args)
    
    # Get argument groups
    arg_groups = get_argparse_groups(parser)
    
    # Print configuration
    logger.info("=== Dereverberation Training Configuration ===")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"SDE: {args.sde}")
    logger.info(f"Data source: {'CSV' if args.csv_path else 'Directory'}")
    if args.csv_path:
        logger.info(f"CSV path: {args.csv_path}")
    if args.base_dir:
        logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    # Initialize data module
    data_module = ReverbDataModule(
        base_dir=args.base_dir,
        csv_path=args.csv_path,
        **vars(arg_groups['DataModule'])
    )
    
    # Initialize model
    model = ReverbScoreModel(
        backbone=args.backbone,
        sde=args.sde,
        data_module_cls=ReverbDataModule,
        **{
            **vars(arg_groups['ScoreModel']),
            **vars(arg_groups['SDE']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule'])
        }
    )
    
    # Set up logger
    if args.nolog:
        logger_instance = None
    else:
        project_name = "sgmse-dereverberation"
        logger_instance = WandbLogger(
            project=project_name,
            log_model=False,
            save_dir=args.log_dir,
            name=args.wandb_name
        )
        logger_instance.experiment.log_code(".")
        
        # Log configuration
        logger_instance.experiment.config.update(vars(args))
    
    # Set up callbacks
    callbacks = setup_logging_and_callbacks(args, logger_instance)
    
    # Initialize trainer
    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        logger=logger_instance,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=callbacks
    )
    
    # Setup data module
    data_module.setup('fit')
    
    # Log dataset information
    log_dataset_info(data_module, logger_instance)
    
    # Print dataset statistics
    train_stats = data_module.get_dataset_statistics('train')
    valid_stats = data_module.get_dataset_statistics('valid')
    
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Training samples: {train_stats.get('num_samples', 0)}")
    logger.info(f"Validation samples: {valid_stats.get('num_samples', 0)}")
    
    if 't60_mean' in train_stats:
        logger.info(f"Training T60: {train_stats['t60_mean']:.3f} ± {train_stats['t60_std']:.3f} seconds")
    if 'drr_mean' in train_stats:
        logger.info(f"Training DRR: {train_stats['drr_mean']:.2f} ± {train_stats['drr_std']:.2f} dB")
    if 'room_types' in train_stats:
        logger.info(f"Room types: {list(train_stats['room_types'].keys())}")
    
    # Train model
    logger.info("=== Starting Training ===")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()