#!/usr/bin/env python3
"""
Enhanced Training Script with Hyperparameter Optimization

This script provides advanced training capabilities including:
1. Automated hyperparameter optimization using Optuna
2. Advanced learning rate scheduling
3. Model ensemble training
4. Gradient accumulation and mixed precision
5. Early stopping with model checkpointing
6. Real-time performance monitoring
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Hyperparameter optimization will be disabled.")

# Import existing modules
from sgmse.model import ScoreModel
from sgmse.data_module import SpecsDataModule

class AdvancedScheduler:
    """
    Advanced learning rate scheduler with multiple strategies
    """
    
    def __init__(self, optimizer: optim.Optimizer, config: Dict):
        self.optimizer = optimizer
        self.config = config
        self.scheduler_type = config.get('scheduler_type', 'cosine_with_restarts')
        
        if self.scheduler_type == 'cosine_with_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=config.get('T_0', 10),
                T_mult=config.get('T_mult', 2),
                eta_min=config.get('eta_min', 1e-6)
            )
        elif self.scheduler_type == 'one_cycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.get('max_lr', 1e-3),
                total_steps=config.get('total_steps', 1000),
                pct_start=config.get('pct_start', 0.3)
            )
        elif self.scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.get('factor', 0.5),
                patience=config.get('patience', 10),
                min_lr=config.get('min_lr', 1e-6)
            )
        else:
            # Default exponential decay
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=config.get('gamma', 0.99)
            )
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler"""
        if self.scheduler_type == 'reduce_on_plateau' and metrics is not None:
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate"""
        return self.scheduler.get_last_lr()

class EnhancedScoreModel(ScoreModel):
    """
    Enhanced version of ScoreModel with advanced training features
    """
    
    def __init__(self, *args, **kwargs):
        # Extract enhanced training parameters
        self.gradient_clip_val = kwargs.pop('gradient_clip_val', 1.0)
        self.accumulate_grad_batches = kwargs.pop('accumulate_grad_batches', 1)
        self.use_mixed_precision = kwargs.pop('use_mixed_precision', True)
        self.scheduler_config = kwargs.pop('scheduler_config', {})
        self.ensemble_config = kwargs.pop('ensemble_config', {})
        
        super().__init__(*args, **kwargs)
        
        # Initialize gradient scaler for mixed precision
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Track metrics for hyperparameter optimization
        self.val_metrics_history = []
        self.best_val_metric = float('inf')
        
        # Ensemble weights (if using ensemble training)
        if self.ensemble_config.get('enable_ensemble', False):
            self.ensemble_alpha = self.ensemble_config.get('alpha', 0.99)
            self.register_buffer('ensemble_weights', None)
    
    def configure_optimizers(self):
        """Configure optimizers with advanced scheduling"""
        # Get base optimizer configuration
        optimizer_config = super().configure_optimizers()
        
        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config
        
        # Create advanced scheduler
        if self.scheduler_config:
            scheduler = AdvancedScheduler(optimizer, self.scheduler_config)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler.scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Enhanced training step with mixed precision and gradient accumulation"""
        if self.use_mixed_precision:
            with autocast():
                loss = super().training_step(batch, batch_idx)
        else:
            loss = super().training_step(batch, batch_idx)
        
        # Log additional metrics
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Enhanced validation step with metrics tracking"""
        if self.use_mixed_precision:
            with autocast():
                loss = super().validation_step(batch, batch_idx)
        else:
            loss = super().validation_step(batch, batch_idx)
        
        return loss
    
    def validation_epoch_end(self, outputs):
        """Track validation metrics for hyperparameter optimization"""
        super().validation_epoch_end(outputs)
        
        # Get current validation loss
        val_loss = self.trainer.callback_metrics.get('val_loss', float('inf'))
        
        # Track metrics history
        self.val_metrics_history.append(val_loss)
        
        # Update best metric
        if val_loss < self.best_val_metric:
            self.best_val_metric = val_loss
        
        # Report to Optuna if available
        if hasattr(self.trainer, 'trial'):
            self.trainer.trial.report(val_loss, self.current_epoch)
            
            # Prune unpromising trials
            if self.trainer.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        super().on_train_epoch_end()
        
        # Update ensemble weights if enabled
        if self.ensemble_config.get('enable_ensemble', False):
            self._update_ensemble_weights()
    
    def _update_ensemble_weights(self):
        """Update ensemble weights using exponential moving average"""
        if self.ensemble_weights is None:
            # Initialize ensemble weights
            self.ensemble_weights = {}
            for name, param in self.named_parameters():
                self.ensemble_weights[name] = param.data.clone()
        else:
            # Update with exponential moving average
            alpha = self.ensemble_alpha
            for name, param in self.named_parameters():
                if name in self.ensemble_weights:
                    self.ensemble_weights[name] = (
                        alpha * self.ensemble_weights[name] + 
                        (1 - alpha) * param.data
                    )

class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna
    """
    
    def __init__(self, config: Dict, data_module: SpecsDataModule):
        self.config = config
        self.data_module = data_module
        self.best_trial = None
        self.best_score = float('inf')
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss to minimize
        """
        # Sample hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        
        # Model architecture parameters
        backbone = trial.suggest_categorical('backbone', ['ncsnpp', 'ncsnpp_v2', 'novel_dereverberation'])
        
        # Loss parameters
        loss_type = trial.suggest_categorical('loss_type', ['score_matching', 'data_prediction', 'denoiser'])
        
        # SDE parameters
        if 'sde_config' in self.config:
            sigma_min = trial.suggest_float('sigma_min', 0.05, 0.5)
            sigma_max = trial.suggest_float('sigma_max', 0.5, 2.0)
            N = trial.suggest_int('N', 30, 100)
        
        # Scheduler parameters
        scheduler_type = trial.suggest_categorical('scheduler_type', 
                                                 ['cosine_with_restarts', 'one_cycle', 'reduce_on_plateau'])
        
        # Update configuration
        trial_config = self.config.copy()
        trial_config.update({
            'lr': lr,
            'batch_size': batch_size,
            'backbone': backbone,
            'loss_type': loss_type,
        })
        
        if 'sde_config' in self.config:
            trial_config['sde_config'] = {
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'N': N
            }
        
        trial_config['scheduler_config'] = {
            'scheduler_type': scheduler_type
        }
        
        # Create model
        model = EnhancedScoreModel(
            backbone=backbone,
            sde=trial_config.get('sde', 'vesde'),
            lr=lr,
            loss_type=loss_type,
            data_module=self.data_module,
            scheduler_config=trial_config['scheduler_config'],
            **trial_config.get('model_config', {})
        )
        
        # Create trainer with pruning callback
        callbacks = [
            PyTorchLightningPruningCallback(trial, monitor='val_loss'),
            EarlyStopping(monitor='val_loss', patience=10, mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
        ]
        
        trainer = pl.Trainer(
            max_epochs=trial_config.get('max_epochs', 50),
            callbacks=callbacks,
            logger=False,  # Disable logging for optimization
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1
        )
        
        # Store trial reference for pruning
        trainer.trial = trial
        
        try:
            # Train model
            trainer.fit(model, self.data_module)
            
            # Return validation loss
            val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
            return val_loss
            
        except optuna.exceptions.TrialPruned:
            # Handle pruned trials
            raise
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')
    
    def optimize(self, n_trials: int = 100, study_name: str = 'dereverberation_optimization') -> Dict:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            study_name: Name of the optimization study
            
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Skipping hyperparameter optimization.")
            return self.config
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best parameters
        self.best_trial = study.best_trial
        self.best_score = study.best_value
        
        print(f"Best trial score: {self.best_score}")
        print(f"Best parameters: {self.best_trial.params}")
        
        # Update config with best parameters
        optimized_config = self.config.copy()
        optimized_config.update(self.best_trial.params)
        
        return optimized_config

class AdvancedTrainingPipeline:
    """
    Main training pipeline with all advanced features
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimized_config = None
        
    def setup_data_module(self) -> SpecsDataModule:
        """Setup data module"""
        return SpecsDataModule(
            base_dir=self.config['base_dir'],
            batch_size=self.config.get('batch_size', 32),
            n_fft=self.config.get('n_fft', 1024),
            hop_length=self.config.get('hop_length', 256),
            num_frames=self.config.get('num_frames', 256),
            **self.config.get('data_config', {})
        )
    
    def run_hyperparameter_optimization(self, data_module: SpecsDataModule) -> Dict:
        """Run hyperparameter optimization if enabled"""
        if self.config.get('enable_hp_optimization', False):
            print("Starting hyperparameter optimization...")
            optimizer = HyperparameterOptimizer(self.config, data_module)
            
            n_trials = self.config.get('hp_optimization_trials', 50)
            optimized_config = optimizer.optimize(n_trials=n_trials)
            
            print("Hyperparameter optimization completed.")
            return optimized_config
        
        return self.config
    
    def create_model(self, config: Dict, data_module: SpecsDataModule) -> EnhancedScoreModel:
        """Create enhanced model"""
        return EnhancedScoreModel(
            backbone=config.get('backbone', 'ncsnpp'),
            sde=config.get('sde', 'vesde'),
            lr=config.get('lr', 1e-4),
            loss_type=config.get('loss_type', 'score_matching'),
            data_module=data_module,
            gradient_clip_val=config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
            use_mixed_precision=config.get('use_mixed_precision', True),
            scheduler_config=config.get('scheduler_config', {}),
            ensemble_config=config.get('ensemble_config', {}),
            **config.get('model_config', {})
        )
    
    def setup_callbacks(self, config: Dict) -> List:
        """Setup training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.get('early_stopping_patience', 20),
                mode='min',
                verbose=True
            ),
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                filename='{epoch}-{val_loss:.4f}',
                save_last=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        return callbacks
    
    def setup_logger(self, config: Dict):
        """Setup logger"""
        if config.get('use_wandb', False):
            return WandbLogger(
                project=config.get('wandb_project', 'dereverberation'),
                name=config.get('experiment_name', 'enhanced_training'),
                config=config
            )
        else:
            return TensorBoardLogger(
                save_dir=config.get('log_dir', 'logs'),
                name=config.get('experiment_name', 'enhanced_training')
            )
    
    def train(self) -> Tuple[EnhancedScoreModel, Dict]:
        """
        Run complete training pipeline
        
        Returns:
            Tuple of (trained_model, training_results)
        """
        print("Setting up training pipeline...")
        
        # Setup data module
        data_module = self.setup_data_module()
        
        # Run hyperparameter optimization
        optimized_config = self.run_hyperparameter_optimization(data_module)
        self.optimized_config = optimized_config
        
        # Create model
        model = self.create_model(optimized_config, data_module)
        
        # Setup callbacks and logger
        callbacks = self.setup_callbacks(optimized_config)
        logger = self.setup_logger(optimized_config)
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=optimized_config.get('max_epochs', 100),
            callbacks=callbacks,
            logger=logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=optimized_config.get('num_gpus', 1),
            precision=16 if optimized_config.get('use_mixed_precision', True) else 32,
            gradient_clip_val=optimized_config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=optimized_config.get('accumulate_grad_batches', 1),
            enable_progress_bar=True,
            **optimized_config.get('trainer_config', {})
        )
        
        # Train model
        print("Starting training...")
        trainer.fit(model, data_module)
        
        # Prepare results
        results = {
            'best_val_loss': model.best_val_metric,
            'total_epochs': trainer.current_epoch,
            'optimized_config': optimized_config
        }
        
        if hasattr(self, 'best_trial') and self.best_trial:
            results['best_trial_params'] = self.best_trial.params
            results['best_trial_score'] = self.best_score
        
        print(f"Training completed. Best validation loss: {model.best_val_metric:.4f}")
        
        return model, results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Training with Hyperparameter Optimization")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training configuration JSON file")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing train/valid/test data")
    parser.add_argument("--output_dir", type=str, default="./enhanced_training_output",
                       help="Output directory for models and logs")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update with command line arguments
    config['base_dir'] = args.base_dir
    config['output_dir'] = args.output_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training pipeline
    pipeline = AdvancedTrainingPipeline(config)
    model, results = pipeline.train()
    
    # Save results
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training results saved to: {results_path}")
    
    # Save final model
    model_path = os.path.join(args.output_dir, 'final_model.ckpt')
    model.trainer.save_checkpoint(model_path)
    print(f"Final model saved to: {model_path}")

if __name__ == "__main__":
    main()