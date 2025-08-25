# Enhanced Dereverberation System for Kaggle Competition

This enhanced version of the SGMSE (Score-based Generative Models for Speech Enhancement) repository includes advanced features specifically designed for Kaggle dereverberation competitions and real-world applications.

## 🎯 Key Features

### 1. **Kaggle Competition Ready**
- **Standardized submission format** with automated CSV generation
- **Model complexity monitoring** to ensure < 50 GMAC/s constraint
- **Dual evaluation metrics**: PESQ for speech, SDR for music
- **Automated content type detection** (speech vs music)

### 2. **Novel Architecture Improvements**
- **Multi-Scale Attention Mechanism** for better frequency processing
- **Frequency-Aware Cross-Attention** with positional encoding
- **Adaptive Spectral Gating** for frequency-specific enhancement
- **Room Impulse Response (RIR) Estimation Module**
- **Perceptual Loss Integration** for better audio quality

### 3. **Advanced Training Pipeline**
- **Automated hyperparameter optimization** using Optuna
- **Ensemble training** with exponential moving averages
- **Mixed precision training** for efficiency
- **Advanced learning rate scheduling** (Cosine with restarts, OneCycle, etc.)
- **Gradient accumulation** and checkpointing

### 4. **Sophisticated Data Augmentation**
- **Synthetic reverb generation** with physically-inspired models
- **Multi-room acoustic simulation** 
- **Frequency-domain augmentations** (SpecAugment-style)
- **Dynamic range manipulation**
- **Adversarial augmentation** for robustness

### 5. **Comprehensive Evaluation**
- **Advanced metrics calculation** beyond basic PESQ/SDR
- **Content-type specific evaluation**
- **Model complexity profiling**
- **Real-time performance monitoring**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kris07hna/deverbkrish.git
cd deverbkrish

# Install dependencies
pip install -r requirements.txt

# Install additional packages for enhanced features
pip install optuna torchinfo wandb
```

### Basic Usage

#### 1. Kaggle Submission

```bash
# Run inference with ensemble models
python kaggle_submission.py \
    --input_dir /path/to/test/audio \
    --output_dir /path/to/enhanced/output \
    --config configs/kaggle_configs.json \
    --submission_csv submission.csv
```

#### 2. Enhanced Training

```bash
# Train with hyperparameter optimization
python enhanced_train.py \
    --config configs/enhanced_training_configs.json \
    --base_dir /path/to/training/data \
    --output_dir ./training_output
```

#### 3. Advanced Metrics Calculation

```bash
# Calculate comprehensive metrics
python calc_metrics.py \
    --clean_dir /path/to/clean/audio \
    --enhanced_dir /path/to/enhanced/audio \
    --noisy_dir /path/to/noisy/audio \
    --use_advanced \
    --kaggle_format
```

## 📁 Project Structure

```
deverbkrish/
├── configs/
│   ├── kaggle_configs.json              # Kaggle submission configurations
│   └── enhanced_training_configs.json   # Training configurations
├── sgmse/
│   ├── backbones/
│   │   └── novel_dereverberation.py     # Novel architecture
│   └── util/
│       ├── advanced_augmentation.py     # Advanced data augmentation
│       └── complexity_monitor.py        # Model complexity monitoring
├── kaggle_submission.py                 # Main Kaggle submission script
├── enhanced_train.py                    # Enhanced training pipeline
├── advanced_metrics.py                  # Advanced metrics calculation
├── calc_metrics.py                      # Updated metrics script
└── README_ENHANCED.md                   # This file
```

## 🏆 Competition Strategy

### 1. **Model Architecture**
- Use the novel dereverberation architecture with multi-scale attention
- Ensure model complexity stays under 50 GMAC/s
- Employ ensemble of 3-5 models with different configurations

### 2. **Training Strategy**
- Start with hyperparameter optimization on a subset of data
- Use advanced augmentation during training
- Implement ensemble training with EMA
- Monitor complexity throughout training

### 3. **Inference Strategy**
- Use adaptive sampling based on input characteristics
- Implement ensemble inference with weighted averaging
- Apply post-processing if beneficial

### 4. **Evaluation Strategy**
- Separate evaluation for speech (PESQ) and music (SDR)
- Use content-type detection for automatic routing
- Monitor real-time performance

## 🛠️ Configuration Guide

### Kaggle Submission Configuration

```json
{
  "model_paths": [
    "checkpoints/ensemble_model_1.ckpt",
    "checkpoints/ensemble_model_2.ckpt",
    "checkpoints/ensemble_model_3.ckpt"
  ],
  "ensemble_weights": [0.4, 0.35, 0.25],
  "max_complexity_gmac": 49.5,
  "sampling_strategy": "adaptive_advanced",
  "adaptive_sampling": {
    "base_steps": 70,
    "max_steps": 150,
    "min_steps": 30,
    "dynamic_adjustment": true
  }
}
```

### Training Configuration

```json
{
  "enable_hp_optimization": true,
  "hp_optimization_trials": 100,
  "backbone": "novel_dereverberation",
  "use_mixed_precision": true,
  "ensemble_config": {
    "enable_ensemble": true,
    "alpha": 0.999
  },
  "scheduler_config": {
    "scheduler_type": "cosine_with_restarts",
    "T_0": 20,
    "T_mult": 2
  }
}
```

## 📊 Performance Optimization

### Model Complexity Management

```python
from sgmse.util.complexity_monitor import monitor_training_complexity

# Monitor model complexity
report = monitor_training_complexity(model, input_shape=(1, 1, 513, 256))
print_complexity_report(report)
```

### Hyperparameter Optimization

```python
from enhanced_train import HyperparameterOptimizer

# Run hyperparameter optimization
optimizer = HyperparameterOptimizer(config, data_module)
best_config = optimizer.optimize(n_trials=50)
```

## 🎵 Content-Specific Processing

The system automatically detects whether input audio is speech or music and applies appropriate processing:

- **Speech**: Optimized for PESQ scores, uses speech-specific augmentations
- **Music**: Optimized for SDR scores, preserves harmonic content

## 📈 Advanced Metrics

Beyond standard metrics, the system calculates:

- **SI-SDR**: Scale-Invariant Source-to-Distortion Ratio
- **Spectral Convergence**: Frequency domain similarity
- **Log-Spectral Distance**: Perceptual quality measure
- **Content-type Detection**: Automatic speech/music classification

## 🔧 Troubleshooting

### Common Issues

1. **Model Complexity Too High**
   ```bash
   # Use complexity monitor to get optimization suggestions
   python -c "from sgmse.util.complexity_monitor import *; 
              report = monitor_training_complexity(model, input_shape);
              print_complexity_report(report)"
   ```

2. **Training Memory Issues**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Slow Inference**
   - Reduce sampling steps
   - Use ensemble weights optimization
   - Enable tensor compilation

## 📝 Citation

If you use this enhanced system, please cite the original SGMSE papers and mention the enhancements:

```bibtex
@misc{enhanced_sgmse_2024,
  title={Enhanced SGMSE for Kaggle Dereverberation Competition},
  author={Enhanced by Advanced AI Systems},
  year={2024},
  note={Based on SGMSE by Richter et al.}
}
```

## 🤝 Contributing

This enhanced system builds upon the excellent foundation of SGMSE. Contributions for further improvements are welcome:

1. Novel architectural components
2. Advanced training techniques
3. Improved evaluation metrics
4. Competition-specific optimizations

## 📧 Support

For issues specific to the enhanced features, please create an issue with the `enhancement` label.

---

**Note**: This enhanced system is specifically designed for Kaggle dereverberation competitions while maintaining compatibility with the original SGMSE framework. All enhancements are modular and can be enabled/disabled as needed.