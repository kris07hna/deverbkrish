# Dereverberation Dataset Management

This document describes the comprehensive dataset management functionality added to SGMSE for dereverberation tasks.

## Overview

The dereverberation dataset management system provides:

1. **CSV-based Dataset Management** - Support for metadata-rich datasets
2. **Enhanced Data Module** - Extended functionality for reverb datasets
3. **Dataset Utilities** - Tools for creating, validating, and analyzing datasets
4. **Training Enhancements** - Specialized training for dereverberation
5. **Evaluation Tools** - Comprehensive metrics for dereverberation performance

## Components

### 1. ReverbDataModule (`sgmse/dereverb_data_module.py`)

Enhanced data module that extends the base `SpecsDataModule` with reverb-specific functionality.

**Key Features:**
- CSV-based dataset loading with metadata
- Support for T60, DRR, and room type filtering
- Compatible with existing directory-based formats
- Automatic metadata handling and validation

**Usage:**
```python
from sgmse.dereverb_data_module import ReverbDataModule

# CSV-based dataset
data_module = ReverbDataModule(
    csv_path="path/to/dataset.csv",
    filter_t60_min=0.3,
    filter_t60_max=2.0,
    filter_room_type=["office", "hall"],
    batch_size=8
)

# Directory-based dataset (compatible with existing format)
data_module = ReverbDataModule(
    base_dir="path/to/audio/",
    format="reverb",  # or "directory"
    batch_size=8
)
```

### 2. Dataset Utilities (`sgmse/util/dataset_utils.py`)

Comprehensive utilities for dataset management and analysis.

**Key Functions:**

#### `create_dataset_csv()`
Creates CSV datasets from directory structures:
```python
from sgmse.util.dataset_utils import create_dataset_csv

df = create_dataset_csv(
    audio_dir="path/to/audio/",
    output_csv="dataset.csv",
    clean_subdir="anechoic",
    reverb_subdir="reverb",
    extract_metadata=True
)
```

#### `validate_dataset_csv()`
Validates dataset CSV files:
```python
from sgmse.util.dataset_utils import validate_dataset_csv

results = validate_dataset_csv("dataset.csv", check_files=True)
if results['valid']:
    print("Dataset is valid!")
    print("Statistics:", results['statistics'])
```

#### `analyze_dataset_statistics()`
Analyzes and visualizes dataset statistics:
```python
from sgmse.util.dataset_utils import analyze_dataset_statistics

analysis = analyze_dataset_statistics(
    csv_path="dataset.csv",
    output_dir="analysis_results/"
)
```

### 3. ACE Corpus Dataset Creation (`preprocessing/create_ace_dataset.py`)

Script for creating structured datasets from ACE Corpus data.

**Usage:**
```bash
python preprocessing/create_ace_dataset.py \
    --ace_corpus_dir /path/to/ace_corpus/ \
    --clean_speech_dir /path/to/clean_speech/ \
    --output_dir /path/to/output/ \
    --sample_rate 16000 \
    --max_samples_per_room 100
```

**Features:**
- Automatic RIR convolution with clean speech
- T60 and DRR calculation
- Room type and acoustic parameter extraction
- Train/validation/test splitting

### 4. Dataset Analysis Tools (`preprocessing/analyze_reverb_dataset.py`)

Comprehensive analysis and visualization tools for reverb datasets.

**Usage:**
```bash
python preprocessing/analyze_reverb_dataset.py \
    --csv_path dataset.csv \
    --output_dir analysis_results/ \
    --analyze_audio
```

**Generated Analysis:**
- T60 and DRR distributions
- Room type analysis
- Correlation matrices
- Statistical summaries
- Visualization plots

### 5. Specialized Training (`train_dereverb.py`)

Enhanced training script with dereverberation-specific features.

**Usage:**
```bash
python train_dereverb.py \
    --csv_path dataset.csv \
    --backbone ncsnpp \
    --sde ouve \
    --batch_size 8 \
    --filter_t60_min 0.3 \
    --filter_t60_max 2.0 \
    --wandb_name "dereverberation_experiment"
```

**Features:**
- CSV dataset support with filtering
- Enhanced logging for reverb metrics
- Automatic dataset statistics logging
- Specialized learning rate scheduling

### 6. Evaluation Tools (`evaluate_dereverb.py`)

Comprehensive evaluation with dereverberation-specific metrics.

**Usage:**
```bash
python evaluate_dereverb.py \
    --clean_dir path/to/clean/ \
    --reverb_dir path/to/reverb/ \
    --enhanced_dir path/to/enhanced/ \
    --csv_path dataset.csv \
    --output_dir evaluation_results/
```

**Metrics Calculated:**
- Standard metrics: PESQ, ESTOI, SI-SDR
- Dereverberation metrics: T60 reduction, DRR improvement
- Room-specific analysis
- Comprehensive visualizations

## CSV Dataset Format

The CSV dataset format supports the following columns:

### Required Columns:
- `clean_path`: Path to clean audio file
- `reverb_path`: Path to reverberant audio file

### Optional Columns:
- `rir_path`: Path to room impulse response file
- `subset`: Dataset subset ("train", "valid", "test")
- `t60`: Reverberation time in seconds
- `drr`: Direct-to-reverberant ratio in dB
- `room_type`: Type of room ("office", "hall", "classroom", etc.)
- `room_size`: Size of room ("small", "medium", "large")
- `distance`: Source-microphone distance in meters
- `duration`: Audio duration in seconds
- `sample_rate`: Sample rate in Hz

### Example CSV:
```csv
clean_path,reverb_path,subset,t60,drr,room_type,room_size
/data/clean/speech1.wav,/data/reverb/speech1.wav,train,0.5,-5.2,office,medium
/data/clean/speech2.wav,/data/reverb/speech2.wav,train,1.2,-8.1,hall,large
```

## Workflow Examples

### 1. Create Dataset from ACE Corpus
```bash
# Step 1: Create dataset
python preprocessing/create_ace_dataset.py \
    --ace_corpus_dir /data/ace_corpus/ \
    --clean_speech_dir /data/wsj0/ \
    --output_dir /data/ace_dataset/

# Step 2: Analyze dataset
python preprocessing/analyze_reverb_dataset.py \
    --csv_path /data/ace_dataset/ace_dataset.csv \
    --output_dir /data/ace_dataset/analysis/

# Step 3: Train model
python train_dereverb.py \
    --csv_path /data/ace_dataset/ace_dataset.csv \
    --backbone ncsnpp \
    --batch_size 8
```

### 2. Convert Existing Directory Dataset to CSV
```python
from sgmse.util.dataset_utils import create_dataset_csv

# Convert existing directory structure to CSV
df = create_dataset_csv(
    audio_dir="/data/existing_dataset/",
    output_csv="/data/existing_dataset/dataset.csv",
    clean_subdir="anechoic",
    reverb_subdir="reverb"
)

print(f"Created CSV with {len(df)} samples")
```

### 3. Filter Dataset by Acoustic Properties
```python
from sgmse.util.dataset_utils import filter_dataset_csv

# Filter dataset for specific T60 range
filtered_df = filter_dataset_csv(
    csv_path="original_dataset.csv",
    output_path="filtered_dataset.csv",
    t60={'min': 0.3, 'max': 1.5},
    room_type=['office', 'classroom']
)
```

## Integration with Existing SGMSE

The dereverberation components are designed to be fully compatible with existing SGMSE functionality:

1. **Backward Compatibility**: Existing directory-based datasets work without changes
2. **Extensible Design**: New features can be added without breaking existing code  
3. **Modular Architecture**: Components can be used independently or together
4. **Consistent API**: Follows SGMSE patterns and conventions

## Performance Considerations

- **Memory Usage**: CSV datasets support lazy loading for large datasets
- **I/O Optimization**: Efficient file handling for large-scale training
- **Parallel Processing**: Multi-worker support for data loading
- **Caching**: Optional metadata caching for improved performance

## Best Practices

1. **Dataset Organization**: Use consistent naming conventions for audio files
2. **Metadata Quality**: Include accurate T60, DRR, and room information when available
3. **Validation**: Always validate datasets before training
4. **Analysis**: Analyze dataset statistics to understand data distribution
5. **Filtering**: Use filtering to create focused training sets for specific conditions

## Troubleshooting

### Common Issues:

1. **Missing Files**: Use `validate_dataset_csv()` to check for missing audio files
2. **Metadata Extraction**: Check filename patterns for automatic metadata extraction
3. **Memory Issues**: Reduce batch size or use dummy mode for testing
4. **Path Issues**: Use absolute paths in CSV files for reliability

### Debug Mode:
```python
# Use dummy mode for quick testing
data_module = ReverbDataModule(
    csv_path="dataset.csv",
    dummy=True,  # Uses only small subset
    batch_size=2
)
```

## Future Extensions

The architecture supports future enhancements such as:
- Additional acoustic metrics (EDT, C50, C80)
- Multi-microphone dataset support
- Real-time dataset augmentation
- Advanced filtering and stratification
- Integration with other reverb datasets (WHAM!, LibriReVerb, etc.)