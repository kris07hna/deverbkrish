"""
Enhanced SGMSE+ Hackathon Solution
=================================

This package implements a comprehensive, SOTA de-reverberation solution for hackathon competitions.

Key Components:
- Enhanced SGMSE+ model with novel architectural improvements
- Advanced data augmentation pipeline
- Model ensembling and test-time augmentation
- Comprehensive evaluation framework
- Complete training and inference scripts

Author: SOTA Hackathon Team
"""

__version__ = "1.0.0"
__author__ = "SOTA Hackathon Team"

# Import main components
from .enhanced_model import (
    EnhancedScoreModel,
    MultiScaleAttentionBlock,
    FrequencyAwareProcessor,
    AdaptiveLossWeighting,
    SpectralConsistencyLoss
)

from .advanced_data_augmentation import (
    AdvancedDataAugmentation,
    SpectralAugmentation,
    DynamicRangeAugmentation,
    RIRSimulation,
    AdaptiveNoiseAugmentation,
    SpectralMorphing
)

from .ensemble_inference import (
    EnsembleInference,
    TestTimeAugmentation,
    UncertaintyEstimator,
    FrequencyBandEnsemble,
    ProgressiveDenoising
)

from .evaluation_framework import (
    ComprehensiveEvaluator,
    StandardMetrics,
    PerceptualMetrics,
    ReverbSpecificMetrics,
    ComputationalMetrics
)

__all__ = [
    # Enhanced Model Components
    'EnhancedScoreModel',
    'MultiScaleAttentionBlock',
    'FrequencyAwareProcessor',
    'AdaptiveLossWeighting',
    'SpectralConsistencyLoss',
    
    # Data Augmentation Components
    'AdvancedDataAugmentation',
    'SpectralAugmentation',
    'DynamicRangeAugmentation',
    'RIRSimulation',
    'AdaptiveNoiseAugmentation',
    'SpectralMorphing',
    
    # Ensemble Inference Components
    'EnsembleInference',
    'TestTimeAugmentation',
    'UncertaintyEstimator',
    'FrequencyBandEnsemble',
    'ProgressiveDenoising',
    
    # Evaluation Framework Components
    'ComprehensiveEvaluator',
    'StandardMetrics',
    'PerceptualMetrics',
    'ReverbSpecificMetrics',
    'ComputationalMetrics'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'enhanced_sgmse_hackathon',
    'version': __version__,
    'description': 'SOTA De-reverberation Solution for Hackathon Competitions',
    'author': __author__,
    'features': [
        'Adaptive Loss Weighting',
        'Multi-Scale Feature Fusion',
        'Spectral Consistency Loss',
        'Progressive Training',
        'Test-Time Augmentation',
        'Model Ensembling',
        'Frequency-Aware Processing',
        'Advanced Data Augmentation'
    ],
    'performance_targets': {
        'PESQ': 3.5,
        'STOI': 0.92,
        'SI-SDR': 18.0,
        'Efficiency': '<10 GMAC/s'
    }
}

def get_package_info():
    """Get package information"""
    return PACKAGE_INFO

def print_welcome():
    """Print welcome message with package info"""
    info = get_package_info()
    
    print("🎯 " + "="*60)
    print(f"🏆 {info['description']}")
    print("🎯 " + "="*60)
    print(f"📦 Package: {info['name']} v{info['version']}")
    print(f"👥 Author: {info['author']}")
    print("\n🚀 Novel Features:")
    for feature in info['features']:
        print(f"  ✅ {feature}")
    print("\n📊 Performance Targets:")
    for metric, target in info['performance_targets'].items():
        print(f"  🎯 {metric}: {target}")
    print("🎯 " + "="*60)
    print("🎉 Ready for SOTA De-reverberation! 🎉")
    print("🎯 " + "="*60)