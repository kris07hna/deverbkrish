"""
Model Complexity Monitor

This module provides comprehensive model complexity analysis and monitoring
to ensure compliance with computational constraints (< 50 GMAC/s).
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

class ComplexityProfiler:
    """
    Profiles model complexity including FLOPs, memory usage, and inference time
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()
    
    def reset(self):
        """Reset profiling statistics"""
        self.flop_counts = defaultdict(int)
        self.memory_usage = {}
        self.timing_stats = {}
        self.layer_stats = {}
    
    def profile_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                     num_runs: int = 10, warmup_runs: int = 3) -> Dict:
        """
        Profile model complexity comprehensively
        
        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape (B, C, H, W) or (B, C, T)
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with complexity metrics
        """
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Profile FLOPs
        flops = self._profile_flops(model, dummy_input)
        
        # Profile memory usage
        memory_stats = self._profile_memory(model, dummy_input)
        
        # Profile inference time
        timing_stats = self._profile_timing(model, dummy_input, num_runs, warmup_runs)
        
        # Calculate GMAC/s
        gmacs = flops / 1e9
        inference_time = timing_stats['avg_time']
        gmacs_per_second = gmacs / inference_time if inference_time > 0 else float('inf')
        
        # Profile per-layer statistics
        layer_stats = self._profile_layers(model, dummy_input)
        
        return {
            'total_flops': flops,
            'gmacs': gmacs,
            'gmacs_per_second': gmacs_per_second,
            'memory_stats': memory_stats,
            'timing_stats': timing_stats,
            'layer_stats': layer_stats,
            'input_shape': input_shape,
            'device': str(self.device),
            'within_constraint': gmacs_per_second < 50.0
        }
    
    def _profile_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> int:
        """Profile FLOPs using hooks"""
        
        def conv2d_flop_count(input_shape, weight_shape, bias_shape=None):
            """Calculate FLOPs for Conv2d"""
            batch_size, in_channels, input_height, input_width = input_shape
            out_channels, in_channels, kernel_height, kernel_width = weight_shape
            
            # Calculate output dimensions (assuming padding preserves dimensions)
            output_height = input_height
            output_width = input_width
            
            # Multiply-accumulate operations
            kernel_flops = kernel_height * kernel_width * in_channels
            output_elements = batch_size * out_channels * output_height * output_width
            flops = kernel_flops * output_elements
            
            # Add bias operations if present
            if bias_shape is not None:
                flops += output_elements
            
            return flops
        
        def linear_flop_count(input_shape, weight_shape, bias_shape=None):
            """Calculate FLOPs for Linear layer"""
            batch_size = input_shape[0]
            in_features = weight_shape[1]
            out_features = weight_shape[0]
            
            # Multiply-accumulate operations
            flops = batch_size * in_features * out_features
            
            # Add bias operations if present
            if bias_shape is not None:
                flops += batch_size * out_features
            
            return flops
        
        def attention_flop_count(seq_len, d_model, num_heads):
            """Calculate FLOPs for attention mechanism"""
            # Q, K, V projections
            flops = 3 * seq_len * d_model * d_model
            
            # Attention computation
            flops += num_heads * seq_len * seq_len * (d_model // num_heads)
            
            # Output projection
            flops += seq_len * d_model * d_model
            
            return flops
        
        total_flops = 0
        
        def flop_hook(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv2d):
                if len(input) > 0 and len(output) > 0:
                    flops = conv2d_flop_count(
                        input[0].shape,
                        module.weight.shape,
                        module.bias.shape if module.bias is not None else None
                    )
                    total_flops += flops
            
            elif isinstance(module, nn.Linear):
                if len(input) > 0:
                    flops = linear_flop_count(
                        input[0].shape,
                        module.weight.shape,
                        module.bias.shape if module.bias is not None else None
                    )
                    total_flops += flops
            
            elif isinstance(module, nn.MultiheadAttention):
                if len(input) > 0:
                    seq_len = input[0].shape[0]
                    d_model = module.embed_dim
                    num_heads = module.num_heads
                    flops = attention_flop_count(seq_len, d_model, num_heads)
                    total_flops += flops
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
                hooks.append(module.register_forward_hook(flop_hook))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops
    
    def _profile_memory(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict:
        """Profile memory usage"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Measure memory before
            mem_before = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                output = model(input_tensor)
                torch.cuda.synchronize()
                
                # Measure memory after
                mem_after = torch.cuda.memory_allocated()
                
                # Calculate peak memory
                peak_memory = torch.cuda.max_memory_allocated()
                
            torch.cuda.empty_cache()
            
            return {
                'memory_before_mb': mem_before / 1024 / 1024,
                'memory_after_mb': mem_after / 1024 / 1024,
                'memory_used_mb': (mem_after - mem_before) / 1024 / 1024,
                'peak_memory_mb': peak_memory / 1024 / 1024
            }
        else:
            # CPU memory profiling is more complex, return placeholder
            return {
                'memory_before_mb': 0,
                'memory_after_mb': 0,
                'memory_used_mb': 0,
                'peak_memory_mb': 0
            }
    
    def _profile_timing(self, model: nn.Module, input_tensor: torch.Tensor,
                       num_runs: int, warmup_runs: int) -> Dict:
        """Profile inference timing"""
        times = []
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(input_tensor)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
        
        # Actual timing runs
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(input_tensor)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'all_times': times
        }
    
    def _profile_layers(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict:
        """Profile individual layer statistics"""
        layer_stats = {}
        
        def layer_hook(name):
            def hook(module, input, output):
                # Calculate layer-specific metrics
                if hasattr(module, 'weight') and module.weight is not None:
                    params = module.weight.numel()
                    if hasattr(module, 'bias') and module.bias is not None:
                        params += module.bias.numel()
                else:
                    params = sum(p.numel() for p in module.parameters())
                
                layer_stats[name] = {
                    'parameters': params,
                    'input_shape': input[0].shape if len(input) > 0 else None,
                    'output_shape': output.shape if torch.is_tensor(output) else None
                }
            return hook
        
        # Register hooks for named modules
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(layer_hook(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_stats

class ComplexityOptimizer:
    """
    Optimize model to meet complexity constraints
    """
    
    def __init__(self, target_gmacs_per_second: float = 50.0):
        self.target_gmacs_per_second = target_gmacs_per_second
        self.profiler = ComplexityProfiler()
    
    def suggest_optimizations(self, complexity_profile: Dict) -> List[str]:
        """
        Suggest optimizations to reduce model complexity
        
        Args:
            complexity_profile: Output from ComplexityProfiler
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        current_gmacs_per_second = complexity_profile['gmacs_per_second']
        
        if current_gmacs_per_second <= self.target_gmacs_per_second:
            suggestions.append("✅ Model already meets complexity constraint!")
            return suggestions
        
        excess_factor = current_gmacs_per_second / self.target_gmacs_per_second
        
        suggestions.append(f"🔍 Current: {current_gmacs_per_second:.2f} GMAC/s")
        suggestions.append(f"🎯 Target: {self.target_gmacs_per_second:.2f} GMAC/s")
        suggestions.append(f"📈 Reduction needed: {excess_factor:.2f}x")
        suggestions.append("")
        
        # Architecture suggestions
        if excess_factor > 2.0:
            suggestions.append("🏗️ ARCHITECTURE CHANGES:")
            suggestions.append("  • Reduce model depth (fewer layers)")
            suggestions.append("  • Reduce channel dimensions")
            suggestions.append("  • Use depthwise separable convolutions")
            suggestions.append("  • Implement model distillation")
            suggestions.append("")
        
        # Optimization suggestions
        suggestions.append("⚡ OPTIMIZATION TECHNIQUES:")
        suggestions.append("  • Enable mixed precision training (FP16)")
        suggestions.append("  • Use gradient checkpointing")
        suggestions.append("  • Implement layer pruning")
        suggestions.append("  • Use knowledge distillation")
        suggestions.append("")
        
        # Inference optimizations
        suggestions.append("🚀 INFERENCE OPTIMIZATIONS:")
        suggestions.append("  • Reduce sampling steps for diffusion")
        suggestions.append("  • Use tensor compilation (torch.compile)")
        suggestions.append("  • Implement dynamic batching")
        suggestions.append("  • Use TensorRT or ONNX optimization")
        suggestions.append("")
        
        # Layer-specific suggestions
        layer_stats = complexity_profile.get('layer_stats', {})
        if layer_stats:
            # Find most parameter-heavy layers
            layer_params = [(name, stats['parameters']) for name, stats in layer_stats.items()]
            layer_params.sort(key=lambda x: x[1], reverse=True)
            
            suggestions.append("🎯 TOP PARAMETER-HEAVY LAYERS:")
            for name, params in layer_params[:5]:
                suggestions.append(f"  • {name}: {params:,} parameters")
            suggestions.append("  Consider reducing these layers first")
            suggestions.append("")
        
        return suggestions
    
    def create_optimized_config(self, current_config: Dict, 
                              complexity_profile: Dict) -> Dict:
        """
        Create optimized configuration to meet complexity constraints
        
        Args:
            current_config: Current model configuration
            complexity_profile: Complexity profiling results
            
        Returns:
            Optimized configuration
        """
        optimized_config = current_config.copy()
        current_gmacs_per_second = complexity_profile['gmacs_per_second']
        
        if current_gmacs_per_second <= self.target_gmacs_per_second:
            return optimized_config
        
        excess_factor = current_gmacs_per_second / self.target_gmacs_per_second
        reduction_factor = 1.0 / excess_factor
        
        # Scale down model dimensions
        if 'hidden_channels' in optimized_config:
            optimized_config['hidden_channels'] = max(
                32, int(optimized_config['hidden_channels'] * reduction_factor ** 0.5)
            )
        
        if 'num_layers' in optimized_config:
            optimized_config['num_layers'] = max(
                2, int(optimized_config['num_layers'] * reduction_factor ** 0.3)
            )
        
        if 'time_embedding_dim' in optimized_config:
            optimized_config['time_embedding_dim'] = max(
                64, int(optimized_config['time_embedding_dim'] * reduction_factor ** 0.4)
            )
        
        # Adjust sampling parameters
        if 'N' in optimized_config:
            optimized_config['N'] = max(20, int(optimized_config['N'] * 0.8))
        
        # Enable optimizations
        optimized_config['use_mixed_precision'] = True
        optimized_config['gradient_checkpointing'] = True
        
        return optimized_config

def monitor_training_complexity(model: nn.Module, input_shape: Tuple[int, ...],
                              target_gmacs_per_second: float = 50.0) -> Dict:
    """
    Monitor model complexity during training
    
    Args:
        model: Model to monitor
        input_shape: Input tensor shape
        target_gmacs_per_second: Target complexity limit
        
    Returns:
        Monitoring results with recommendations
    """
    profiler = ComplexityProfiler()
    optimizer = ComplexityOptimizer(target_gmacs_per_second)
    
    # Profile model
    complexity_profile = profiler.profile_model(model, input_shape)
    
    # Get optimization suggestions
    suggestions = optimizer.suggest_optimizations(complexity_profile)
    
    # Prepare monitoring report
    report = {
        'complexity_profile': complexity_profile,
        'within_constraint': complexity_profile['within_constraint'],
        'suggestions': suggestions,
        'timestamp': time.time()
    }
    
    return report

def print_complexity_report(report: Dict):
    """Print formatted complexity report"""
    profile = report['complexity_profile']
    
    print("=" * 60)
    print("🔬 MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    print(f"📊 Total FLOPs: {profile['total_flops']:,}")
    print(f"⚡ GMACs: {profile['gmacs']:.2f}")
    print(f"🚀 GMAC/s: {profile['gmacs_per_second']:.2f}")
    print(f"⏱️  Avg inference time: {profile['timing_stats']['avg_time']:.4f}s")
    print(f"💾 Memory used: {profile['memory_stats']['memory_used_mb']:.2f} MB")
    print(f"🎯 Within constraint: {'✅ YES' if profile['within_constraint'] else '❌ NO'}")
    print()
    
    if report['suggestions']:
        print("💡 OPTIMIZATION SUGGESTIONS:")
        print("-" * 40)
        for suggestion in report['suggestions']:
            print(suggestion)
    
    print("=" * 60)

if __name__ == "__main__":
    # Example usage
    from sgmse.model import ScoreModel
    
    # Create a dummy model for testing
    model = ScoreModel(backbone='ncsnpp_v2')
    input_shape = (1, 1, 513, 256)  # Batch size 1, 1 channel, freq bins, time frames
    
    # Monitor complexity
    report = monitor_training_complexity(model, input_shape)
    print_complexity_report(report)