#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path


def load_feature_file(filepath):
    """Load saved features from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_activation_distribution(activations, layer_name, save_dir):
    """Plot distribution of activation values"""
    activation = activations[0].numpy()  # [C, H, W]
    
    # Flatten all values
    values = activation.flatten()
    
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(1, 3, 1)
    plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{layer_name} - Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    
    # Box plot
    plt.subplot(1, 3, 2)
    plt.boxplot(values)
    plt.title(f'{layer_name} - Box Plot')
    plt.ylabel('Activation Value')
    
    # Cumulative distribution
    plt.subplot(1, 3, 3)
    sorted_values = np.sort(values)
    y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    plt.plot(sorted_values, y)
    plt.title(f'{layer_name} - Cumulative Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Cumulative Probability')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{layer_name}_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'sparsity': np.mean(values == 0),
        'num_channels': activation.shape[0],
        'spatial_size': activation.shape[1:]
    }


def plot_channel_statistics(activations, layer_name, save_dir):
    """Plot per-channel statistics"""
    activation = activations[0].numpy()  # [C, H, W]
    
    # Calculate per-channel statistics
    channel_means = np.mean(activation, axis=(1, 2))
    channel_stds = np.std(activation, axis=(1, 2))
    channel_maxs = np.max(activation, axis=(1, 2))
    channel_sparsity = np.mean(activation == 0, axis=(1, 2))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Channel means
    axes[0, 0].plot(channel_means)
    axes[0, 0].set_title('Channel Means')
    axes[0, 0].set_xlabel('Channel Index')
    axes[0, 0].set_ylabel('Mean Activation')
    
    # Channel standard deviations
    axes[0, 1].plot(channel_stds)
    axes[0, 1].set_title('Channel Standard Deviations')
    axes[0, 1].set_xlabel('Channel Index')
    axes[0, 1].set_ylabel('Std Activation')
    
    # Channel maximums
    axes[1, 0].plot(channel_maxs)
    axes[1, 0].set_title('Channel Maximums')
    axes[1, 0].set_xlabel('Channel Index')
    axes[1, 0].set_ylabel('Max Activation')
    
    # Channel sparsity
    axes[1, 1].plot(channel_sparsity)
    axes[1, 1].set_title('Channel Sparsity')
    axes[1, 1].set_xlabel('Channel Index')
    axes[1, 1].set_ylabel('Fraction of Zeros')
    
    plt.suptitle(f'{layer_name} - Per-Channel Statistics')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{layer_name}_channel_stats.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_spatial_activation_patterns(activations, layer_name, save_dir, num_channels=9):
    """Plot spatial patterns of top activated channels"""
    activation = activations[0].numpy()  # [C, H, W]
    
    # Find channels with highest mean activation
    channel_means = np.mean(activation, axis=(1, 2))
    top_channels = np.argsort(channel_means)[-num_channels:]
    
    # Create subplot grid
    cols = 3
    rows = (num_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    if rows == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(top_channels):
        if i >= len(axes):
            break
            
        feature_map = activation[channel_idx]
        
        im = axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Ch {channel_idx} (mean: {channel_means[channel_idx]:.3f})')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(len(top_channels), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{layer_name} - Top Activated Channels')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{layer_name}_spatial_patterns.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_layers(feature_data, save_dir):
    """Compare statistics across different layers"""
    layer_stats = {}
    
    for layer_name, activations in feature_data['activations'].items():
        if activations:
            activation = activations[0].numpy()
            values = activation.flatten()
            
            layer_stats[layer_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'sparsity': np.mean(values == 0),
                'num_params': activation.size,
                'num_channels': activation.shape[0] if len(activation.shape) > 2 else 1,
                'spatial_size': activation.shape[1:] if len(activation.shape) > 2 else activation.shape
            }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    layers = list(layer_stats.keys())
    means = [layer_stats[l]['mean'] for l in layers]
    stds = [layer_stats[l]['std'] for l in layers]
    sparsities = [layer_stats[l]['sparsity'] for l in layers]
    num_channels = [layer_stats[l]['num_channels'] for l in layers]
    
    # Mean activations across layers
    axes[0, 0].bar(range(len(layers)), means)
    axes[0, 0].set_title('Mean Activation by Layer')
    axes[0, 0].set_xticks(range(len(layers)))
    axes[0, 0].set_xticklabels(layers, rotation=45)
    axes[0, 0].set_ylabel('Mean Activation')
    
    # Standard deviation across layers
    axes[0, 1].bar(range(len(layers)), stds)
    axes[0, 1].set_title('Activation Std by Layer')
    axes[0, 1].set_xticks(range(len(layers)))
    axes[0, 1].set_xticklabels(layers, rotation=45)
    axes[0, 1].set_ylabel('Std Activation')
    
    # Sparsity across layers
    axes[1, 0].bar(range(len(layers)), sparsities)
    axes[1, 0].set_title('Sparsity by Layer')
    axes[1, 0].set_xticks(range(len(layers)))
    axes[1, 0].set_xticklabels(layers, rotation=45)
    axes[1, 0].set_ylabel('Fraction of Zeros')
    
    # Number of channels
    axes[1, 1].bar(range(len(layers)), num_channels)
    axes[1, 1].set_title('Number of Channels by Layer')
    axes[1, 1].set_xticks(range(len(layers)))
    axes[1, 1].set_xticklabels(layers, rotation=45)
    axes[1, 1].set_ylabel('Number of Channels')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'layer_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\nLayer Statistics Summary:")
    print("-" * 80)
    print(f"{'Layer':<15} {'Shape':<15} {'Mean':<8} {'Std':<8} {'Sparsity':<10} {'Channels':<8}")
    print("-" * 80)
    
    for layer_name in layers:
        stats = layer_stats[layer_name]
        shape_str = str(stats['spatial_size'])
        print(f"{layer_name:<15} {shape_str:<15} {stats['mean']:<8.3f} {stats['std']:<8.3f} "
              f"{stats['sparsity']:<10.1%} {stats['num_channels']:<8}")


def main():
    parser = argparse.ArgumentParser(description='Analyze saved feature maps')
    parser.add_argument('feature_file', help='Path to saved feature file (.pkl)')
    parser.add_argument('--save_dir', default='feature_analysis', help='Directory to save analysis plots')
    parser.add_argument('--layers', nargs='+', help='Specific layers to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.feature_file}...")
    feature_data = load_feature_file(args.feature_file)
    
    # Print basic info
    print(f"Batch index: {feature_data['batch_idx']}")
    print(f"Timestep: {feature_data['timestep']}")
    if feature_data['batch_info']:
        print(f"Batch info: {feature_data['batch_info']}")
    
    activations = feature_data['activations']
    layers_to_analyze = args.layers if args.layers else list(activations.keys())
    
    print(f"\nAnalyzing layers: {layers_to_analyze}")
    
    # Analyze each layer
    for layer_name in layers_to_analyze:
        if layer_name in activations and activations[layer_name]:
            print(f"\nAnalyzing {layer_name}...")
            
            # Distribution analysis
            stats = plot_activation_distribution(activations[layer_name], layer_name, args.save_dir)
            print(f"  Shape: {stats['spatial_size']}, Channels: {stats['num_channels']}")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Sparsity: {stats['sparsity']:.1%}")
            
            # Channel-wise analysis
            if len(activations[layer_name][0].shape) > 2:  # Has spatial dimensions
                plot_channel_statistics(activations[layer_name], layer_name, args.save_dir)
                plot_spatial_activation_patterns(activations[layer_name], layer_name, args.save_dir)
        else:
            print(f"No activations found for layer {layer_name}")
    
    # Compare across layers
    print("\nGenerating layer comparison...")
    compare_layers(feature_data, args.save_dir)
    
    print(f"\nAnalysis complete! Check {args.save_dir} for generated plots.")


if __name__ == '__main__':
    main()