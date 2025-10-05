# Feature Map Inspection Guide

This guide explains how to extract, analyze, and visualize feature maps from trained CNN models, specifically for video-based action recognition networks with temporal sequences.

## Overview

The feature inspection toolkit consists of three main components:

1. **`feature_inspector.py`** - Core inspection class with hooks and extraction logic
2. **`inspect_features.py`** - Main script to extract features from trained models
3. **`analyze_features.py`** - Analysis and visualization tool for saved features

## Table of Contents

- [Installation Requirements](#installation-requirements)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Understanding the Output](#understanding-the-output)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Code Integration](#code-integration)

## Installation Requirements

Ensure you have the following Python packages installed:

```bash
pip install torch torchvision matplotlib seaborn numpy pickle-tools
```

## Quick Start

### 1. Extract Features from a Trained Model

```bash
cd /home/gregz9/Desktop/MasterThesis/src/interaction_hotspots

# Basic feature extraction
python inspect_features.py \
    --checkpoint /path/to/your/trained_model.pth \
    --dset Robofarmer-II \
    --batch_size 4 \
    --size 512 \
    --save_dir my_features
```

### 2. Analyze the Extracted Features

```bash
# Analyze saved features
python analyze_features.py my_features/features_b0_t0_Robofarmer-II_512.pkl \
    --save_dir feature_analysis
```

## Detailed Usage

### Feature Extraction (`inspect_features.py`)

#### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | ✅ | - | Path to trained model checkpoint (.pth file) |
| `--dset` | ❌ | `Robofarmer-II` | Dataset name |
| `--batch_size` | ❌ | `2` | Batch size for data loading |
| `--max_len` | ❌ | `8` | Maximum sequence length |
| `--size` | ❌ | `512` | Input image resolution |
| `--workers` | ❌ | `1` | Number of data loading workers |
| `--save_dir` | ❌ | `feature_maps` | Directory to save extracted features |
| `--batch_idx` | ❌ | `0` | Which sample in batch to inspect (0-indexed) |
| `--timestep` | ❌ | `0` | Which timestep in sequence to inspect (0-indexed) |
| `--dense_gaze` | ❌ | `False` | Use all gaze points (ignore sample rate) |

#### Example Commands

**Basic extraction:**
```bash
python inspect_features.py --checkpoint model.pth
```

**High-resolution extraction:**
```bash
python inspect_features.py \
    --checkpoint best_model.pth \
    --size 1024 \
    --batch_size 2 \
    --save_dir features_1024
```

**Extract from middle of sequence:**
```bash
python inspect_features.py \
    --checkpoint model.pth \
    --batch_idx 1 \
    --timestep 4 \
    --save_dir features_middle
```

**Dense gaze analysis:**
```bash
python inspect_features.py \
    --checkpoint model.pth \
    --dense_gaze \
    --save_dir features_dense
```

#### What Gets Extracted

The script extracts feature maps from these backbone layers:
- `conv1` - Initial convolution layer
- `bn1` - First batch normalization
- `relu` - First ReLU activation
- `maxpool` - Initial max pooling
- `layer1` - First residual block group
- `layer2` - Second residual block group  
- `layer3` - Third residual block group
- `layer4` - Fourth residual block group

### Feature Analysis (`analyze_features.py`)

#### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `feature_file` | ✅ | - | Path to saved feature file (.pkl) |
| `--save_dir` | ❌ | `feature_analysis` | Directory to save analysis plots |
| `--layers` | ❌ | `all` | Specific layers to analyze |

#### Example Commands

**Analyze all layers:**
```bash
python analyze_features.py features_b0_t0_Robofarmer-II_512.pkl
```

**Analyze specific layers:**
```bash
python analyze_features.py features_b0_t0_Robofarmer-II_512.pkl \
    --layers layer1 layer2 layer3 layer4 \
    --save_dir analysis_resnet_layers
```

**Analyze early layers:**
```bash
python analyze_features.py features_b0_t0_Robofarmer-II_512.pkl \
    --layers conv1 bn1 relu maxpool \
    --save_dir analysis_early_layers
```

## Understanding the Output

### Generated Files

#### Feature Extraction Output
```
feature_maps/
├── features_b0_t0_Robofarmer-II_512.pkl  # Extracted features (binary)
├── conv1_b0_t0.png                       # Feature map visualizations
├── layer1_b0_t0.png
├── layer2_b0_t0.png
├── layer3_b0_t0.png
└── layer4_b0_t0.png
```

#### Analysis Output
```
feature_analysis/
├── conv1_distribution.png          # Activation value distributions
├── conv1_channel_stats.png         # Per-channel statistics
├── conv1_spatial_patterns.png      # Spatial activation patterns
├── layer1_distribution.png
├── layer1_channel_stats.png
├── layer1_spatial_patterns.png
├── ...
└── layer_comparison.png            # Cross-layer comparison
```

### Interpretation Guide

#### 1. Feature Map Visualizations
- **Bright regions**: High activation values
- **Dark regions**: Low/zero activation values
- **Channel diversity**: Different channels detect different features
- **Spatial patterns**: Show what spatial structures activate each channel

#### 2. Distribution Plots
- **Histogram**: Shows distribution of activation values
- **Box plot**: Shows quartiles and outliers
- **Cumulative distribution**: Shows percentage of activations below each value

#### 3. Channel Statistics
- **Channel means**: Average activation per channel
- **Channel stds**: Variability of activations per channel
- **Channel maximums**: Peak activation values per channel
- **Channel sparsity**: Fraction of zero activations (ReLU effect)

#### 4. Key Metrics to Monitor

| Metric | Good Range | Interpretation |
|--------|------------|----------------|
| **Mean Activation** | 0.1 - 2.0 | Too low: dead neurons, Too high: saturation |
| **Sparsity** | 30% - 70% | Too low: inefficient, Too high: information loss |
| **Channel Variance** | Moderate spread | All similar: redundancy, Too varied: instability |

## Advanced Usage

### 1. Integrating with Custom Models

To use with your own model, modify the hook registration in `feature_inspector.py`:

```python
# In FeatureMapInspector._register_hooks()
def _register_hooks(self):
    # For custom model structure
    backbone = self.model.your_backbone_name
    
    # Define your layer names
    layer_names = [
        'your_conv1', 'your_bn1', 'your_relu',
        'your_block1', 'your_block2', # etc.
    ]
    
    for name in layer_names:
        if hasattr(backbone, name):
            layer = getattr(backbone, name)
            hook = layer.register_forward_hook(get_activation(name))
            self.hooks.append(hook)
```

### 2. Multi-Sample Analysis

Extract features from multiple samples:

```bash
# Extract from different samples
for i in {0..3}; do
    python inspect_features.py \
        --checkpoint model.pth \
        --batch_idx $i \
        --save_dir features_sample_$i
done
```

### 3. Temporal Analysis

Extract features across timesteps:

```bash
# Extract from different timesteps
for t in {0..7}; do
    python inspect_features.py \
        --checkpoint model.pth \
        --timestep $t \
        --save_dir features_time_$t
done
```

### 4. Programmatic Usage

Use the inspector directly in Python:

```python
from feature_inspector import FeatureMapInspector

# Load your model and data
model = load_your_model()
batch = get_your_batch()

# Create inspector
inspector = FeatureMapInspector(model, save_dir="my_features")

# Set target
inspector.set_target(batch_idx=0, timestep=2)

# Extract features
activations = inspector.extract_features(batch)

# Save and visualize
inspector.save_features(batch_info={"experiment": "test1"})
inspector.visualize_feature_maps("layer2", num_channels=16)

# Cleanup
inspector.cleanup()
```

## Code Integration

### Adding to Existing Training Scripts

Add feature inspection to your training loop:

```python
# In your training script
from feature_inspector import FeatureMapInspector

def train_epoch(model, dataloader, optimizer):
    # Your training code...
    
    # Add inspection every N epochs
    if epoch % 10 == 0:
        inspect_features_during_training(model, dataloader, epoch)

def inspect_features_during_training(model, dataloader, epoch):
    inspector = FeatureMapInspector(model, f"features_epoch_{epoch}")
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Extract features
    model.eval()
    with torch.no_grad():
        inspector.set_target(batch_idx=0, timestep=0)
        activations = inspector.extract_features(batch)
        inspector.save_features(batch_info={"epoch": epoch})
    
    model.train()
    inspector.cleanup()
```

### Custom Analysis Functions

Create custom analysis functions:

```python
def compare_models(model1_features, model2_features, save_dir):
    """Compare features between two different models"""
    
    # Load both feature files
    data1 = load_feature_file(model1_features)
    data2 = load_feature_file(model2_features)
    
    # Compare layer by layer
    for layer in data1['activations']:
        if layer in data2['activations']:
            act1 = data1['activations'][layer][0].numpy()
            act2 = data2['activations'][layer][0].numpy()
            
            # Calculate differences
            diff = np.abs(act1 - act2)
            
            # Plot comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(np.mean(act1, axis=0), cmap='viridis')
            axes[0].set_title(f'Model 1 - {layer}')
            
            axes[1].imshow(np.mean(act2, axis=0), cmap='viridis')
            axes[1].set_title(f'Model 2 - {layer}')
            
            axes[2].imshow(np.mean(diff, axis=0), cmap='hot')
            axes[2].set_title(f'Difference - {layer}')
            
            plt.savefig(f'{save_dir}/comparison_{layer}.png')
            plt.show()
```

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or image resolution:
```bash
python inspect_features.py --checkpoint model.pth --batch_size 1 --size 256
```

#### 2. **Hook Registration Errors**
```
AttributeError: 'Module' object has no attribute 'layer1'
```
**Solution:** Check your model architecture and update layer names in `_register_hooks()`.

#### 3. **Empty Activations**
```
No activations found for layer X
```
**Solution:** Ensure the layer name exists and the forward pass reaches that layer.

#### 4. **File Not Found**
```
FileNotFoundError: [Errno 2] No such file or directory: 'model.pth'
```
**Solution:** Use absolute paths:
```bash
python inspect_features.py --checkpoint /full/path/to/model.pth
```

### Memory Management

For large models or high-resolution inputs:

```python
# In feature_inspector.py, modify to save immediately
def get_activation(name):
    def hook(model, input, output):
        if should_save_activation():
            # Save immediately to disk instead of storing in memory
            torch.save(output.detach().cpu(), f'temp_{name}.pt')
    return hook
```

### Debugging Tips

1. **Check model loading:**
```python
# Verify model loads correctly
checkpoint = torch.load('model.pth', map_location='cpu')
print(checkpoint.keys())
```

2. **Verify layer names:**
```python
# Print all layer names
for name, module in model.named_modules():
    print(name)
```

3. **Test with small data:**
```bash
# Use minimal settings for testing
python inspect_features.py \
    --checkpoint model.pth \
    --batch_size 1 \
    --max_len 2 \
    --size 224
```

## Best Practices

1. **Start Small:** Begin with low resolution and small batches
2. **Clean Up:** Always call `inspector.cleanup()` to free memory
3. **Batch Processing:** For multiple extractions, process one at a time
4. **Save Metadata:** Include relevant information in `batch_info`
5. **Version Control:** Keep track of which model checkpoint was used
6. **Compare Systematically:** Use consistent settings when comparing different models

## Example Workflow

Here's a complete workflow for analyzing a trained model:

```bash
# 1. Extract features from trained model
python inspect_features.py \
    --checkpoint /path/to/best_model.pth \
    --dset Robofarmer-II \
    --size 512 \
    --batch_idx 0 \
    --timestep 0 \
    --save_dir experiment_1

# 2. Extract from middle of sequence for comparison
python inspect_features.py \
    --checkpoint /path/to/best_model.pth \
    --dset Robofarmer-II \
    --size 512 \
    --batch_idx 0 \
    --timestep 4 \
    --save_dir experiment_1

# 3. Analyze both extractions
python analyze_features.py experiment_1/features_b0_t0_Robofarmer-II_512.pkl \
    --save_dir analysis_t0

python analyze_features.py experiment_1/features_b0_t4_Robofarmer-II_512.pkl \
    --save_dir analysis_t4

# 4. Compare results manually or create custom comparison script
```

This workflow will give you comprehensive insights into what your CNN is learning and how features evolve throughout the temporal sequence.