# Visualization Guide

This guide explains how to use `viz.py` to create visual comparisons of your trained models against ground truth data and human gaze patterns.

## What Does This Script Do?

Think of `viz.py` as your visual debugging tool. It takes a trained model and creates side-by-side comparison images showing:

1. **Original image** - The raw input image
2. **Model prediction** - Where your AI thinks people will look (colored heatmap overlay)
3. **Ground truth** - Where people actually interacted (annotation data)
4. **Gaze data** - Where people actually looked during the task

This helps you **visually understand** how well your model is performing and where it might be going wrong.

## Basic Usage

### Visualize a Trained Model
```bash
python viz.py --checkpoint your_model_folder --model_type best_loss
```

This creates a folder of comparison images showing how your model performs on each test image.

## Why Would You Use This?

### Debug Model Performance
- **See patterns**: Does your model consistently miss certain types of interactions?
- **Spot issues**: Are the heatmaps too blurry, too sharp, or in wrong locations?
- **Compare versions**: Visual comparison between different model checkpoints

### Understand Your Data
- **Gaze vs annotations**: See how human eye movements relate to actual interaction points
- **Action types**: Different actions (cut stem, cut flower, cut leaf) are shown in different colors
- **Data quality**: Spot potential issues in your annotation or gaze data

## Key Arguments Explained

### Essential Settings
- `--checkpoint`: Your model folder name (e.g., `LSTM_12012024`)
  - **What it does**: Tells the script which trained model to visualize
  - **Why important**: You need this to load your model weights

- `--model_type`: Which checkpoint to use (`best_loss` | `best_acc` | `final`)
  - **best_loss**: Model with lowest training loss (usually most stable)
  - **best_acc**: Model with highest accuracy (might be overfitted)
  - **final**: Last epoch model (might be undertrained)

### Visual Quality Control
- `--res`: Output image size (default: 224)
  - **Lower (224)**: Faster processing, standard quality
  - **Higher (512, 1024)**: Better visual detail, slower processing
  - **Trade-off**: Quality vs speed - use higher for final presentations

- `--gaussianSize`: How "spread out" the heatmap points look (default: 33)
  - **Smaller values (15-25)**: Sharp, precise hotspots
  - **Larger values (50-99)**: Smooth, blurry hotspots
  - **Rule of thumb**: Increase this when you increase `--res`

### Dataset Control
- `--split`: Which data to visualize (`train` | `val` | `test`)
  - **train**: See how model performs on training data (should be good)
  - **val**: See validation performance (real test of generalization)
  - **test**: Final evaluation (use sparingly)

- `--edge_det`: Add edge detection overlay
  - **What it does**: Shows detected edges in the image
  - **When useful**: Debug if model is focusing on edges vs objects

## Examples with Explanations

### Quick Model Check
```bash
python viz.py --checkpoint LSTM_15012024 --model_type best_loss
```
**Purpose**: Fast check of your best model's visual performance on validation set.

### High-Quality Presentation Images
```bash
python viz.py \
    --checkpoint GazeLSTM_20012024 \
    --model_type best_acc \
    --res 512 \
    --gaussianSize 75 \
    --split val
```
**Purpose**: Create publication-quality visualization images with smooth, high-resolution heatmaps.

### Debug Training Issues
```bash
python viz.py \
    --checkpoint LSTM_10012024 \
    --model_type final \
    --split train \
    --res 256
```
**Purpose**: See if model is learning properly by checking performance on training data.

### Edge Detection Analysis
```bash
python viz.py \
    --checkpoint baseline_model \
    --edge_det \
    --res 384
```
**Purpose**: Check if model predictions correlate with image edges (might indicate overfitting to low-level features).

## Understanding the Output

### What You'll See
The script creates a 2x2 grid for each image with this exact layout:

```
┌─────────────────────┬─────────────────────┐
│ Original Image      │ Model Predictions   │
│ (uimg)              │ (overlay)           │
│ - Raw input image   │ - AI model heatmap  │
│ - Edge detected if  │ - Multi-colored     │
│   --edge_det used   │ - Shows all actions │
└─────────────────────┼─────────────────────┤
│ Ground Truth        │ Human Gaze Data     │
│ (overlay2)          │ (overlay3)          │
│ - Annotation data   │ - Eye tracking data │
│ - Single color      │ - Single color      │
│ - Where interaction │ - Where people      │
│   actually happened │   actually looked   │
└─────────────────────┴─────────────────────┘
```

**Grid positions (left-to-right, top-to-bottom):**
1. **Top-left**: Original image (or edge-detected version if `--edge_det` used)
2. **Top-right**: Model prediction heatmap overlaid on image
3. **Bottom-left**: Ground truth interaction heatmap (annotation data)
4. **Bottom-right**: Human gaze heatmap (eye tracking data)

### Color Coding
Each action type gets a different color:
- **Magenta**: Cut stem actions
- **Yellow**: Cut flower actions  
- **Red**: Cut leaf actions

### Output Location
Images are saved to:
```
../../data/datasets/{dataset}/visualizations/{split}/{model}_{timestamp}/
```

## Resolution Guidelines Explained

### Why Resolution Matters
- **Input images**: Your model was trained on specific image sizes
- **Heatmap quality**: Higher resolution shows finer details but takes more memory
- **Gaussian spread**: Larger images need larger Gaussian kernels for smooth appearance

### Recommended Settings
```bash
# Standard quality (fast)
--res 224 --gaussianSize 33

# High quality (balanced)
--res 384 --gaussianSize 55

# Publication quality (slow)
--res 512 --gaussianSize 75

# Very high resolution (memory intensive)
--res 1024 --gaussianSize 150
```

## Troubleshooting Common Issues

### Heatmaps Look Too Thin/Translucent
**Problem**: At higher resolutions, heatmaps become barely visible
**Solution**: Increase `--gaussianSize` proportionally with `--res`
```bash
# Instead of this (bad)
--res 512 --gaussianSize 33

# Do this (good)
--res 512 --gaussianSize 75
```

### Out of Memory Errors
**Problem**: GPU runs out of memory during processing
**Solutions**:
1. Reduce `--res` (e.g., from 512 to 256)
2. Process fewer images at once
3. Use CPU-only processing (slower but uses system RAM)

### Wrong Model Loading
**Problem**: "Model not found" or dimension mismatch errors
**Solutions**:
1. Check checkpoint path exists: `../../data/datasets/Robofarmer-II/checkpoints/{your_checkpoint}/`
2. Verify model type matches training: use same `--model` argument as training
3. Ensure checkpoint contains the exact file: `{model_type}.pt`

## Pro Tips

### Visual Quality
1. **Smooth heatmaps**: Use `gaussianSize = res / 7` as starting point
2. **Sharp details**: For scientific analysis, use higher resolutions
3. **Presentation**: Use consistent resolution across all comparisons

### Analysis Strategy
1. **Start small**: Use `--res 224` for quick overview
2. **Zoom in**: Use higher resolution for detailed analysis of specific failures
3. **Compare models**: Generate visualizations for multiple checkpoints to see improvement

### Workflow Integration
1. **After training**: Always run visualization to check if training worked
2. **Before publication**: Generate high-resolution versions for papers/presentations
3. **Debugging**: Use train/val split comparisons to diagnose overfitting

The goal is to **see** what your model learned, not just read numbers. Good visualizations can reveal insights that metrics alone cannot capture.