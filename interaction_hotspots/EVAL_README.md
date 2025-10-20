# Evaluation Guide

This guide explains how to use `eval.py` for evaluating interaction hotspot models using various metrics and comparison methods.

## Overview

The `eval.py` script evaluates model performance by comparing:
- **Model predictions** vs **Ground truth heatmaps** (default)
- **Gaze maps** vs **Ground truth heatmaps** (`--ground_vs_gaze`)
- **Gaze maps** vs **Model predictions** (`--gaze_vs_model`)

## Basic Usage

### Evaluate Model vs Ground Truth
```bash
python eval.py --checkpoint model_checkpoint --model_type best_loss
```

### Evaluate Gaze vs Ground Truth
```bash
python eval.py --ground_vs_gaze
```

### Evaluate Gaze vs Model
```bash
python eval.py --checkpoint model_checkpoint --gaze_vs_model
```

## Required Arguments

For model evaluation, you need:
- `--checkpoint`: Checkpoint directory name (not full path)

## Key Arguments

### Dataset Configuration
- `--dset`: Dataset name [default: Robofarmer-II]
- `--split`: Data split to evaluate (`val` | `test`) [default: val]

### Model Selection
- `--checkpoint`: Checkpoint directory name (e.g., `LSTM_12012024`)
- `--model_type`: Checkpoint type (`best_loss` | `best_acc` | `final`) [default: best_loss]
- `--model`: Model architecture (`LSTM` | `BaseGazeLSTM` | `GazeLSTM`) [default: LSTM]

### Resolution Settings
- `--heatmap_res`: Input image/heatmap resolution [default: 224]
- `--eval_res`: Evaluation resolution for metrics, i.e. KLD, SIM and AUC-J. This is the size that is used when computing this metrics [default: 28]
- `--gaussianSize`: Gaussian kernel size for heatmaps. Controlls the spread of points when generating heatmaps. The default values is perfect for when images are approx 224x224. [default: 33]

### Evaluation Modes
- `--ground_vs_gaze`: Compare ground truth vs gaze maps
- `--gaze_vs_model`: Compare gaze maps vs model predictions
- (default): Compare ground truth vs model predictions

### System Settings
- `--batch_size`: Evaluation batch size [default: 64]
- `--num_workers`: DataLoader workers [default: 2]
- `--num_evals`: Number of evaluation runs [default: 1]

## Examples

### Evaluate best loss model on validation set
```bash
python eval.py \
    --checkpoint LSTM_12012024 \
    --model_type best_loss \
    --model LSTM \
    --split val
```

### Evaluate best accuracy model on test set
```bash
python eval.py \
    --checkpoint GazeLSTM_15012024 \
    --model_type best_acc \
    --model GazeLSTM \
    --split test
```

### Compare gaze maps vs ground truth
```bash
python eval.py \
    --ground_vs_gaze \
    --heatmap_res 256 \
    --gaussianSize 49
```

### Compare gaze vs model with higher resolution
```bash
python eval.py \
    --checkpoint LSTM_12012024 \
    --gaze_vs_model \
    --heatmap_res 512 \
    --eval_res 28 \
    --gaussianSize 99
```

## Evaluation Metrics

The script computes three key metrics:

### KLD (Kullback-Leibler Divergence)
- **Lower is better** (0 = perfect match)
- Measures distributional difference between predictions and ground truth
- Sensitive to probability mass allocation

### SIM (Similarity)
- **Higher is better** (1 = perfect match, 0 = no similarity)
- Normalized correlation between heatmaps
- Measures structural similarity

### AUC-J (Area Under Curve - Judd)
- **Higher is better** (1 = perfect, 0.5 = random)
- Area under ROC curve for fixation prediction
- Measures discriminative power

## Output Files

### Evaluation Results
Saved to: `../../data/datasets/Robofarmer-II/evaluation_metrics/{split}_{timestamp}/`
- `meta.json`: Complete evaluation metadata including:
  - Model configuration and training parameters
  - Evaluation settings and results
  - Metric scores with mean and standard error

### Console Output
Pure example output: 
```
KLD: 0.245 ± 0.012 (150/150)
SIM: 0.678 ± 0.018 (150/150)
AUC-J: 0.734 ± 0.015 (150/150)
--------------------
```

## Resolution Guidelines

### Gaussian Kernel Size
- **224x224 heatmaps**: `--gaussianSize 33` (default)
- **~400x400 heatmaps**: `--gaussianSize 49`
- **Larger heatmaps**: `--gaussianSize 99`
- **28x28 feature maps**: `--gaussianSize 5`

### Evaluation Resolution
- **Low-res evaluation**: `--eval_res 28` (faster, standard)
- **High-res evaluation**: `--eval_res 224` (more accurate, slower) 

## Checkpoint Path Structure

Checkpoints should be located at:
```
../../data/datasets/Robofarmer-II/checkpoints/{checkpoint}/{model_type}.pt
```

For example:
- `--checkpoint LSTM_12012024` looks for:
  - `../../data/datasets/Robofarmer-II/checkpoints/LSTM_12012024/best_loss.pt`
  - `../../data/datasets/Robofarmer-II/checkpoints/LSTM_12012024/meta.json`

## Tips

1. **Memory**: Reduce `--batch_size` for GPU memory issues
2. **Speed**: Use lower `--eval_res` for faster evaluation
3. **Accuracy**: Use higher `--heatmap_res` and `--eval_res` for more precise metrics
4. **Baselines**: Use `--ground_vs_gaze` to establish baseline performance
5. **Comparison**: Use `--gaze_vs_model` to see how models compare to human gaze
6. **Multiple runs**: Increase `--num_evals` for statistical robustness
