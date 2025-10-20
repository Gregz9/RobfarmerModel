# Training and Evaluation Guide

This guide explains how to use `train.py` for training and evaluating interaction hotspot models.

## Overview

The `train.py` script supports training and evaluation of three model types:
- **LSTM**: Basic frame-based LSTM model
- **BaseGazeLSTM**: LSTM with gaze input
- **GazeLSTM**: Constrained LSTM with gaze attention

## Basic Usage

### Training a Model
```bash
python train.py --model LSTM --dset Robofarmer-II --max_epochs 20
```

### Training with Validation
```bash
python train.py --model GazeLSTM --dset Robofarmer-II --validate --max_epochs 20
```

### Evaluating a Model
```bash
python train.py --model LSTM --dset Robofarmer-II --test --checkpoint path/to/checkpoint.pt
```

## Required Arguments

- `--dset`: Dataset name (`Robofarmer`, `Robofarmer-II`, `epic`)

## Key Arguments

### Model Configuration
- `--model`: Model type (`LSTM` | `BaseGazeLSTM` | `GazeLSTM`) [default: LSTM]
- `--max_len`: Frame sequence length [default: 8]
- `--resolution`: Image/gazemap size [default: 224]

### Training Parameters
- `--batch_size`: Training batch size [default: 8]
- `--max_epochs`: Number of training epochs [default: 20]
- `--lr`: Learning rate [default: 1e-4]
- `--weight_decay`: Optimizer weight decay [default: 5e-4]
- `--decay_after`: Epoch for 10x learning rate decay [default: 15]

### Gaze Settings
- `--dense_gaze`: Use all gaze points (ignores sample rate)
- `--gaussianSize`: Gaussian kernel size for gaze heatmaps [default: 33]

### Execution Modes
- `--validate`: Run validation during training
- `--test`: Evaluate on test set (requires `--checkpoint`)
- `--finetune`: Fine-tune from checkpoint (requires `--checkpoint`)

### System Settings
- `--workers`: DataLoader workers [default: 8]
- `--parallel`: Use multiple GPUs with DataParallel
- `--cv_dir`: Checkpoint save directory [default: cv/tmp/]

## Examples

### Train GazeLSTM with validation
```bash
python train.py \
    --model GazeLSTM \
    --dset Robofarmer-II \
    --validate \
    --max_epochs 25 \
    --batch_size 16 \
    --resolution 256 \
    --dense_gaze
```

### Fine-tune from checkpoint
```bash
python train.py \
    --model LSTM \
    --dset Robofarmer-II \
    --finetune \
    --checkpoint path/to/best_loss.pt \
    --max_epochs 10 \
    --lr 1e-5
```

**Note**: When using `--finetune`, only the final two layers (layer3 and layer4) of the CNN backbone are loaded from the checkpoint. This selective loading helps with transfer learning by preserving high-level features while allowing adaptation to new datasets.

### Test model performance
```bash
python train.py \
    --model GazeLSTM \
    --dset Robofarmer-II \
    --test \
    --checkpoint path/to/best_accuracy.pt
```

## Output Files

The script generates several output files:

### Checkpoints
Saved to: `~/Desktop/MasterThesis/data/datasets/{dset}/checkpoints/{model}_{timestamp}/`
- `best_loss.pt`: Best validation/training loss
- `best_accuracy.pt`: Best accuracy
- `final.pt`: Final epoch checkpoint
- `checkpoint.pt`: Periodic checkpoint (every 3 epochs)
- `meta.json`: Training metadata

### Metrics
Saved to: `~/Desktop/MasterThesis/data/datasets/{dset}/training_metrics/`
- `{timestamp}_{epochs}_epochs_training_{model}.json`: Training metrics
- `{timestamp}_{epochs}_epochs_validation_{model}.json`: Validation metrics
- `{timestamp}_{epochs}_epochs_test_{model}.json`: Test metrics

### TensorBoard Logs
Saved to: `~/Desktop/MasterThesis/data/dataset/{dset}/runs/{model}_{timestamp}/`

## Model Performance Tracking

The script tracks multiple loss components:
- **Total Loss**: Combined loss across all components
- **Class Loss**: Classification loss for verb prediction
- **Attention Loss**: Gaze attention mechanism loss
- **Anticipation Loss**: How well does the model anticipate frames from inactive ones
- **Batch Accuracy**: Classification accuracy per batch

## Tips

1. **Memory**: Reduce `--batch_size` if you encounter GPU memory issues
2. **Resolution**: Higher `--resolution` improves quality but requires more memory
3. **Dense Gaze**: Use `--dense_gaze` for better gaze heatmap quality
4. **Validation**: Always use `--validate` to monitor overfitting
5. **Checkpoints**: Best models are automatically saved based on loss and accuracy
