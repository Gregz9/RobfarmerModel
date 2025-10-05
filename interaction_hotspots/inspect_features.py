#!/usr/bin/env python3

import torch
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_inspector import inspect_model_features
from load_data import load_data
from models.cons_rnn import cons_frame_lstm
from models import backbones


def load_trained_model(checkpoint_path, num_classes, max_len):
    """Load a trained model from checkpoint"""
    
    # Create model
    model = cons_frame_lstm(
        num_classes=num_classes, 
        max_len=max_len, 
        backbone=backbones.dr50_n28,
        hidden_size=2048,
        ant_loss="triplet"
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from {checkpoint_path}")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Inspect CNN feature maps')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--dset', default='Robofarmer-II', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--max_len', type=int, default=8, help='Max sequence length')
    parser.add_argument('--size', type=int, default=512, help='Input image size')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--save_dir', default='feature_maps', help='Directory to save features')
    parser.add_argument('--batch_idx', type=int, default=0, help='Which batch sample to inspect')
    parser.add_argument('--timestep', type=int, default=0, help='Which timestep to inspect')
    parser.add_argument('--dense_gaze', action='store_true', help='Use dense gaze points')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    trainloader, valloader, num_classes = load_data(
        dset=args.dset,
        batch_size=args.batch_size,
        max_len=args.max_len,
        workers=args.workers,
        size=args.size,
        dense_gaze=args.dense_gaze
    )
    
    print(f"Dataset loaded. Number of classes: {num_classes}")
    
    # Load trained model
    print("Loading model...")
    model = load_trained_model(args.checkpoint, num_classes, args.max_len)
    
    # Inspect features
    print(f"Inspecting features for batch[{args.batch_idx}], timestep[{args.timestep}]...")
    
    # Create custom inspector for more control
    from feature_inspector import FeatureMapInspector
    
    inspector = FeatureMapInspector(model, args.save_dir)
    inspector.set_target(batch_idx=args.batch_idx, timestep=args.timestep)
    
    model.eval()
    
    try:
        # Get a batch from validation set (more stable than train)
        dataloader = valloader if valloader else trainloader
        batch = next(iter(dataloader))
        
        # Move to GPU
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        print(f"Batch shape: {batch['frames'].shape}")
        print(f"Extracting features for sample {args.batch_idx}, timestep {args.timestep}")
        
        # Extract features
        activations = inspector.extract_features(batch)
        
        # Save features with metadata
        batch_info = {
            'verb': batch['verb'][args.batch_idx].item() if 'verb' in batch else None,
            'noun': batch['noun'][args.batch_idx].item() if 'noun' in batch else None,
            'length': batch['length'][args.batch_idx].item() if 'length' in batch else None,
            'dataset': args.dset,
            'checkpoint': args.checkpoint,
            'image_size': args.size
        }
        
        inspector.save_features(batch_info=batch_info, suffix=f"{args.dset}_{args.size}")
        
        # Print statistics
        print("\nFeature Map Statistics:")
        stats = inspector.get_feature_statistics()
        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name}:")
            print(f"  Shape: {layer_stats['shape']}")
            print(f"  Mean: {layer_stats['mean']:.4f} ± {layer_stats['std']:.4f}")
            print(f"  Range: [{layer_stats['min']:.4f}, {layer_stats['max']:.4f}]")
            print(f"  Sparsity: {layer_stats['sparsity']:.1%}")
        
        # Visualize key layers
        print("\nGenerating visualizations...")
        key_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        
        for layer in key_layers:
            if layer in activations and activations[layer]:
                print(f"Visualizing {layer}...")
                inspector.visualize_feature_maps(layer, num_channels=16, save_plot=True)
        
        print(f"\nFeature inspection complete! Check {args.save_dir} for saved files.")
        
    except Exception as e:
        print(f"Error during feature inspection: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        inspector.cleanup()


if __name__ == '__main__':
    main()