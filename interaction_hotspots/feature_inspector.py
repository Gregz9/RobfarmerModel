import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


class FeatureMapInspector:
    """
    Tool for extracting and saving feature maps from backbone CNN layers
    """
    
    def __init__(self, model, save_dir="feature_maps"):
        self.model = model
        self.save_dir = save_dir
        self.activations = defaultdict(list)
        self.hooks = []
        self.target_batch_idx = 0  # Which batch sample to inspect
        self.target_timestep = 0   # Which timestep to inspect
        
        os.makedirs(save_dir, exist_ok=True)
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on backbone layers"""
        def get_activation(name):
            def hook(model, input, output):
                # Only save for target batch and timestep
                if hasattr(self, '_current_batch_idx') and hasattr(self, '_current_timestep'):
                    if (self._current_batch_idx == self.target_batch_idx and 
                        self._current_timestep == self.target_timestep):
                        # Clone to avoid memory issues
                        self.activations[name].append(output.detach().cpu().clone())
            return hook
        
        # Register hooks on key backbone layers
        backbone = self.model.backbone
        layer_names = [
            'conv1', 'bn1', 'relu', 'maxpool',
            'layer1', 'layer2', 'layer3', 'layer4'
        ]
        
        for name in layer_names:
            if hasattr(backbone, name):
                layer = getattr(backbone, name)
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
                print(f"Registered hook on {name}")
    
    def set_target(self, batch_idx=0, timestep=0):
        """Set which batch sample and timestep to extract features from"""
        self.target_batch_idx = batch_idx
        self.target_timestep = timestep
        print(f"Target set to batch[{batch_idx}], timestep[{timestep}]")
    
    def extract_features(self, batch):
        """
        Extract features during model forward pass
        """
        self.activations.clear()
        
        # Get batch dimensions
        frames = batch["frames"]
        B, T = frames.shape[:2]
        
        print(f"Processing batch with shape: B={B}, T={T}")
        
        # Flatten frames for backbone processing
        frames_flat = frames.view(B * T, *frames.shape[2:])
        
        # Process each frame individually to track timesteps
        for t in range(T):
            for b in range(B):
                self._current_batch_idx = b
                self._current_timestep = t
                
                # Get the specific frame
                frame_idx = b * T + t
                frame = frames_flat[frame_idx:frame_idx+1]
                
                # Forward pass through backbone
                with torch.no_grad():
                    _ = self.model.backbone(frame)
                
                # Only extract for target sample
                if b == self.target_batch_idx and t == self.target_timestep:
                    break
            
            if hasattr(self, '_current_batch_idx') and self._current_batch_idx == self.target_batch_idx:
                break
        
        return self.activations
    
    def save_features(self, batch_info=None, suffix=""):
        """Save extracted features to disk"""
        if not self.activations:
            print("No activations to save! Run extract_features() first.")
            return
        
        # Create filename
        filename = f"features_b{self.target_batch_idx}_t{self.target_timestep}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pkl"
        
        save_path = os.path.join(self.save_dir, filename)
        
        # Convert to regular dict and save
        features_dict = {
            'activations': dict(self.activations),
            'batch_idx': self.target_batch_idx,
            'timestep': self.target_timestep,
            'batch_info': batch_info
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"Features saved to {save_path}")
        
        # Print summary
        for layer_name, activations in self.activations.items():
            if activations:
                shape = activations[0].shape
                print(f"  {layer_name}: {shape}")
    
    def load_features(self, filename):
        """Load previously saved features"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'rb') as f:
            features_dict = pickle.load(f)
        return features_dict
    
    def visualize_feature_maps(self, layer_name, num_channels=16, save_plot=True):
        """Visualize feature maps from a specific layer"""
        if layer_name not in self.activations or not self.activations[layer_name]:
            print(f"No activations found for layer {layer_name}")
            return
        
        # Get the activation tensor
        activation = self.activations[layer_name][0]  # [1, C, H, W]
        activation = activation.squeeze(0)  # [C, H, W]
        
        # Select channels to visualize
        num_channels = min(num_channels, activation.shape[0])
        channels_to_show = np.linspace(0, activation.shape[0]-1, num_channels, dtype=int)
        
        # Create subplot grid
        cols = 4
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
        
        for i, channel_idx in enumerate(channels_to_show):
            if i >= len(axes):
                break
                
            feature_map = activation[channel_idx].numpy()
            
            im = axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Channel {channel_idx}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(len(channels_to_show), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Feature Maps - {layer_name} (B{self.target_batch_idx}, T{self.target_timestep})')
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, f'{layer_name}_b{self.target_batch_idx}_t{self.target_timestep}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def get_feature_statistics(self):
        """Get statistics about extracted features"""
        stats = {}
        for layer_name, activations in self.activations.items():
            if activations:
                activation = activations[0]
                stats[layer_name] = {
                    'shape': activation.shape,
                    'mean': activation.mean().item(),
                    'std': activation.std().item(),
                    'min': activation.min().item(),
                    'max': activation.max().item(),
                    'num_zero': (activation == 0).sum().item(),
                    'sparsity': (activation == 0).float().mean().item()
                }
        return stats
    
    def cleanup(self):
        """Remove hooks to free memory"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        print("Hooks removed and activations cleared")


# Usage example
def inspect_model_features(model, dataloader, save_dir="feature_maps"):
    """
    Main function to inspect model features
    
    Args:
        model: Trained model
        dataloader: DataLoader with batch data
        save_dir: Directory to save feature maps
    """
    
    # Create inspector
    inspector = FeatureMapInspector(model, save_dir)
    
    # Set model to eval mode
    model.eval()
    
    try:
        # Get first batch
        batch = next(iter(dataloader))
        
        # Move to GPU if available
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        # Set target (first sample, first timestep)
        inspector.set_target(batch_idx=0, timestep=0)
        
        # Extract features
        print("Extracting features...")
        activations = inspector.extract_features(batch)
        
        # Save features
        batch_info = {
            'verb': batch['verb'][0].item() if 'verb' in batch else None,
            'video_id': 'unknown',  # Add if available in batch
            'frame_id': 'unknown'   # Add if available in batch
        }
        inspector.save_features(batch_info=batch_info, suffix="sample")
        
        # Print statistics
        print("\nFeature Statistics:")
        stats = inspector.get_feature_statistics()
        for layer_name, layer_stats in stats.items():
            print(f"{layer_name}:")
            print(f"  Shape: {layer_stats['shape']}")
            print(f"  Mean: {layer_stats['mean']:.4f}, Std: {layer_stats['std']:.4f}")
            print(f"  Sparsity: {layer_stats['sparsity']:.2%}")
        
        # Visualize some layers
        key_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        for layer in key_layers:
            if layer in activations:
                print(f"\nVisualizing {layer}...")
                inspector.visualize_feature_maps(layer, num_channels=16)
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        
    finally:
        # Cleanup
        inspector.cleanup()


if __name__ == "__main__":
    # Example usage
    print("Feature Map Inspector created")
    print("Use inspect_model_features(model, dataloader) to extract and visualize features")