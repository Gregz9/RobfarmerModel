import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import glob
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance
import cv2
import gc  # Import garbage collector

from utils import util
import data
from data import opra, epic
from models import rnn, backbones, intcam

cudnn.benchmark = True 
parser = argparse.ArgumentParser()
parser.add_argument('--dset', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--inp', default=None)
parser.add_argument('--out', default=None)
parser.add_argument('--batch_size', type=int, default=1, help='Number of images to process at once')
parser.add_argument('--image_size', type=int, default=256, help='Size of output images')
args = parser.parse_args()


def resize_tensor(tensor, sz):
    with torch.no_grad():  # Prevent memory leaks from autograd
        tensor = F.interpolate(tensor, (sz, sz), mode='bilinear', align_corners=True)
    return tensor


def blur(tensor, sz, Z):  # (3, 224, 224)
    with torch.no_grad():
        tensor_np = tensor.permute(1, 2, 0).cpu().numpy()
        k_size = int(np.sqrt(sz**2) / Z)
        if k_size % 2 == 0:
            k_size += 1
        tensor_np = cv2.GaussianBlur(tensor_np, (k_size, k_size), 0)
        tensor = torch.from_numpy(tensor_np).permute(2, 0, 1)
    return tensor


def post_process(hmaps, sz):
    with torch.no_grad():
        # Normalize more efficiently
        max_vals = hmaps.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-12
        hmaps_norm = hmaps / max_vals
        
        hmaps_np = hmaps_norm.cpu().numpy()
        
        processed = []        
        for c in range(hmaps_np.shape[0]):
            hmap = hmaps_np[c].copy()  # Make a copy to avoid modifying the original
            hmap[hmap < 0.5] = 0
            hmap = cv2.GaussianBlur(hmap, (3, 3), 0)
            processed.append(hmap)
            
        processed = np.array(processed)
        processed = torch.from_numpy(processed).float()
        processed = resize_tensor(processed.unsqueeze(0), sz)[0]
        
    return processed


color_map = {
    'red': [1, 0, 0], 
    'green': [0, 1, 0], 
    'blue': [0, 0, 1],
    'cyan': [0, 1, 1], 
    'magenta': [1, 0, 1], 
    'yellow': [1, 1, 0]
}


def generate_color_map(hmaps, colors, sz):
    with torch.no_grad():
        colors = [color_map[c] for c in colors]
        colors = 1 - torch.FloatTensor(colors).unsqueeze(2).unsqueeze(2)  # invert colors
        
        vals, idx = torch.sort(hmaps, 0, descending=True)
        cmap = torch.zeros_like(hmaps)
        for c in range(hmaps.shape[0]):
            cmap[c][idx[0] == c] = vals[0][idx[0] == c]
        
        # More memory-efficient expansion
        cmap = cmap.unsqueeze(1).expand(-1, 3, -1, -1)  # (C, 3, H, W)
        
        # Process one by one to save memory
        colored_maps = []
        for hmap, color in zip(cmap, colors):
            colored_maps.append(hmap * color)
        
        cmap = torch.stack(colored_maps, 0)  # (C, 3, H, W)
        del colored_maps  # Free memory
        
        cmap = resize_tensor(cmap, sz)
        cmap, _ = cmap.max(0)
        
        # blur the heatmap to make it smooth
        cmap = blur(cmap, sz, 9)
        cmap = 1 - cmap  # invert heatmap: white background
        
        # Convert to PIL and back more carefully
        cmap_pil = transforms.ToPILImage()(cmap.cpu())
        cmap_pil = ImageEnhance.Color(cmap_pil).enhance(1.5)
        cmap_pil = ImageEnhance.Contrast(cmap_pil).enhance(1.5)
        cmap = transforms.ToTensor()(cmap_pil)
        
    return cmap


def overlay_colored_heatmaps(uimg, hmaps, viz_idx, colors, sz):
    with torch.no_grad():
        # Select indices first to reduce memory
        hmaps_subset = hmaps[viz_idx]
        
        # post process heatmaps: normalize each channel, blur, threshold
        hmaps_proc = post_process(hmaps_subset, sz)
        
        # generate color map from each heatmap channel
        cmap = generate_color_map(hmaps_proc, colors, sz)
        
        # generate per-pixel alpha channel and overlay
        alpha = (1 - cmap).mean(0)
        overlay = (1 - alpha) * uimg + alpha * cmap
        
    return overlay


def visualize(path, viz_verbs, colors, sz=256):
    torch.cuda.empty_cache()  # Clear GPU cache before processing
    
    with torch.no_grad():  # Prevent tracking history
        # load image
        img = Image.open(path).convert('RGB')
        img = util.default_transform('val')(img)
        
        # Move to GPU, generate heatmaps, then move back to CPU
        img_cuda = img.cuda().unsqueeze(0)
        verb_indices = list(range(len(dataset.verbs)))
        hmaps = gcam.generate_cams(img_cuda, verb_indices)  # (1, T, C, 28, 28)
        hmaps = hmaps.mean(1).squeeze(0).cpu()  # (C, 28, 28)
        
        # Clean up GPU memory
        del img_cuda
        torch.cuda.empty_cache()
        
        # Process image on CPU
        uimg = util.unnormalize(img)
        uimg = F.interpolate(uimg.unsqueeze(0), (sz, sz), mode='bilinear', align_corners=False)[0]
        
        viz_idx = [dataset.verbs.index(v) for v in viz_verbs]
        overlay = overlay_colored_heatmaps(uimg, hmaps, viz_idx, colors, sz)
        
        # display heatmaps next to original
        viz_imgs = [uimg, overlay]
        grid = torchvision.utils.make_grid(viz_imgs, nrow=1, padding=2)
        grid_pil = transforms.ToPILImage()(grid)
        
        # Clean up
        del hmaps, uimg, overlay, viz_imgs, grid
        
    return grid_pil


def process_batch(image_paths, viz_verbs, colors, sz=256):
    results = []
    for path in image_paths:
        try:
            img = visualize(path, viz_verbs, colors, sz)
            results.append((path, img))
            print(f"Processed {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")
        
        # Force garbage collection after each image
        gc.collect()
        torch.cuda.empty_cache()
        
    return results


#-----------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    if args.dset == 'opra':
        dataset = opra.OPRAHeatmaps(root=data._DATA_ROOTS['opra'], split='val')
        viz_verbs = ['hold', 'rotate', 'push']
        colors = ['red', 'green', 'blue']
    elif args.dset == 'epic':
        dataset = epic.EPICHeatmaps(root=data._DATA_ROOTS['epic'], split='val')
        viz_verbs = ['cut', 'mix', 'open', 'close']
        colors = ['red', 'green', 'cyan', 'cyan']
    
    # Disable cuDNN to avoid potential memory issues
    torch.backends.cudnn.enabled = False
    
    # Set up model
    net = rnn.frame_lstm(len(dataset.verbs), max_len=-1, backbone=backbones.dr50_n28)
    
    # Load checkpoint efficiently
    checkpoint = torch.load(args.load, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    print(f'Loaded checkpoint from {os.path.basename(args.load)}')
    del checkpoint  # Free memory
    
    # Set up IntCAM
    gcam = intcam.IntCAM(net)
    gcam.eval().cuda()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Get all image paths
    image_paths = glob.glob(f'{args.inp}/*.jpg')
    print(f"Found {len(image_paths)} images to process")
    
    # Process images in batches to manage memory
    batch_size = args.batch_size
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        results = process_batch(batch_paths, viz_verbs, colors, args.image_size)
        
        # Save results
        for path, img in results:
            out_path = f'{args.out}/{os.path.basename(path)}'
            img.save(out_path)
            
        # Force cleanup after each batch
        gc.collect()
        torch.cuda.empty_cache()
