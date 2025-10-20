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
import json
from copy import deepcopy
import subprocess
from datetime import datetime
import tqdm

import warnings
from utils import util
import data
from data import epic
from interaction_hotspots.models import rnn, gaze_rnn, cons_rnn, backbones, intcam
from util_funcs.image_processing import gabor_edge_aug

# Just filtering away warnings -> Unless things brake, do not remove
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights*")

cudnn.benchmark = True
parser = argparse.ArgumentParser()

parser.add_argument("--dset", type=str, default="Robofarmer-II")
# Load is equivalent to checkpoint path
# TODO: Print
parser.add_argument(
    "--checkpoint",
    type=str,
    help="Name of the directory holding checkpoints for a spcific model",
)
parser.add_argument(
    "--model_type",
    default=str,
    help="Which type of checkpoint to use: best_loss | best_acc | final",
)
parser.add_argument(
    "--model",
    type=str,
    default="LSTM",
    help="Tells the program which model to use: LSTM | baseGazeLSTM | GazeLSTM",
)

parser.add_argument("--split", type=str, default="train")

parser.add_argument(
    "--res",
    type=int,
    default=224,
    help="Controls the size of displayed visualizations. Output has dimensions = (C, args.res, args.res)",
)

parser.add_argument("--gaussianSize", type=int, default=33)
parser.add_argument(
    "--edge_det",
    action="store_true",
    help="Tells the program to generate edge detection images",
)

args = parser.parse_args()

models = {
    "LSTM": rnn.frame_lstm,
    "BaseGazeLSTM": gaze_rnn.frame_lstm_gaze,
    "GazeLSTM": cons_rnn.cons_frame_lstm,
}


def resize_tensor(tensor, sz):
    tensor = F.interpolate(tensor, (sz, sz), mode="bilinear", align_corners=True)
    return tensor


def blur(tensor, sz, Z):  # (3, 224, 224)
    tensor = tensor.permute(1, 2, 0).numpy()
    # k_size = int(np.sqrt(sz**2) / Z)
    k_size = int(sz / Z)
    if k_size % 2 == 0:
        k_size += 1
    k_size = max(3, min(k_size, 51))
    tensor = cv2.GaussianBlur(tensor, (k_size, k_size), 0)
    tensor = torch.from_numpy(tensor).permute(2, 0, 1)
    return tensor


def post_process(hmaps, sz):
    hmaps = torch.stack([hmap / (hmap.max() + 1e-12) for hmap in hmaps], 0)
    hmaps = hmaps.numpy()

    processed = []
    for c in range(hmaps.shape[0]):
        hmap = hmaps[c]
        hmap[hmap < 0.5] = 0
        hmap = cv2.GaussianBlur(hmap, (3, 3), 0)
        processed.append(hmap)
    processed = np.array(processed)
    processed = torch.from_numpy(processed).float()
    processed = resize_tensor(processed.unsqueeze(0), sz)[0]

    return processed


def generate_color_map(hmaps, colors, sz):
    colors = [color_map[c] for c in colors]
    colors = 1 - torch.FloatTensor(colors).unsqueeze(2).unsqueeze(2)  # invert colors

    vals, idx = torch.sort(hmaps, 0, descending=True)
    cmap = torch.zeros(hmaps.shape)
    for c in range(hmaps.shape[0]):
        cmap[c][idx[0] == c] = vals[0][idx[0] == c]

    cmap = cmap.unsqueeze(1).expand(
        cmap.shape[0], 3, cmap.shape[-1], cmap.shape[-1]
    )  # (C, 3, 224, 224)

    cmap = [hmap * color for hmap, color in zip(cmap, colors)]
    cmap = torch.stack(cmap, 0)  # (C, 3, 14, 14)

    cmap = resize_tensor(cmap, sz)
    cmap, _ = cmap.max(0)

    # blur the heatmap to make it smooth
    blur_z = max(6, 18 - sz / 224)
    cmap = blur(cmap, sz, blur_z)
    cmap = 1 - cmap  # invert heatmap: white background

    # improve contrast for visibility
    cmap = transforms.ToPILImage()(cmap)
    cmap = ImageEnhance.Color(cmap).enhance(1.5)
    cmap = ImageEnhance.Contrast(cmap).enhance(1.5)
    cmap = transforms.ToTensor()(cmap)

    return cmap


def generate_gt(dset, dataset: epic.EPICHeatmaps, **kwargs):

    # os.makedirs(f"../../data/datasets/{args,dset}/output/", exist_ok=True)

    dataset.heatmaps = dataset.init_hm_loader()
    gazemaps = []
    heatmaps, keys = [], []
    for index in tqdm.tqdm(range(len(dataset))):
        entry = dataset.data[index]
        # For convenience, the gazemaps were stacked to three channels, all idenctical
        gazemap = dataset[index]["gazemap"][0, :, :]
        heatmap = dataset[index]["heatmap"]
        image_key = "_".join(entry["image"])
        verb_key = str(entry["verb"])

        # NOTE: The image key is not suppose to be the index, it is suppsoed to be the name of the file?
        # image_key = image_key.encode("utf-8")
        # verb_key = verb_key.encode("utf-8")

        hm_key = (image_key, verb_key)
        # Create the heatmap
        heatmap = heatmap / (heatmap.sum() + 1e-12)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(kwargs["size"], kwargs["size"]),
            mode="bilinear",
            align_corners=False,
        )[0][0]

        heatmaps.append(heatmap)
        gazemaps.append(gazemap)
        keys.append(hm_key)
    if not heatmaps:
        print("No heatmaps were generated. Please check your keys and dataset.")
        return

    heatmaps = torch.stack(heatmaps, 0)
    gazemaps = torch.stack(gazemaps, 0)
    gazemaps = {"heatmaps": gazemaps, "keys": deepcopy(keys)}
    heatmaps = {"heatmaps": heatmaps, "keys": deepcopy(keys)}

    return heatmaps, gazemaps


def generate_single_color_map(hmaps, verb_gt_index, colors, sz):
    """Generate heatmap with single color based on verb_gt index"""
    # Use the color corresponding to the verb_gt index
    color = color_map[colors[verb_gt_index]]
    color = 1 - torch.FloatTensor(color).unsqueeze(1).unsqueeze(1)  # invert color

    # Combine all heatmap channels into a single intensity map
    combined_hmap = hmaps.sum(0)  # Sum across all channels

    # Normalize the combined heatmap
    if combined_hmap.max() > 0:
        combined_hmap = combined_hmap / combined_hmap.max()

    # Apply the single color to the combined heatmap
    cmap = combined_hmap.unsqueeze(0).expand(
        3, combined_hmap.shape[0], combined_hmap.shape[1]
    )  # (3, H, W)
    cmap = cmap * color  # Apply color

    cmap = resize_tensor(cmap.unsqueeze(0), sz)[0]  # Resize

    # blur the heatmap to make it smooth
    cmap = blur(cmap, sz, 9)
    cmap = 1 - cmap  # invert heatmap: white background

    # improve contrast for visibility
    cmap = transforms.ToPILImage()(cmap)
    cmap = ImageEnhance.Color(cmap).enhance(1.5)
    cmap = ImageEnhance.Contrast(cmap).enhance(1.5)
    cmap = transforms.ToTensor()(cmap)

    return cmap


def overlay_colored_heatmaps(
    uimg, hmaps, viz_idx, colors, sz, single_color=False
):  # (C, 224, 224)

    # post process heatmaps: normalize each channel, blur, threshold
    if not single_color:
        hmaps = post_process(hmaps, sz)  # (C, 224, 224)
    hmaps = hmaps[viz_idx]

    # generate color map from each heatmap channel
    if single_color and len(viz_idx) == 1:
        # Use single color based on the first (and only) index
        cmap = generate_single_color_map(hmaps, viz_idx[0], colors, sz)
    else:
        cmap = generate_color_map(hmaps, colors, sz)

    # generate per-pixel alpha channel and overlay
    alpha = (1 - cmap).mean(0)
    overlay = (1 - alpha) * uimg + alpha * cmap

    return overlay


def edge_detection(img, low_thr=40, high_thr=90):

    # Convert to grayscale (use RGB2GRAY if input is from PyTorch, which uses RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    import matplotlib.pyplot as plt

    # Using Canny edge detection with gabor filters
    gabor_img = gabor_edge_aug(gray_img)
    # Normalize gabor output to ensure it's in proper uint8 range
    gabor_img = cv2.normalize(gabor_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Canny with lower thresholds for better edge detection
    edge_img = cv2.Canny(gabor_img, low_thr, high_thr)
    # Stack to create 3-channel image (channels-last format for numpy)
    edge_result = np.stack([edge_img, edge_img, edge_img], axis=0)

    return edge_result


def visualize(dataset, index, gt_heatmaps, gazemaps, path, viz_verbs, colors, sz=224):

    # load image
    img = Image.open(path).convert("RGB")
    img = util.default_transform(args.split, size=args.res)(img)

    # generate heatmaps
    hmaps = gcam.generate_cams(
        img.cuda().unsqueeze(0), list(range(len(dataset.verbs)))
    )  # (1, T, C, 28, 28)
    hmaps = hmaps.mean(1).squeeze(0).cpu()  # (C, 28, 28)

    gt_heatmap = torch.stack(
        [
            gt_heatmaps["heatmaps"][index],
            gt_heatmaps["heatmaps"][index],
            gt_heatmaps["heatmaps"][index],
        ]
    )
    gazemap = torch.stack(
        [
            gazemaps["heatmaps"][index],
            gazemaps["heatmaps"][index],
            gazemaps["heatmaps"][index],
        ]
    )
    # Load ground truths and gazemaps

    # overlay heatmaps on original image
    uimg = util.unnormalize(img)
    uimg2 = deepcopy(uimg)
    uimg3 = deepcopy(uimg)

    uimg = F.interpolate(
        uimg.unsqueeze(0), (sz, sz), mode="bicubic", align_corners=False
    )[0]
    uimg2 = F.interpolate(
        uimg2.unsqueeze(0), (sz, sz), mode="bicubic", align_corners=False
    )[0]
    uimg3 = F.interpolate(
        uimg3.unsqueeze(0), (sz, sz), mode="bicubic", align_corners=False
    )[0]

    viz_idx = [dataset.verbs.index(v) for v in viz_verbs]
    overlay = overlay_colored_heatmaps(uimg, hmaps, viz_idx, colors, sz)

    # For gt_heatmaps and gazemaps, use single color when verb_gt has length 1
    verb_gt_idx = [dataset[index]["verb"]]
    overlay2 = overlay_colored_heatmaps(
        uimg2, gt_heatmap, verb_gt_idx, colors, sz, single_color=True
    )
    overlay3 = overlay_colored_heatmaps(
        uimg3, gazemap, verb_gt_idx, colors, sz, single_color=True
    )
    # If you use
    if args.edge_det:
        uimg = (
            torch.from_numpy(edge_detection(uimg.cpu().numpy().transpose(1, 2, 0)))
            / 255
        )
    # display heatmaps next to original
    viz_imgs = [uimg, overlay, overlay2, overlay3]
    grid = torchvision.utils.make_grid(viz_imgs, nrow=2, padding=2)
    grid = transforms.ToPILImage()(grid)
    return grid


# -----------------------------------------------------------------------------------------------------#
#
if __name__ == "__main__":

    color_map = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "cyan": [0, 1, 1],
        "magenta": [1, 0, 1],
        "yellow": [1, 1, 0],
        "white": [1, 1, 1],
    }

    colors = [
        "magenta",  # cut stem
        "yellow",  # cut flower
        "red",  # cut leafs
    ]

    input_path = os.path.join(
        f"../../data/datasets/{args.dset}/inactive_images",
        f"{args.split}_images",
    )

    dataset_path = f"../../data/datasets/{args.dset}"
    checkpoints_path = os.path.join(dataset_path, "checkpoints", args.checkpoint)

    checkpoint_path = os.path.join(
        checkpoints_path,
        f"{args.model_type}.pt",
    )

    dataset = epic.EPICHeatmaps(
        root=data._DATA_ROOTS[args.dset],
        split=args.split,
        d_name=args.dset,
        size=args.res,
        sample_rate=None,
        dense_gaze=True,
        gaussianSize=args.gaussianSize,
        normalize=False,
    )
    annot_file = open(
        os.path.join(data._DATA_ROOTS["Robofarmer-II"], "annotation.json"), "r"
    )
    annotation = json.load(annot_file)
    annot_file.close()

    viz_verbs = annotation["verbs"]

    colors = [
        "magenta",  # cut stem
        "yellow",  # cut flower
        "red",  # cut leafs
    ]

    # Dry run of the model to find the feature map output size from final layer of CNN backbone
    backbone = backbones.dr50_n28()
    backbone.eval()
    dummy_input = torch.rand(1, 3, args.res, args.res)
    spatial_dim = dummy_output = backbone(dummy_input).shape[-1]

    torch.backends.cudnn.enabled = False
    # net = rnn.frame_lstm(len(dataset.verbs), max_len=-1, backbone=backbones.dr50_n28)
    net = models[args.model]
    net = net(
        len(dataset.verbs),
        max_len=-1,
        backbone=backbones.dr50_n28,
        spatial_dim=spatial_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    custom_state_dict = net.state_dict()

    # Custom loading of weights
    for key, value in checkpoint.items():
        if key in custom_state_dict:
            if key == "fc.weight":
                custom_state_dict[key][:, :] = checkpoint[key][: len(dataset.verbs), :]
            elif key == "fc.bias":
                custom_state_dict[key][:, :] = checkpoint[key][: len(dataset.verbs)]
            elif custom_state_dict[key].shape == value.shape:
                custom_state_dict[key] = value
            else:
                continue
        # else:
        #     print(f"Layer not found in the model: {key}. Skipping.")

    if "ckpt_E_20.pth" in args.checkpoint:
        # NOTE: Load EPIC-Kitchen parameters
        net.load_state_dict(custom_state_dict)
    else:
        # NOTE: Load trained parameters
        net.load_state_dict(checkpoint["state_dict"])

    print("Loaded checkpoint from %s" % os.path.basename(args.checkpoint))

    gcam = intcam.IntCAM(net)
    gcam.eval().cuda()

    # Create the directory structure
    vis_path = os.path.join(dataset_path, "visualizations")
    if not os.path.exists(vis_path):
        try:
            subprocess.run(["mkdir", vis_path])
        except:
            print(f"Could not create directory: {vis_path}")

    split_path = os.path.join(vis_path, args.split)
    if not os.path.exists(split_path):
        try:
            subprocess.run(["mkdir", split_path])
        except:
            print(f"Could not create directory: {split_path}")

    maps_path = os.path.join(
        split_path, f"{args.model}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    )
    if not os.path.exists(maps_path):
        try:
            subprocess.run(["mkdir", maps_path])
        except:
            print(f"Could not create directory: {maps_path}")

    # for fl in glob.glob("%s/*.jpg" % args.inp):
    gt_heatmaps, gazemaps = generate_gt(args.dset, dataset, size=args.res)
    for i in range(len(dataset)):
        img_path = f"../../data/datasets/Robofarmer-II/inactive_images/{args.split}_images/{dataset[i]['image_name']}"
        img = visualize(
            dataset, i, gt_heatmaps, gazemaps, img_path, viz_verbs, colors, sz=args.res
        )
        img.save("%s/%s" % (maps_path, dataset[i]["image_name"]))

    meta_data = {}

    checkpoint_path = checkpoint_path.removesuffix(f"/{args.model_type}.pt")
    if os.path.exists(os.path.join(checkpoint_path, "meta.json")):
        model_meta_file = open(os.path.join(checkpoint_path, "meta.json"), "r")
        model_meta = json.load(model_meta_file)
        model_meta_file.close()
        meta_data = {"model": model_meta}

    # Save metadata from run
    action_color_map = {}
    action_color_map["cut stem"] = {"color": "magenta", "color_code": [1, 0, 1]}
    action_color_map["cut flower"] = {"color": "yellow", "color_code": [1, 1, 0]}
    action_color_map["cut leaf"] = {"color": "red", "color_code": [1, 0, 0]}
    meta_data["action_color"] = action_color_map

    visualization_meta = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        "gaussianSize": args.gaussianSize,
        "action_color": action_color_map,
        "edge_det": True if args.edge_det is not None else False,
        "resolution": f"{args.res}x{args.res}",
        "split": args.split,
    }

    meta_file = open(os.path.join(maps_path, "meta.json"), "w")
    json.dump(meta_data, meta_file, indent=2)
    meta_file.close()

    # Due to key errors, remove the file "heatmaps.h5" in the dataset directory
    if os.path.exists("../../data/datasets/Robofarmer-II/heatmaps.h5"):
        try:
            subprocess.run(
                [
                    "rm",
                    "-rf",
                    "../../data/datasets/Robofarmer-II/heatmaps.h5",
                ]
            )
        except:
            print(
                f"Could not remove file: {os.path.exists("../../data/datasets/Robofarmer-II/heatmaps.h5")}"
            )
