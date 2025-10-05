import torch
import argparse
import tqdm
import os
import glob
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import torch.utils.data as tdata

from utils import evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--dset", default="opra")
parser.add_argument("--load", default=None)
parser.add_argument("--res", type=int, default=28)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()
# ------------------------------------------------------------#

import data
from data import opra, epic
import torch.nn.functional as F


def generate_gt(dset):

    os.makedirs("data/%s/output/" % dset, exist_ok=True)

    dataset = tdata.Dataset()

    if dset == "opra":
        dataset = opra.OPRAHeatmaps(root=data._DATA_ROOTS[dset], split="val")
    elif dset == "epic":
        dataset = epic.EPICHeatmaps(root=data._DATA_ROOTS[dset], split="val")

    dataset.heatmaps = dataset.init_hm_loader()

    heatmaps, keys = [], []
    for index in tqdm.tqdm(range(len(dataset))):
        entry = dataset.data[index]
        image_key = "_".join(entry["image"]).encode("utf-8")
        verb_key = str(entry["verb"]).encode("utf-8")

        # NOTE: The image key is not suppose to be the index, it is suppsoed to be the name of the file?

        # image_key = image_key.encode("utf-8")
        # verb_key = verb_key.encode("utf-8")

        hm_key = (image_key, verb_key)

        if hm_key not in dataset.heatmaps.map:
            print(f"KeyError: {hm_key} not found in heatmaps.map")
            continue  # Skip this key and continue with the next iteration

        heatmap = dataset.heatmaps(hm_key)
        heatmap = torch.from_numpy(heatmap)
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0][0]

        heatmap = heatmap / (heatmap.sum() + 1e-12)
        heatmaps.append(heatmap)
        keys.append(hm_key)
    if not heatmaps:
        print("No heatmaps were generated. Please check your keys and dataset.")
        return

    heatmaps = torch.stack(heatmaps, 0)
    torch.save(
        {"heatmaps": heatmaps, "keys": keys},
        os.path.expanduser(
            "~/Desktop/MasterThesis/data/datasets/Robofarmer/output/gt.pth"
        ),
    )


# ------------------------------------------------------------#

from interaction_hotspots.models import intcam


def generate_heatmaps(dset, load, batch_size):

    testset = torch.utils.data.Dataset()

    if dset == "opra":
        testset = opra.OPRAHeatmaps(root=data._DATA_ROOTS[dset], split="val")
    elif dset == "epic":
        testset = epic.EPICHeatmaps(root=data._DATA_ROOTS[dset], split="val")

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    from interaction_hotspots.models import rnn, backbones

    torch.backends.cudnn.enabled = False
    net = rnn.frame_lstm(len(testset.verbs), max_len=-1, backbone=backbones.dr50_n28)

    checkpoint = torch.load(load, map_location="cpu")

    custom_state_dict = net.state_dict()

    for key, value in checkpoint.items():
        if key in custom_state_dict:
            if key == "fc.weight":
                custom_state_dict[key][:, :] = checkpoint[key][: len(testset.verbs), :]
            elif key == "fc.bias":
                custom_state_dict[key][:, :] = checkpoint[key][: len(testset.verbs)]
            elif custom_state_dict[key].shape == value.shape:
                custom_state_dict[key] = value
            else:
                continue
        else:
            print(f"Layer not found in the model: {key}. Skipping.")

    net.load_state_dict(custom_state_dict)
    net.eval().cuda()
    print("Loaded checkpoint from %s" % os.path.basename(load))

    gcam = intcam.IntCAM(net)

    heatmaps = []
    for batch in tqdm.tqdm(testloader, total=len(testloader)):
        img, verb = batch["img"], batch["verb"]

        # img_names = batch["image_name"]

        # assert img.shape[0] == 4, f"Shape of the batch is not matching: {img.shape}"
        # for i in range(img.shape[0]):
        #     assert img[i].shape == torch.Size([3, 224, 224])

        masks = gcam.generate_cams(img.cuda(), [verb])  # (B, T, C, 7, 7)
        mask = masks.mean(1)  # (B, C, 7, 7) <-- average across hallucinated time dim
        mask = mask.squeeze(1)  # get rid of single class dim
        heatmaps.append(mask.cpu())

    heatmaps = torch.cat(heatmaps, 0)  # (N, C, 7, 7)

    keys = [testset.key(entry) for entry in testset.data]
    torch.save({"heatmaps": heatmaps, "keys": keys}, "%s.%s.heatmaps" % (load, dset))


# ------------------------------------------------------------#

if __name__ == "__main__":

    # generate gt heatmaps if they do not already exist
    # print("data/%s/output/gt.pth" % (args.dset))
    # if not os.path.exists("data/%s/output/gt.pth" % (args.dset)):
    generate_gt("epic")

    # generate heatmap predictions if they do not already exist
    # if args.load and not os.path.exists("%s.%s.heatmaps" % (args.load, args.dset)):
    generate_heatmaps(args.dset, args.load, args.batch_size)
    # print("loading checkpoint:", args.load)
    gt = torch.load(
        os.path.expanduser(
            "~/Desktop/MasterThesis/data/datasets/Robofarmer/output/gt.pth"
        )
    )

    baselines = evaluation.Baselines(gt["heatmaps"].shape[0])
    heval = evaluation.Evaluator(gt, res=args.res, log=args.load)

    # Comment in other methods to compare
    predictions = {
        # 'center': baselines.gaussian(),
        # 'egogaze': baselines.checkpoint('data/%s/output/egogaze.pth'%(args.dset)),
        # 'mlnet': baselines.checkpoint('data/%s/output/mlnet.pth'%(args.dset)),
        # 'deepgaze2': baselines.checkpoint('data/%s/output/deepgaze2.pth'%(args.dset)),
        # 'salgan': baselines.checkpoint('data/%s/output/salgan.pth'%(args.dset)),
        "hotspots": baselines.checkpoint("%s.%s.heatmaps" % (args.load, args.dset)),
        # 'img2heatmap': baselines.checkpoint('data/%s/output/img2heatmap.pth'%(args.dset)),
    }
    for method in predictions:
        # print(method)
        heatmaps = predictions[method]
        # print(f"Prediction shape: {heatmaps['heatmaps'].shape}")
        # print(f"Prediction keys count: {len(heatmaps['keys'])}")
        # print(f"GT keys count: {len(gt['keys'])}")
        # print(
        #     f"Prediction min/max: {heatmaps['heatmaps'].min()}/{heatmaps['heatmaps'].max()}"
        # )
        scores = heval.evaluate(heatmaps)
