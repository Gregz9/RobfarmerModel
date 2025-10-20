from re import A, S
import torch
import argparse
import tqdm
import os
import warnings
import json
from datetime import datetime
import numpy as np
from copy import deepcopy
import subprocess
import cv2 as cv

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights*")

import torch.utils.data as tdata
from utils import evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--dset", default="Robofarmer-II")
parser.add_argument("--split", type=str, default="val")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--gaussianSize", type=int, default=33)
# For images beteween 224 and approx. 400, use gaussianSize=49, larger use 99. For images with shape 28x28 use gaussianSize=5
parser.add_argument("--heatmap_res", type=int, default=224)
parser.add_argument("--eval_res", type=int, default=28)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument(
    "--test",
    action="store_true",
    help="Tells the evaluation program whether to use test set or not",
)
parser.add_argument(
    "--ground_vs_gaze",
    action="store_true",
    help="Tells the program to run evaluation of gazemaps vs ground truth heatmaps",
)
parser.add_argument(
    "--gaze_vs_model",
    action="store_true",
    help="Tells the program to run evaluation of gazemaps vs model heatmaps",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="best_loss",
    help="Type of model checkpoint to load: best_loss | best_acc | final",
)
parser.add_argument(
    "--model",
    type=str,
    default="LSTM",
    help="Tells the program which model to use: LSTM | baseGazeLSTM | GazeLSTM",
)
parser.add_argument(
    "--num_evals",
    type=int,
    default=1,
    help="Tells the program how many times it should evaluate the heatmaps",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="Specifiy the number of parallel workers for the dataloader",
)

args = parser.parse_args()
# ------------------------------------------------------------#

eval_metrics_path = "../../data/datasets/Robofarmer-II/evaluation_metrics.json"
import data
from data import opra, epic
import torch.nn.functional as F


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


# ------------------------------------------------------------#

from interaction_hotspots.models import intcam, backbones, rnn, gaze_rnn, cons_rnn

models = {
    "LSTM": rnn.frame_lstm,
    "BaseGazeLSTM": gaze_rnn.frame_lstm_gaze,
    "GazeLSTM": cons_rnn.cons_frame_lstm,
}


def generate_heatmaps(
    dset_name, dataset, dataloader, load, batch_size, split="val", **kwargs
):

    # Dry run of the model to find the feature map output size from final layer of CNN backbone
    backbone = backbones.dr50_n28()
    backbone.eval()
    dummy_input = torch.rand(1, 3, args.heatmap_res, args.heatmap_res)
    spatial_dim = dummy_output = backbone(dummy_input).shape[-1]

    torch.backends.cudnn.enabled = False

    net = models[args.model]
    net = net(
        len(dataset.verbs),
        max_len=-1,
        backbone=backbones.dr50_n28,
        spatial_dim=spatial_dim,
    )

    checkpoint = torch.load(load, map_location="cpu")

    custom_state_dict = net.state_dict()

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
        # print(f"Layer not found in the model: {key}. Skipping.")

    net.load_state_dict(custom_state_dict)
    net.eval().cuda()
    print("Loaded checkpoint from %s" % os.path.basename(load))

    gcam = intcam.IntCAM(net)

    heatmaps = []
    for batch in dataloader:
        img, verb = batch["img"], batch["verb"]

        masks = gcam.generate_cams(img.cuda(), [verb])  # (B, T, C, 7, 7)
        mask = masks.mean(1)  # (B, C, 7, 7) <-- average across hallucinated time dim
        mask = mask.squeeze(1)  # get rid of single class dim
        heatmaps.append(mask.cpu())

    heatmaps = torch.cat(heatmaps, 0)  # (N, C, 7, 7)

    keys = [dataset.key(entry) for entry in dataset.data]

    return {"heatmaps": heatmaps, "keys": keys}


# ------------------------------------------------------------#

if __name__ == "__main__":

    # Load dataset
    dataset = epic.EPICHeatmaps(
        root=data._DATA_ROOTS[args.dset],
        split=args.split,
        d_name=args.dset,
        size=args.heatmap_res,
        sample_rate=None,
        dense_gaze=True,
        gaussianSize=args.gaussianSize,
        normalize=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    gt, gazemaps = generate_gt(args.dset, dataset, size=args.heatmap_res)

    # generate heatmap predictions if they do not already exist
    # idx = 0
    # while True:
    #     frames = np.concatenate(
    #         [gt["heatmaps"][idx] * 100, gazemaps["heatmaps"][idx] * 100], axis=0
    #     )
    #     cv.imshow("Gaze map vs Annotation", frames)
    #
    #     key = cv.waitKey(40) & 0xFF
    #     if key == ord("q"):
    #         break
    #     elif key == 83:
    #         idx = idx + 1
    #     elif key == 81:
    #         idx = idx - 1

    checkpoint_path = os.path.join(
        f"../../data/datasets/Robofarmer-II/checkpoints/{args.checkpoint}/{args.model_type}.pt"
    )
    heatmaps = generate_heatmaps(
        args.dset,
        dataset,
        dataloader,
        checkpoint_path,
        args.batch_size,
        size=args.heatmap_res,
    )
    # gt = torch.load(
    #     os.path.expanduser(f"../../data/datasets/{args.dset}/output/gt.pth")
    # )

    if args.gaze_vs_model:
        baselines = evaluation.Baselines(gazemaps["heatmaps"].shape[0])
        # Ground truths are resized inside the evaluator. If args.res is not set, use args.size
        heval = evaluation.Evaluator(gazemaps, res=args.eval_res)

        # heatmaps = baselines.checkpoint("%s.%s.heatmaps" % (args.load, args.dset))
        scores, _ = heval.evaluate(heatmaps)
    elif args.ground_vs_gaze:
        baselines = evaluation.Baselines(gt["heatmaps"].shape[0])
        # Ground truths are resized inside the evaluator. If args.res is not set, use args.size
        heval = evaluation.Evaluator(gt, res=args.eval_res)

        # heatmaps = baselines.checkpoint("%s.%s.heatmaps" % (args.load, args.dset))
        scores, _ = heval.evaluate(gazemaps)
    # Standard ground truths vs models heatmaps
    else:
        baselines = evaluation.Baselines(gt["heatmaps"].shape[0])
        # Ground truths are resized inside the evaluator. If args.res is not set, use args.size
        heval = evaluation.Evaluator(gt, res=args.eval_res)
        # heatmaps = baselines.checkpoint("%s.%s.heatmaps" % (args.load, args.dset))
        scores, _ = heval.evaluate(heatmaps)

    write_out = []
    score_metrics = {}
    for key in ["KLD", "SIM", "AUC-J"]:
        key_score = [s[key] for s in scores if s[key] is not None]
        mean, stderr = np.mean(key_score), np.std(key_score) / (np.sqrt(len(key_score)))
        log_str = "%s: %.3f ± %.3f (%d/%d)" % (
            key,
            mean,
            stderr,
            len(key_score),
            len(dataset),
        )
        score_metrics[key] = {"mean": float(mean), "stderr": float(stderr)}
        write_out.append(log_str)
    write_out.append("-" * 20)
    write_out = "\n".join(write_out)
    print(write_out)

    # Save meta data from run
    checkpoint_path = os.path.join(
        f"../../data/datasets/Robofarmer-II/checkpoints/{args.checkpoint}/{args.model_type}.pt"
    )
    evaluation_meta = {}
    checkpoint_path = checkpoint_path.removesuffix(f"/{args.model_type}.pt")
    if os.path.exists(os.path.join(checkpoint_path, "meta.json")):
        model_meta_file = open(os.path.join(checkpoint_path, "meta.json"), "r")
        model_meta = json.load(model_meta_file)
        model_meta_file.close()

        # Create the meta data for this evaluation run
        evaluation_meta = {
            "model": model_meta,
        }

    evaluation_meta["evaluation"] = {
        "timestamp": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
        "batch_size": args.batch_size,
        "valset": not args.test,
        "testset": args.test,
        "heatmaps_res": args.heatmap_res,
        "eval_res": args.eval_res,
        "model_type": args.model_type,
        "run_type": (
            "gaze_vs_model"
            if args.gaze_vs_model or not args.ground_vs_gaze
            else ("ground_vs_gaze" if args.ground_vs_gaze else "ground_vs_model")
        ),
        "KLD": score_metrics["KLD"],
        "SIM": score_metrics["SIM"],
        "AUC-J": score_metrics["AUC-J"],
    }

    eval_meta_dir = f"../../data/datasets/Robofarmer-II/evaluation_metrics/{args.split}_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}"
    try:
        subprocess.run(["mkdir", eval_meta_dir])
    except:
        print(f"Could not create directory: {eval_meta_dir}")

    # Save the meta file
    meta_file_path = os.path.join(eval_meta_dir, "meta.json")

    meta_file = open(meta_file_path, "w")
    json.dump(evaluation_meta, meta_file, indent=2)
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

#
