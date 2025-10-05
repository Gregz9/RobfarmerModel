import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import argparse
import tqdm
import torchnet as tnt
import collections
import logging
import subprocess
from datetime import datetime
import json

from interaction_hotspots.models import rnn, gaze_rnn, cons_rnn, backbones
from utils import util

# NOTE: Possible model choices
models = {
    "LSTM": rnn.frame_lstm,
    "BaseGazeLSTM": gaze_rnn.frame_lstm_gaze,
    "GazeLSTM": cons_rnn.cons_frame_lstm,
}

# NOTE: Defining custom type
LossMeters = collections.defaultdict[str, tnt.meter.MovingAverageValueMeter]
cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="LSTM", help="LSTM|BaseGazeLSTM|GazeLSTM")
parser.add_argument("--dset", help="Robofarmer|Robofarmer-II|epic")
parser.add_argument(
    "--max_len", default=8, type=int, help="Length of frame sequence input to LSTM"
)
parser.add_argument(
    "--cv_dir", default="cv/tmp/", help="Directory for saving checkpoint models"
)
parser.add_argument("--batch_size", default=8, type=int, help="Batch size for training")
parser.add_argument(
    "--max_epochs", default=20, type=int, help="Total number of training epochs"
)
parser.add_argument("--lr", default=1e-4, type=float, help="Initial learning rate")
parser.add_argument(
    "--weight_decay", default=5e-4, type=float, help="Weight decay for optimizer"
)
parser.add_argument(
    "--decay_after",
    default=15,
    type=float,
    help="Epoch for scheduler to decay lr by 10x",
)
parser.add_argument(
    "--parallel", action="store_true", default=False, help="Use nn.DataParallel"
)
parser.add_argument(
    "--validate",
    action="store_true",
    help="This flag tells the program whether to run validation for each epoch of training",
)
# NOTE: Should only be used once or twice. Test sets should never be re-used multiple times
parser.add_argument(
    "--test",
    action="store_true",
    help="This flag tells the program whether to run the model on the test set",
)
parser.add_argument("--resolution", type=int, default=224, help="The dimension for the images and gazemaps")
parser.add_argument("--dense_gaze", action="store_true", help="Tells the program to ignore sample rate, and use all gaze points in a clip to generate a gazemap")

parser.add_argument("--finetune", action="store_true", help="Tells the program whether we wish to finetune saved checkpoint or not")
parser.add_argument("--checkpoint", type=str, help="Path the checkpoint used in finetuning")

parser.add_argument("--workers", type=int, default=8, help="Workers for dataloader")
parser.add_argument("--log_every", default=10, type=int, help="Logging frequency")
args = parser.parse_args()

# NOTE: Storing locations
checkpoint_path = os.path.expanduser(
    f"/app/data/datasets/{args.dset}/checkpoints/{args.model}_{datetime.now().strftime('%d-%m-%Y_%H-%M')}"
)

# NOTE: If directories do not exist, create them
if not os.path.exists(checkpoint_path):
    try:
        subprocess.run(["mkdir", checkpoint_path])
    except Exception as e:
        print(f"Error while creating directory: {e}")

metrics_path = os.path.expanduser(
    f"/app/data/datasets/{args.dset}/training_metrics/"
)

if not os.path.exists(metrics_path):
    try:
        subprocess.run(["mkdir", metrics_path])
    except Exception as e:
        print(f"Error while creating directory: {e}")

loss_file = os.path.join(
    metrics_path,
    f"{args.model}_{datetime.now().strftime('%d%m%Y%H%M')}.json",
)

os.makedirs(args.cv_dir, exist_ok=True)
logging.basicConfig(
    filename="%s/run.log" % args.cv_dir,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

# def save(epoch):
#     logger.info("Saving state, epoch: %d" % epoch)
#     state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
#     checkpoint = {"net": state_dict, "args": args, "epoch": epoch}
#     torch.save(checkpoint, "%s/ckpt_E_%d.pth" % (args.cv_dir, epoch))


def save_model(epoch, model, optimizer, checkpoint_path, suffix="loss"):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(
            checkpoint_path,
            f"{suffix}.pt",
        ),
    )
    
    f = open(os.path.join(checkpoint_path, "meta.json"), "w")
    metadata = {
        "start_time": datetime.now().strftime("%d%m%Y%H%M"),
        "model_name": model.name,
        "dataset": args.dset,
        "epoch": epoch,
        "batch_size": args.batch_size,
        "lr.rate": args.lr,
        "weight_decay" : args.weight_decay,
        "decay_after" : args.decay_after,
        "dense_gaze" : True if args.dense_gaze is not None else False,
        "image_res" : f"{args.resolution}x{args.resolution}"
    }
    json.dump(metadata, f, indent=2)

# NOTE: Dataset classes is equivalent to the verb classes
def load_params(model, checkpoint_path, dataset_classes, fine_tuning=False):
    from collections import OrderedDict

    # NOTE: Do not load any params
    if not fine_tuning:
        return model

    saved_params = OrderedDict()
    checkpoint = torch.load(checkpoint_path)
    if "net" in checkpoint.keys():
        saved_params = checkpoint["net"]
    elif "state_dict" in checkpoint.keys():
        saved_params = checkpoint["state_dict"]

    custom_state_dict = model.state_dict()
    # NOTE: Check the output dimensions of the model and saved parameters
    if not fine_tuning:
        try:
            model.load_state_dict(saved_params)
        except Exception as e:
            print(f"Error while loading model: {e}")

    if fine_tuning:
        for key, value in saved_params.items():
            if key in custom_state_dict:
                if "fc.weight" in key or "fc.bias" in key:
                    continue
                elif custom_state_dict[key].shape == value.shape:
                    custom_state_dict[key] = value
                else:
                    print(f"Layer not found in the model: {key}. Skipping.")

        with torch.no_grad():
            nn.init.xavier_normal(model.fc.weight)
            nn.init.zeros_(model.fc.bias)

        model.load_state_dict(custom_state_dict)

    return model


def train(epoch, writer, loader, class_weights=None) -> LossMeters:

    net.train()

    iteration = 0
    total_iters = len(loader)
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))
    avg_acc = 0

    for batch in loader:
        # print(batch["verbs"])
        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch, class_weights=class_weights)
        loss_dict = {k: v.mean() for k, v in loss_dict.items()}
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred_idx = pred.max(1)
        correct = (pred_idx == batch["verb"]).float().sum()
        batch_acc = correct / pred.shape[0]
        avg_acc += batch_acc
        loss_meters["bacc %"].add(batch_acc.item())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())
        loss_meters["total_loss"].add(loss.item())

        if iteration % args.log_every == 0:
            log_str = "epoch: %d + %d/%d | " % (epoch, iteration, total_iters)
            log_str += " | ".join(
                ["%s: %.3f" % (k, v.value()[0]) for k, v in loss_meters.items()]
            )
            logger.info(log_str)

        if iteration % len(loader) == 0 and iteration != 0:
            log_str = "epoch: %d + %d/%d | " % (epoch, iteration, total_iters)
            log_str += " | ".join(
                ["%s: %.3f" % (k, v.value()[0]) for k, v in loss_meters.items()]
            )
            logger.info(log_str)

        iteration += 1

    writer.add_scalar("Avg. Total Loss", loss_meters["total_loss"].value()[0], epoch)
    writer.add_scalar("Avg. Batch Acc.", loss_meters["bacc %"].value()[0], epoch)
    # writer.add_scalar("Avg. Aux. Loss", loss_meters["aux_loss"].value()[0], epoch)
    writer.add_scalar("Avg. Attention Loss", loss_meters["attention_loss"].value()[0], epoch)
    writer.add_scalar("Avg. Class Loss", loss_meters["cls_loss"].value()[0], epoch)
    writer.add_scalar("Avg. Ant. Loss", loss_meters["ant_loss"].value()[0], epoch)

    return loss_meters


def validate(epoch, writer, loader, class_weights=None) -> LossMeters:

    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))
    net.eval()
    iteration = 0
    total_iters = len(loader)
    for batch in loader:
        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch, class_weights=class_weights)

        loss_dict = {k: v.mean() for k, v in loss_dict.items()}
        loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_meters["val_" + k].add(v.item())
        loss_meters["val_total_loss"].add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred_idx = pred.max(1)
        correct = (pred_idx == batch["verb"]).float().sum()
        batch_acc = correct / pred.shape[0]
        loss_meters["val_bacc %"].add(batch_acc)

        if iteration % args.log_every == 0:
            log_str = "epoch: %d + %d/%d | " % (epoch, iteration, total_iters)
            log_str += " | ".join(
                ["%s: %.3f" % (k, v.value()[0]) for k, v in loss_meters.items()]
            )
            logger.info(log_str)

        if iteration % len(loader) == 0 and iteration != 0:
            log_str = "epoch: %d + %d/%d | " % (epoch, iteration, total_iters)
            log_str += " | ".join(
                ["%s: %.3f" % (k, v.value()[0]) for k, v in loss_meters.items()]
            )
        iteration += 1

    writer.add_scalar(
        "Val Avg. Total Loss", loss_meters["val_total_loss"].value()[0], epoch
    )
    writer.add_scalar(
        "Val Avg. Batch Acc.", loss_meters["val_bacc %"].value()[0], epoch
    )
    # writer.add_scalar(
        # "Val Avg. Aux. Loss", loss_meters["val_aux_loss"].value()[0], epoch
    # )
    writer.add_scalar(
        "Val Avg. Class Loss", loss_meters["val_cls_loss"].value()[0], epoch
    )
    writer.add_scalar(
        "Val Avg. Ant. Loss", loss_meters["val_ant_loss"].value()[0], epoch
    )

    return loss_meters


# ----------------------------------------------------------------------------------------------------------------------------------------#
# TODO: Move the imports
import interaction_hotspots.data as data
from data import epic


# NOTE: Always load training set by default
def load_data(val=False, test=False):

    trainset = epic.EPICInteractions(
        root=data._DATA_ROOTS[args.dset],
        split="train",
        d_name=args.dset,
        max_len=args.max_len,
        size=args.resolution,
        dense_gaze=args.dense_gaze

    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=trainset.data_sampler(),
    )

    if val:
        valset = epic.EPICInteractions(
            root=data._DATA_ROOTS[args.dset],
            split="val",
            d_name=args.dset,
            max_len=args.max_len,
            size=args.resolution,
            dense_gaze=args.dense_gaze
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=valset.data_sampler(),
        )

        return trainset, trainloader, valset, valloader

    # TODO: Implemenet loading of the test set

    return trainset, trainloader, None, None


def save_metrics(metrics, loss_file_name) -> None:

    loss_file = os.path.join(
        metrics_path,
        loss_file_name,
    )
    with open(loss_file, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


trainset, trainloader, valset, valloader = load_data(val=args.validate)

# Class weights
train_labels = np.bincount(np.array([trainset.data[i]["verb"] for i in range(len(trainset))]))
train_labels = {idx : np.sum(train_labels)/(len(train_labels) * train_labels[idx]) for idx in range(len(train_labels))}
class_weights = torch.tensor(list(train_labels.values()), dtype=torch.float32).cuda()

# Using triplet loss insead of MSE
ANT_LOSS = "triplet"

# if class_weights is not None:
# Import the models

net = models[args.model]
torch.backends.cudnn.enabled = False
net = net(
    len(trainset.verbs),
    trainset.max_len,
    backbone=backbones.dr50_n28,
    ant_loss=ANT_LOSS,
)

net = load_params(net, args.checkpoint, len(trainset.verbs), args.finetune)


# NOTE: Transfer the model to GPU
net.cuda()

# NOTE: If multiple GPUs are available
if args.parallel:
    net = nn.DataParallel(net)


optim_params = list(filter(lambda p: p.requires_grad, net.parameters()))
logger.info("# params to optimize %s" % len(optim_params))
# if args.model == "GazeLSTM":
    # optim_sigma = net.attention_sigma
    # optimizer = optim.Adam([{"params" : optim_params}, {"params" : [optim_sigma]}], lr=args.lr, weight_decay=args.weight_decay)
# else: 
#
optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[args.decay_after], gamma=0.1
)

# Pahts for backup of params and metrics
try:
    subprocess.run(["mkdir", checkpoint_path])
    subprocess.run(["mkdir", metrics_path])
except:
    print("COULD NOT CREATE DIRECTORIES!")

log_path = os.path.join(
    f"/app/data/datasets/{args.dset}/runs",
    f"{args.model}_{datetime.now().strftime('%d%m%Y%H%M')}",
)


writer = SummaryWriter(log_dir=log_path)

# Containers for dumping of metrics
training_metrics = []
validation_metrics = []

# Comparison values
best_loss = np.inf
best_acc = 0
start_epoch = 1  # or load checkpoint
loss_file_name = f"{datetime.now().strftime('%d%m%Y%H%M')}_{args.max_epochs}_"

for epoch in range(start_epoch, args.max_epochs + 1):
    logger.info("LR = %.2E" % scheduler.get_lr()[0])
    train_metrics = train(epoch, writer, trainloader, class_weights)

    training_metrics.append(
        {
            "epoch": epoch,
            "total_loss": float(train_metrics["total_loss"].value()[0]),
            "cls_loss": float(train_metrics["cls_loss"].value()[0]),
            "ant_loss": float(train_metrics["ant_loss"].value()[0]),
            # "aux_loss": float(train_metrics["aux_loss"].value()[0]),
            "accuracy": float(train_metrics["bacc %"].value()[0]),
        }
    )

    save_metrics(training_metrics, loss_file_name + f"epochs_training_{net.name}.json")

    if args.validate:
        val_metrics = validate(epoch, writer, valloader, class_weights)

        validation_metrics.append(
            {
                "epoch": epoch,
                "total_loss": float(val_metrics["val_total_loss"].value()[0]),
                "cls_loss": float(val_metrics["val_cls_loss"].value()[0]),
                "ant_loss": float(val_metrics["val_ant_loss"].value()[0]),
                # "aux_loss": float(val_metrics["aux_loss"].value()[0]),
                "accuracy": float(val_metrics["val_bacc %"].value()[0]),
            }
        )
        save_metrics(
            validation_metrics,
            loss_file_name + f"epochs_validation_{net.name}.json",
        )

    scheduler.step()

    if epoch % args.max_epochs == 0:
        save_model(epoch, net, optimizer, checkpoint_path, f"final_epoch_{epoch}")
    if epoch % 3 == 0:
        save_model(epoch, net, optimizer, checkpoint_path, "checkpoint")
    if float(train_metrics["total_loss"].value()[0]) < best_loss:
        best_loss = float(train_metrics["total_loss"].value()[0])
        save_model(epoch, net, optimizer, checkpoint_path, "best_loss")
    if float(train_metrics["bacc %"].value()[0]) > best_acc:
        best_acc = float(train_metrics["bacc %"].value()[0])
        save_model(epoch, net, optimizer, checkpoint_path, "best_accuracy")


writer.flush()
writer.close()
try:
    subprocess.run(["rm", "-rf", os.path.expanduser("/Desktop/MasterThesis/data/datasets/Robofarmer-II/tmp_heatmaps")])
except:
    print("Something is fucking wrong man")
