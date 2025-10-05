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

from interaction_hotspots.utils import util

# NOTE: Defining a custom type
LossMeters = collections.defaultdict[str, tnt.meter.MovingAverageValueMeter]

cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--dset", default="epic", help="opra|epic")
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
    help="This falg tells the program whether to run validation for each epoch of training",
)
parser.add_argument("--workers", type=int, default=8, help="Workers for dataloader")
parser.add_argument("--log_every", default=10, type=int, help="Logging frequency")
args = parser.parse_args()

# NOTE: Storing locations
checkpoint_path = f"/home/gregz9/Desktop/MasterThesis/data/checkpoints"
checkpoint_path = f"/home/gregz9/Desktop/MasterThesis/data/checkpoints/{datetime.now().strftime('%d%m%Y%H%M')}_{args.max_epochs}_epochs_hands_model"
metrics_path = f"/home/gregz9/Desktop/MasterThesis/data/training_metrics/"
loss_file = os.path.join(
    metrics_path,
    f"{datetime.now().strftime('%d%m%Y%H%M')}_{args.max_epochs}_epochs_hands_model.json",
)

os.makedirs(args.cv_dir, exist_ok=True)
logging.basicConfig(
    filename="%s/run.log" % args.cv_dir,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))


def save(epoch):
    logger.info("Saving state, epoch: %d" % epoch)
    state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
    checkpoint = {"net": state_dict, "args": args, "epoch": epoch}
    torch.save(checkpoint, "%s/ckpt_E_%d.pth" % (args.cv_dir, epoch))


def save_model(epoch, model, optimizer, checkpoint_path, suffix="loss"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(
            checkpoint_path,
            f"base_model_best_{suffix}.pt",
        ),
    )


def train(epoch, writer, loader) -> LossMeters:

    net.train()

    iteration = 0
    total_iters = len(loader)
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))
    avg_acc = 0

    for batch in loader:
        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

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
    writer.add_scalar("Avg. Aux. Loss", loss_meters["aux_loss"].value()[0])
    writer.add_scalar("Avg. Class Loss", loss_meters["cls_loss"].value()[0])
    writer.add_scalar("Avg. Ant. Loss", loss_meters["ant_loss"].value()[0])

    return loss_meters


def validate(epoch, writer, loader) -> LossMeters:

    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))
    net.eval()
    iteration = 0
    total_iters = len(loader)
    for batch in loader:
        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

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

    # Pretty print of

    writer.add_scalar(
        "Val Avg. Total Loss", loss_meters["total_loss"].value()[0], epoch
    )
    writer.add_scalar("Val Avg. Batch Acc.", loss_meters["bacc %"].value()[0], epoch)
    writer.add_scalar("Val Avg. Aux. Loss", loss_meters["aux_loss"].value()[0])
    writer.add_scalar("Val Avg. Class Loss", loss_meters["cls_loss"].value()[0])
    writer.add_scalar("Val Avg. Ant. Loss", loss_meters["ant_loss"].value()[0])

    return loss_meters


# ----------------------------------------------------------------------------------------------------------------------------------------#
import interaction_hotspots.data as data
from interaction_hotspots.data import opra, epic


def load_data(train=True, val=False):

    trainset = epic.EPICInteractions(
        root=data._DATA_ROOTS[args.dset], split="train", max_len=args.max_len
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,  # args.workers,
        sampler=trainset.data_sampler(),
    )

    if val:
        valset = epic.EPICInteractions(
            root=data._DATA_ROOTS[args.dset], split="val", max_len=args.max_len
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=valset.data_sampler(),
        )

        return trainset, trainloader, valset, valloader

    return trainset, trainloader, None, None


def save_metrics(metrics, loss_file_name) -> None:

    loss_file = os.path.join(
        metrics_path,
        loss_file_name,
    )
    with open(loss_file, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


trainset, trainloader, valset, valloader = load_data(train=True, val=args.validate)

# Always using Epic as base
ANT_LOSS = "triplet"


from interaction_hotspots.models import rnn, hand_rnn, backbones

torch.backends.cudnn.enabled = False
net = hand_rnn.frame_lstm_hands(
    len(trainset.verbs),
    trainset.max_len,
    backbone=backbones.dr50_n28,
    ant_loss=ANT_LOSS,
)
net.cuda()

if args.parallel:
    net = nn.DataParallel(net)

optim_params = list(filter(lambda p: p.requires_grad, net.parameters()))
logger.info("# params to optimize %s" % len(optim_params))
optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[args.decay_after], gamma=0.1
)

# NOTE: Paths for backup of params and metrics
try:
    subprocess.run(["mkdir", checkpoint_path])
    subprocess.run(["mkdir", metrics_path])
except:
    print("COULD NOT CREATE DIRECTORIES!")

log_path = os.path.join(
    os.path.expanduser("~/Desktop/MasterThesis/data/runs"),
    f"hands_model_{args.max_epochs}_{args.batch_size}_{args.lr}_{args.weight_decay}",
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
    train_metrics = train(epoch, writer, trainloader)

    training_metrics.append(
        {
            "epoch": epoch,
            "total_loss": float(train_metrics["total_loss"].value()[0]),
            "cls_loss": float(train_metrics["cls_loss"].value()[0]),
            "ant_loss": float(train_metrics["ant_loss"].value()[0]),
            "aux_loss": float(train_metrics["aux_loss"].value()[0]),
            "accuracy": float(train_metrics["bacc %"].value()[0]),
        }
    )
    save_metrics(training_metrics, loss_file_name + "epochs_training_base_model.json")

    if args.validate:
        val_metrics = validate(epoch, writer, valloader)

        validation_metrics.append(
            {
                "epoch": epoch,
                "total_loss": float(val_metrics["total_loss"].value()[0]),
                "cls_loss": float(val_metrics["cls_loss"].value()[0]),
                "ant_loss": float(val_metrics["ant_loss"].value()[0]),
                "aux_loss": float(val_metrics["aux_loss"].value()[0]),
                "accuracy": float(val_metrics["bacc %"].value()[0]),
            }
        )
        save_metrics(
            validation_metrics, loss_file_name + "epochs_validation_base_model.json"
        )

    scheduler.step()

    if epoch % args.max_epochs == 0:
        save_model(epoch, net, optimizer, checkpoint_path, f"{epoch}")
    if float(train_metrics["total_loss"].value()[0]) < best_loss:
        best_loss = float(train_metrics["total_loss"].value()[0])
        save_model(epoch, net, optimizer, checkpoint_path, "loss")
    if float(train_metrics["bacc %"].value()[0]) > best_acc:
        best_acc = float(train_metrics["bacc %"].value()[0])
        save_model(epoch, net, optimizer, checkpoint_path, "acc")


writer.flush()
writer.close()
