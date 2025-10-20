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
parser.add_argument("--model", default="LSTM", help="LSTM | BaseGazeLSTM | GazeLSTM")
parser.add_argument("--dset", help="Robofarmer | Robofarmer-II | epic")
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
parser.add_argument(
    "--resolution",
    type=int,
    default=224,
    help="The dimension for the images and gazemaps",
)
parser.add_argument(
    "--dense_gaze",
    action="store_true",
    help="Tells the program to ignore sample rate, and use all gaze points in a clip to generate a gazemap",
)
parser.add_argument("--gaussianSize", type=int, default=33)

parser.add_argument(
    "--finetune",
    action="store_true",
    help="Tells the program whether we wish to finetune saved checkpoint or not",
)
parser.add_argument(
    "--checkpoint", type=str, help="Path the checkpoint used in finetuning"
)

parser.add_argument("--workers", type=int, default=8, help="Workers for dataloader")
parser.add_argument("--log_every", default=10, type=int, help="Logging frequency")
args = parser.parse_args()

# NOTE: Storing locations
checkpoint_path = f"/app/data/datasets/{args.dset}/checkpoints/{args.model}_{datetime.now().strftime('%d%m%Y%H%M')}"
# NOTE: If directories do not exist, create them
if not os.path.exists(checkpoint_path):
    try:
        subprocess.run(["mkdir", checkpoint_path])
    except Exception as e:
        print(f"Error while creating directory: {e}")

metrics_path = f"/app/data/datasets/{args.dset}/training_metrics/"
if not os.path.exists(metrics_path):
    try:
        subprocess.run(["mkdir", metrics_path])
    except Exception as e:
        print(f"Error while creating directory: {e}")

loss_file = os.path.join(
    metrics_path,
    f"{args.model}_{datetime.now().strftime('%d%m%Y%H%M')}.json",
)

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))


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
        "weight_decay": args.weight_decay,
        "decay_after": args.decay_after,
        "dense_gaze": True if args.dense_gaze is not None else False,
        "image_res": f"{args.resolution}x{args.resolution}",
        "sequence_length": args.max_len,
        "finetuned": args.finetune,
        "train_set": not args.test,
        "val_set": args.validate,
        "test_set": args.test,
    }

    json.dump(metadata, f, indent=2)


# NOTE: Dataset classes is equivalent to the verb classes
def load_params(model, checkpoint_path, dataset_classes, fine_tuning=False):
    from collections import OrderedDict

    # NOTE: Do not load any params
    if not fine_tuning and not args.test:
        return model

    saved_params = OrderedDict()
    checkpoint = torch.load(checkpoint_path)
    if "net" in checkpoint.keys():
        saved_params = checkpoint["net"]
    elif "state_dict" in checkpoint.keys():
        saved_params = checkpoint["state_dict"]

    if args.test:
        model.load_state_dict(saved_params)
        return model

    custom_state_dict = model.state_dict()
    # NOTE: Check the output dimensions of the model and saved parameters
    if not fine_tuning:
        try:
            model.load_state_dict(saved_params)
        except Exception as e:
            print(f"Error while loading model: {e}")

    if fine_tuning:
        # Only load layer3 and layer4 from the backbone during fine-tuning
        for key, value in saved_params.items():
            if key in custom_state_dict:
                # Skip final classification layers
                if "fc.weight" in key or "fc.bias" in key:
                    continue
                # Only load layer3 and layer4 from backbone
                elif (
                    "backbone.rnet.layer3" in key or "backbone.rnet.layer4" in key
                ) and custom_state_dict[key].shape == value.shape:
                    custom_state_dict[key] = value
                    print(f"Loaded backbone layer: {key}")
                # Load all non-backbone parameters (RNN layers, etc.)
                elif (
                    "backbone.rnet" not in key
                    and custom_state_dict[key].shape == value.shape
                ):
                    custom_state_dict[key] = value
                else:
                    print(f"Skipping layer (fine-tuning mode): {key}")

        # Initialize final classification layers
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
    writer.add_scalar(
        "Avg. Attention Loss", loss_meters["attention_loss"].value()[0], epoch
    )
    writer.add_scalar("Avg. Class Loss", loss_meters["cls_loss"].value()[0], epoch)
    writer.add_scalar("Avg. Ant. Loss", loss_meters["ant_loss"].value()[0], epoch)

    return loss_meters


def test_validate(epoch, writer, loader, split="val", class_weights=None) -> LossMeters:

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
            loss_meters[f"{split}_" + k].add(v.item())
        loss_meters[f"{split}_total_loss"].add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred_idx = pred.max(1)
        correct = (pred_idx == batch["verb"]).float().sum()
        batch_acc = correct / pred.shape[0]
        loss_meters[f"{split}_bacc %"].add(batch_acc)

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
        f"{split.capitalize()} Avg. Total Loss",
        loss_meters[f"{split}_total_loss"].value()[0],
        epoch,
    )
    writer.add_scalar(
        f"{split.capitalize()} Avg. Batch Acc.",
        loss_meters[f"{split}_bacc %"].value()[0],
        epoch,
    )

    writer.add_scalar(
        f"{split.capitalize()} Avg. Attention Loss",
        loss_meters[f"{split}_attention_loss"].value()[0],
        epoch,
    )
    writer.add_scalar(
        f"{split.capitalize()} Avg. Class Loss",
        loss_meters[f"{split}_cls_loss"].value()[0],
        epoch,
    )
    writer.add_scalar(
        f"{split.capitalize()} Avg. Ant. Loss",
        loss_meters[f"{split}_ant_loss"].value()[0],
        epoch,
    )
    return loss_meters


# ----------------------------------------------------------------------------------------------------------------------------------------#

# TODO: Move the imports
import interaction_hotspots.data as data
from data import epic


# NOTE: Always load training set by default
def load_data(val=False, test=False):

    valset, valloader = None, None
    testset, testloader = None, None
    trainset = epic.EPICInteractions(
        root=data._DATA_ROOTS[args.dset],
        split="train",
        d_name=args.dset,
        max_len=args.max_len,
        size=args.resolution,
        dense_gaze=args.dense_gaze,
        gaussianSize=args.gaussianSize,
        normalize=True,
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
            dense_gaze=args.dense_gaze,
            gaussianSize=args.gaussianSize,
            normalize=True,
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=valset.data_sampler(),
        )

    if test:
        testset = epic.EPICInteractions(
            root=data._DATA_ROOTS[args.dset],
            split="test",
            d_name=args.dset,
            max_len=args.max_len,
            size=args.resolution,
            dense_gaze=args.dense_gaze,
            gaussianSize=args.gaussianSize,
            normalize=True,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=testset.data_sampler(),
        )

    return trainset, trainloader, valset, valloader, testset, testloader


def save_metrics(metrics, loss_file_name) -> None:

    loss_file = os.path.join(
        metrics_path,
        loss_file_name,
    )
    with open(loss_file, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)


# Dry run to find dimensions for the AvgPool2d kernel
backbone = backbones.dr50_n28()
backbone.eval()
dummy_input = torch.rand(1, 3, args.resolution, args.resolution)
spatial_dim = dummy_output = backbone(dummy_input).shape[-1]

trainset, trainloader, valset, valloader, testset, testloader = load_data(
    val=args.validate, test=args.test
)

# Class weights
train_labels = np.bincount(
    np.array([trainset.data[i]["verb"] for i in range(len(trainset))])
)
train_labels = {
    idx: np.sum(train_labels) / (len(train_labels) * train_labels[idx])
    for idx in range(len(train_labels))
}
class_weights = torch.tensor(list(train_labels.values()), dtype=torch.float32).cuda()

# Using triplet loss insead of MSE
ANT_LOSS = "triplet"

net = models[args.model]
torch.backends.cudnn.enabled = False
net = net(
    len(trainset.verbs),
    trainset.max_len,
    backbone=backbones.dr50_n28,
    ant_loss=ANT_LOSS,
    spatial_dim=spatial_dim,
)

# Load params - Remember to load pre-trained params for testing
if args.test and not args.checkpoint:
    raise ValueError("--checkpoint argument is required when --test flag is set")
net = load_params(net, args.checkpoint, len(trainset.verbs), args.finetune)

# NOTE: Transfer the model to GPU
net.cuda()

# NOTE: If multiple GPUs are available
if args.parallel:
    net = nn.DataParallel(net)

optim_params = list(filter(lambda p: p.requires_grad, net.parameters()))
logger.info("# params to optimize %s" % len(optim_params))

optimizer, scheduler = None, None

if not args.test:
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.decay_after], gamma=0.1
    )

# Paths for backup of params and metrics
try:
    subprocess.run(["mkdir", checkpoint_path])
    subprocess.run(["mkdir", metrics_path])
except:
    print("COULD NOT CREATE DIRECTORIES!")

log_path = os.path.join(
    f"/app/data/dataset/{args.dset}/runs",
    f"{args.model}_{datetime.now().strftime('%d%m%Y%H%M')}",
)

writer = SummaryWriter(log_dir=log_path)

# Containers for dumping of metrics
training_metrics = []
validation_metrics = []
test_metrics = []

# Comparison values
best_loss = np.inf
best_acc = 0
start_epoch = 1  # or load checkpoint
loss_file_name = f"{datetime.now().strftime('%d%m%Y%H%M')}_{args.max_epochs}_"

for epoch in range(start_epoch, args.max_epochs + 1):
    logger.info("LR = %.2E" % scheduler.get_lr()[0])

    # NOTE: Auxilary loss is exlcueded at it does not imporove model perfomance
    #
    if not args.test:
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

        save_metrics(
            training_metrics, loss_file_name + f"epochs_training_{net.name}.json"
        )
        scheduler.step()

        if epoch % args.max_epochs == 0:
            save_model(epoch, net, optimizer, checkpoint_path, f"final")
        if epoch % 3 == 0:
            save_model(epoch, net, optimizer, checkpoint_path, "checkpoint")
        if float(train_metrics["total_loss"].value()[0]) < best_loss:
            best_loss = float(train_metrics["total_loss"].value()[0])
            save_model(epoch, net, optimizer, checkpoint_path, "best_loss")
        if float(train_metrics["bacc %"].value()[0]) > best_acc:
            best_acc = float(train_metrics["bacc %"].value()[0])
            save_model(epoch, net, optimizer, checkpoint_path, "best_accuracy")

    if args.validate and not args.test:
        val_metrics = test_validate(epoch, writer, valloader, split="val")

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

        if epoch % args.max_epochs == 0:
            save_model(epoch, net, optimizer, checkpoint_path, f"final")
        if epoch % 3 == 0:
            save_model(epoch, net, optimizer, checkpoint_path, "checkpoint")
        if float(val_metrics["total_loss"].value()[0]) < best_loss:
            best_loss = float(val_metrics["total_loss"].value()[0])
            save_model(epoch, net, optimizer, checkpoint_path, "best_loss")
        if float(val_metrics["bacc %"].value()[0]) > best_acc:
            best_acc = float(val_metrics["bacc %"].value()[0])
            save_model(epoch, net, optimizer, checkpoint_path, "best_accuracy")

    if args.test:
        metrics = test_validate(1, writer, valloader, split="test")

        test_metrics.append(
            {
                "epoch": epoch,
                "total_loss": float(metrics["test_total_loss"].value()[0]),
                "cls_loss": float(metrics["test_cls_loss"].value()[0]),
                "accuracy": float(metrics["val_bacc %"].value()[0]),
            }
        )
        save_metrics(
            validation_metrics,
            loss_file_name + f"epochs_test_{net.name}.json",
        )

    #  COmpare training and validation metrics
    if not args.test and args.validate:
        writer.add_scalars(
            "Train/Val Total Loss",
            {
                "Train: Total Loss": float(train_metrics["total_loss"].value()[0]),
                "Val: Total Loss": float(val_metrics["val_total_loss"].value()[0]),
            },
            epoch,
        )
        writer.add_scalars(
            "Train/Val Attention Loss",
            {
                "Train: Attention Loss": float(
                    train_metrics["attention_loss"].value()[0]
                ),
                "Val: Attention Loss": float(
                    val_metrics["val_attention_loss"].value()[0]
                ),
            },
            epoch,
        )
        writer.add_scalars(
            "Train/Val Class Loss",
            {
                "Train: Class Loss": float(train_metrics["class_loss"].value()[0]),
                "Val: Class Loss": float(val_metrics["val_class_loss"].value()[0]),
            },
            epoch,
        )
        writer.add_scalars(
            "Train/Val Ant Loss",
            {
                "Train: Ant. Loss": float(train_metrics["ant_loss"].value()[0]),
                "Val: Ant. Loss": float(val_metrics["val_ant_loss"].value()[0]),
            },
            epoch,
        )
        writer.add_scalars(
            "Train/Val Accuracy",
            {
                "Train: Accuracy": float(train_metrics["bacc %"].value()[0]),
                "Val: Accuracy": float(val_metrics["val_bacc %"].value()[0]),
            },
            epoch,
        )


# Close files
writer.flush()
writer.close()
