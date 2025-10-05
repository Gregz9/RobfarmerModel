import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import argparse

import torchnet as tnt
import collections
import logging
import subprocess
from datetime import datetime
import json

from utils import util
from utils import evaluation

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
parser.add_argument("--workers", type=int, default=8, help="Workers for dataloader")
parser.add_argument("--log_every", default=10, type=int, help="Logging frequency")
args = parser.parse_args()

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


epoch_losses = []


def train(epoch):

    net.train()

    iteration = 0
    total_iters = len(trainloader)
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))
    avg_acc = 0
    for batch in trainloader:

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

        if iteration % np.ceil(len(trainset) / args.batch_size) == 0 and iteration != 0:
            log_str = "epoch: %d + %d/%d | " % (epoch, iteration, total_iters)
            log_str += " | ".join(
                ["%s: %.3f" % (k, v.value()[0]) for k, v in loss_meters.items()]
            )
            logger.info(log_str)

        iteration += 1

    # return loss_meters, avg_acc / iteration
    net.eval()
    for batch in valloader:

        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

        loss_dict = {k : v.mean() for }


# ----------------------------------------------------------------------------------------------------------------------------------------#
import data
from data import opra, epic

if args.dset == "opra":
    trainset = opra.OPRAInteractions(
        root=data._DATA_ROOTS[args.dset], split="train", max_len=args.max_len
    )
    ant_loss = "mse"

elif args.dset == "epic":
    trainset = epic.EPICInteractions(
        root=data._DATA_ROOTS[args.dset], split="train", max_len=args.max_len
    )
    ant_loss = "triplet"


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=1,  # args.workers,
    sampler=trainset.data_sampler(),
)

from models import rnn, backbones

torch.backends.cudnn.enabled = False
net = rnn.frame_lstm(
    len(trainset.verbs),
    trainset.max_len,
    backbone=backbones.dr50_n28,
    ant_loss=ant_loss,
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

# loss_file = open("training_metrics.json", "w")

training_metrics = []

checkpoint_path = f"/home/gregz9/Desktop/MasterThesis/data/checkpoints/{datetime.now().strftime('%d%m%Y%H%M')}_{args.max_epochs}_epochs_base_model"
metrics_path = f"/home/gregz9/Desktop/MasterThesis/data/training_metrics/"
try:
    subprocess.run(["mkdir", checkpoint_path])
    subprocess.run(["mkdir", metrics_path])
except:
    print("COULD NOT CREATE DIRECTORIES!")


best_loss = np.inf
best_acc = 0
start_epoch = 1  # or load checkpoint
for epoch in range(start_epoch, args.max_epochs + 1):
    logger.info("LR = %.2E" % scheduler.get_lr()[0])

    loss_metrics, acc = train(epoch)

    training_metrics.append(
        {
            "epoch": epoch,
            "total_loss": float(loss_metrics["total_loss"].value()[0]),
            "cls_loss": float(loss_metrics["cls_loss"].value()[0]),
            "ant_loss": float(loss_metrics["ant_loss"].value()[0]),
            "aux_loss": float(loss_metrics["aux_loss"].value()[0]),
            "accuracy": float(acc),
        }
    )
    scheduler.step()

    if epoch % args.max_epochs == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(
                checkpoint_path,
                "base_model_{epoch}.pt",
            ),
        )

    if loss_metrics["total_loss"].value()[0].item() < best_loss:
        best_loss = loss_metrics["total_loss"].value()[0].item()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(
                checkpoint_path,
                f"base_model_best_loss.pt",
            ),
        )
    if acc > best_acc:
        best_acc = acc
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(
                checkpoint_path,
                f"base_model_best_acc.pt",
            ),
        )

# NOTE: Use try/except
json_object = json.dumps(training_metrics, indent=4)
loss_file = os.path.join(
    metrics_path,
    f"{datetime.now().strftime('%d%m%Y%H%M')}_{args.max_epochs}_epochs_base_model.json",
)
metrics_file = open(loss_file, "w")
json.dump(json_object, metrics_file)
metrics_file.close()
# if epoch == args.max_epochs:
#     save(epoch)
