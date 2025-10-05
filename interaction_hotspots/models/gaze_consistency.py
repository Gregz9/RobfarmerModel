import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import cv2 as cv
from utilities import load_jsonl_data, findTime


class AttentionUncertainty(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionUncertainty, self).__init__()

        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(feature_dim, 1)
        )

    def forward(self, attention_maps):

        self.sigma = self.uncertainty_head(attention_maps)

        return self.sigma


def attentionConsistencyLoss(attention_maps, gaze_maps, sigma):

    return (1 / (2 * sigma**2)) * torch.mean(
        (attention_maps - gaze_maps) ** 2
    ) + torch.log(sigma)


def refine_cams(cam_original, image_shape):

    if image_shape[0] != cam_original.size(2) or image_shape[1] != cam_original.size(3):
        cam_original = F.interpolate(
            cam_original, image_shape, mode="bilinear", align_corners=True
        )
    B, C, H, W = cam_original.size()
    cams = []
    for idx in range(C):
        cam = cam_original[:, idx, :, :]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        norm = cam_max - cam_min
        norm[norm == 0] = 1e-5
        cam = (cam - cam_min) / norm
        cam = cam.view(B, H, W).unsqueeze(1)
        cams.append(cam)
    cams = torch.cat(cams, dim=1)
    sigmoid_cams = torch.sigmoid(100 * (cams - 0.4))
    return cams, sigmoid_cams


# Note example usage
def _forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    fmap = self.layer4(x)
    x = self.avgpool(fmap)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    weight = self.fc.weight

    # NOTE: If the shapes match, Using a 1x1 convolution with stride=1 is equivalent to a dot product
    cam = F.conv2d(fmap, weight.detach().unsqueeze(2).unsqueeze(3), bias=None)

    cams, sigmoid_cams = refine_cams(cam, self.image_shape)
    return x, cams, sigmoid_cams


def point2Heatmap(
    pointList, gaussianSize=99, normalize=True, heatmapShape=(900, 900), offset=(0, 0)
):
    canvas = np.zeros(heatmapShape)
    for p in pointList:
        if p[1] < heatmapShape[0] and p[0] < heatmapShape[1]:
            canvas[p[1]][p[0]] = 255
    g = cv.GaussianBlur(canvas, (gaussianSize, gaussianSize), 0, 0)
    if normalize:
        g = cv.normalize(
            g, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
    return g


def overlayHeatmap(image, heatmap, alpha=0.3):
    heatmap_colored = cv.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv.COLORMAP_JET
    )
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv.resize(heatmap_colored, (image.shape[1], image.shape[0]))

    result = cv.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return result


def load_gaze(path):
    gaze_data = load_jsonl_data(path)

    gaze_data = list(
        filter(lambda x: len(x["data"]) > 0 and "gaze2d" in x["data"], gaze_data)
    )

    return gaze_data


def augmentPoints(*points):

    avg_x, avg_y = np.mean(np.array([point[0] for point in points])), np.mean(
        np.array([point[1] for point in points])
    )

    low_x, high_x = 0.95 * avg_x, 1.05 * avg_x
    low_y, high_y = 0.95 * avg_y, 1.05 * avg_y

    gen_x = np.random.uniform
    gen_y = np.random.uniform

    return [(int(gen_x(low_x, high_x)), int(gen_y(low_y, high_y))) for i in range(10)]


from torchvision.models import resnet18, ResNet18_Weights

if __name__ == "__main__":

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    # NOTE: a way of removing layers from a model? -> Is it working perfectly for all types of architectures?
    backbone = torch.nn.Sequential(*list(model.children())[:-2])
    rand_tensor = torch.rand(16, 64, 3, 224, 224)

    shape = rand_tensor.shape
    flat = (shape[0] * shape[1],) + shape[2:]

    model.eval()
    x = rand_tensor.view(flat)
    print(x.shape)
    out = backbone(x)
    print(out.shape)
    u_flat = (
        shape[0],
        shape[1],
    ) + out.shape[1:]

    output = out.view(u_flat)
    print(f"The shape of out: {out.shape}")
    print(f"The shape of output {output.shape}")

    weights = model.fc.weight.detach().unsqueeze(2).unsqueeze(3)
    weights_copy = model.fc.weight.detach().numpy()

    print(f"The shape of weights: {weights.shape}")
    # print(f"The shape of weights copy: {weights_copy.shape}")
    cam = F.conv2d(out, weights, bias=None)

    print(f"Print shape of CAMS after conv2d: {cam.shape}")

    # out2 = out.clone().data.cpu().numpy()
    #
    # # NOTE: Simulated indexes
    # sample_classes = np.random.randint(0, 1000, size=(1024))
    #
    # cams = []
    # for cls_idx in sample_classes:
    #     # print(weights_copy[cls_idx].shape)
    #     # print(out2[0].reshape((out.shape[1], out.shape[2] * out.shape[3])).shape)
    #     cams.append(
    #         weights_copy[cls_idx].dot(
    #             out2[0].reshape((out.shape[1], out.shape[2] * out.shape[3]))
    #         )
    #     )
    #     # cams2 = weights_copy.dot(
    #     #     out2.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
    #     # )
    # cams = np.array(cams)
    #
    # print(f"Shape of cams: {cam.shape}")
    # print(f"Shape of cams2: {cams.shape}")

    """
    How to process cams when working with time series - videos in this case?

    Similarily to passing all the images through the backbone CNN, gathre the batch dimension B and the temporal dimension T:

    (B, T, Y) -> (B*T, Y) -> Module  -> (B*T, Y/channels)

    When using torch.nn.functional.conv2d you will have to choose the mask from among all the generated which belongs to
    the currently predicted class. In code that means:

    weights = fc.weight.detach()
    cams = F.conv2d(fmaps, weights.unsqueeze(1).unsqueeze(-1), bias=None)

    The shape of the resulting cams will be: [B*T, C, H*W] where C is not representing channels, but rather classes. In other words, using
    convolution we have computed an class attention map for each of the possible classes. So when backpropagating, we'll do the following:

    active_cams = cams[torch.arange(B), true_batch_labels]

    loss = consistency_loss(active_cams, gaze_maps)

    As to the way gaze_maps will be computed, instead of using the gaze at one point, we'll focus at using gaze at the intervals between
    the sampled frames. Currently we're using 8 frames for the sequence. For that to be a viable solution, the sampling process of frames
    will have to be exactly replicated.
    """

    # NOTE: This part must be seeded in order to be reproducible
    # def sample(self, clip):
    #     np.random.seed(1337)
    #     if len(clip) <= self.max_len:
    #         return clip
    #     # Just a default value
    #     st = 0
    #     if self.split == "train":
    #         st = np.random.randint(0, len(clip) - self.max_len)
    #     elif self.split == "val":
    #         st = len(clip) // 2 - self.max_len // 2
    #     clip = clip[st : st + self.max_len]
    #
    #     return clip

    # NOTE: Simulate loading frames just like in the data preparation
    # annotation = json.load(
    #     open(
    #         os.path.expanduser(
    #             f"~/Desktop/MasterThesis/data/datasets/Robofarmer-II/annotation.json"
    #         ),
    #         "r",
    #     )
    # )
    #
    # data = annotation["train_clips"] + annotation["test_clips"]
    #
    # for entry in data:
    #     entry["frames"] = [
    #         (entry["v_id"], f_id)
    #         for f_id in range(entry["start"], entry["stop"] + 1, 5)
    #     ]
    #
    # base_path = os.path.expanduser("~/Desktop/MasterThesis/data/datasets/Robofarmer-II")
    # gaze_base_path = os.path.expanduser(
    #     "~/Desktop/MasterThesis/data/datasets/Robofarmer-II/videos"
    # )
    # p_id = data[3]["p_id"]
    # v_id = data[3]["v_id"]
    # frames = data[3]["frames"]
    # # print(data[3])
    # first_frame = frames[1]
    # prev_frame = frames[0]
    # img = cv.imread(
    #     os.path.join(
    #         base_path, p_id, "rgb_frames", v_id, f"frame_{int(first_frame[1]):010d}.jpg"
    #     )
    # )
    # h, w, _ = img.shape
    # img = cv.resize(img, (int(w / 2), int(h / 2)), interpolation=cv.INTER_CUBIC)
    # h, w, _ = img.shape
    #
    # gaze_data = load_gaze(os.path.join(gaze_base_path, v_id, "gazedata"))

    # timestamps = [dat["timestamp"] for dat in gaze_data]
    # timestep = first_frame[1]
    #
    # # prev_timestep = second_frame[1]
    #
    # gaze_points = [
    #     gaze_data[findTime(t / 25, timestamps)]["data"]["gaze2d"]
    #     for t in range(timestep + 2)
    # ]
    # print(gaze_points)
    #
    # gaze_points = [(int(point[0] * w), int(point[1] * h)) for point in gaze_points]
    #
    # # add_points = augmentPoints(gaze_points)
    #
    # heatmap = point2Heatmap(gaze_points, heatmapShape=(h, w))
    #
    # frame = overlayHeatmap(img, heatmap)
    #
    # while True:
    #
    #     cv.imshow("Gaze map", frame)
    #
    #     key = cv.waitKey(40) & 0xFF
    #     if key == ord("q"):
    #         cv.destroyAllWindows()
    #         exit()

    # cams, sigmoid_cams = refine_cams(rand_tensor, (300, 300))

    # layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1))
    # out_cams = layer(sigmoid_cams)
