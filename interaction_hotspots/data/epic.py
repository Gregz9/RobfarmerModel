import collections
import os
import sys
from re import A
from PIL import Image
import tqdm
import numpy as np
import json
import pickle
from typing import override, overload
import subprocess
import cv2 as cv
import torch
import random
from datetime import datetime

main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

from interaction_hotspots.utils import util
from interaction_hotspots.data.hotspot_dataset import VideoInteractions, HeatmapDataset
from interaction_hotspots.data.hotspot_dataset import generate_heatmaps
from preprocess.ho_types import FrameDetections
import interaction_hotspots.data as data

# ---------------------------------------------------------------------------------------------------#


class ImageLoader:
    def __init__(self, root, d_name, rgb_or_det="rgb"):

        self.d_name = d_name

        if rgb_or_det == "rgb" or rgb_or_det == "det":
            # self.frame_dir = "%s/frames_rgb_flow/rgb/train/" % (root)
            ignore_dirs = [
                "inactive_images",
                "visualizations",
                "New_Data_Graminor",
                "output",
                "checkpoints",
                "runs",
                "training_metrics",
                "train_heatmaps",
                "val_heatmaps",
                "videos",
                "gaze_heatmaps",
                "evaluation_metrics",
                "annotation_backup",
                "test",
            ]

            participant_dirs = [
                dir_name
                for dir_name in os.listdir(data._DATA_ROOTS[d_name])
                if dir_name not in ignore_dirs
                and os.path.isdir(os.path.join(data._DATA_ROOTS[d_name], dir_name))
            ]

            base_part_dirs = [
                os.path.join(root, participant, "rgb_frames")
                for participant in participant_dirs
            ]

            all_rgb_dirs = []
            dir_map = {}
            for dir_name in base_part_dirs:
                vid_dirs = [
                    os.path.join(dir_name, vid_dir) for vid_dir in os.listdir(dir_name)
                ]
                vid_names = [vid_name for vid_name in os.listdir(dir_name)]

                for vid_name in vid_names:
                    dir_map[vid_name] = [vid_ for vid_ in vid_dirs if vid_name in vid_][
                        0
                    ]

                all_rgb_dirs.extend(vid_dirs)

            self.all_rgb_dirs = dir_map
            self.frame_dirs = "rgb_frames"
            self.prefix = "frame_"

    def __call__(self, v_id, f_id):
        file_name = f"{self.prefix}{int(f_id):010d}.jpg"
        full_name = os.path.join(self.all_rgb_dirs[v_id], file_name)

        img = Image.open(full_name).convert("RGB")
        return img


def expand_box(img, box, expand):
    # add a little padding to the box
    width, height = int(box[2]) - int(box[0]), int(box[3]) - int(box[1])
    delta_x, delta_y = int(width * expand), int(height * expand)

    W, H = img.size
    xmin, ymin, xmax, ymax = box
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)

    pad_x_left = 0
    pad_x_right = 0
    if delta_x >= 0:
        pad_x_left = min(delta_x // 2, xmin)
        pad_x_right = min(delta_x // 2, W - xmax)
    pad_y_bot = 0
    pad_y_top = 0
    if delta_y >= 0:
        pad_y_top = min(delta_y // 2, ymin)
        pad_y_bot = min(delta_y // 2, H - ymax)

    W, H = img.size
    new_box = [
        max(xmin - pad_x_left, 0),
        max(ymin - pad_y_top, 0),
        min(xmax + pad_x_right, W - 1),
        min(ymax + pad_y_bot, H - 1),
    ]
    new_box = [xmin, ymin, xmax, ymax]

    return new_box


class CropPerturb:

    def __init__(self, root, d_name):
        self.loader = ImageLoader(root, d_name, "det")

    def __call__(self, entry):
        v_id, f_id, box = entry["v_id"], entry["f_id"], entry["bbox"]

        det_img = self.loader(v_id, f_id)

        expand = np.random.uniform(0.5, 1.5)
        xmin, ymin, xmax, ymax = expand_box(det_img, box, expand)
        box_W = xmax - xmin
        box_H = ymax - ymin

        tx = np.random.uniform(-0.2, 0.2)
        ty = np.random.uniform(-0.2, 0.2)
        offset_x, offset_y = tx * box_W, ty * box_H

        _xmin = xmin + offset_x
        _ymin = ymin + offset_y
        _xmax = xmax + offset_x
        _ymax = ymax + offset_y

        box = list(map(int, [_xmin, _ymin, _xmax, _ymax]))

        crop = det_img.crop(box)

        return crop


class GazeMapLoader:
    """
    Default shape is set to 28x28, becuase experiments with interpolation show that
    """

    def __init__(
        self,
        images,
        clips,
        d_name,
        sample_rate,
        shape=(28, 28),
        gaussianSize=99,
        **kwargs,
    ):
        self.clips = clips
        self.images = images
        # Images as in annotaion[f"{split}_images"]
        self.images = images
        self.shape = shape
        self.dataset_path = os.path.abspath(f"../../data/datasets/{d_name}")
        self.gazemap_path = os.path.join(self.dataset_path, "gaze_heatmaps")
        if not os.path.exists(self.gazemap_path):
            try:
                subprocess.run(["mkdir", self.gazemap_path])
            except:
                print(f"ERROR: Cannot create directory: {self.gazemap_path}")

        self.gaussianSize = gaussianSize
        self.sample_rate = sample_rate
        self.dense_gaze = kwargs["dense_gaze"]
        self.dataset = d_name
        self.split = kwargs["split"]
        self.normalize = kwargs["normalize"]

        self.evaluation = (
            kwargs["evaluation"] if kwargs["evaluation"] is not None else False
        )

        if not os.path.exists(os.path.join(self.gazemap_path, self.split)):
            try:
                subprocess.run(["mkdir", os.path.join(self.gazemap_path, self.split)])
            except:
                print(
                    f"ERROR: Cannot create directory: {os.path.join(self.gazemap_path, self.split)}"
                )

        if self.dense_gaze:
            self.gazemap_path = os.path.join(
                self.gazemap_path,
                self.split,
                f"{self.shape[0]}x{self.shape[1]}_dense",
            )
        else:
            self.gazemap_path = os.path.join(
                self.gazemap_path,
                self.split,
                f"{self.shape[0]}x{self.shape[1]}_{self.sample_rate}",
            )

        if not os.path.exists(self.gazemap_path):
            try:
                subprocess.run(["mkdir", self.gazemap_path])
            except:
                print(f"ERROR: Cannot create directory: {self.gazemap_path}")

        # Generate the heatmaps
        self.gen_gazemaps()

    def point2Heatmap(
        self,
        pointList,
        gaussianSize=99,
        normalize=True,
        heatmapShape=(900, 900),
        offset=(0, 0),
    ):
        canvas = np.zeros(heatmapShape)
        for p in pointList:
            if p[1] < heatmapShape[0] and p[0] < heatmapShape[1]:
                canvas[p[1]][p[0]] = 1
        g = cv.GaussianBlur(canvas, (gaussianSize, gaussianSize), 0, 0)
        if normalize:
            g = cv.normalize(
                g, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
            )

        # To avoid winning from dataloader
        g = np.dstack((g, g, g))
        return g

    def check_modified(self, mins=30):

        annot_path = f"../../data/datasets/{self.dataset}/annotation.json"
        if os.path.exists(annot_path) and os.path.exists(
            os.path.join(self.gazemap_path, "meta.json")
        ):
            mod_time = os.path.getmtime(annot_path)

            meta_file = open(os.path.join(self.gazemap_path, "meta.json"), "r")
            meta_data = json.load(meta_file)
            meta_file.close()

            meta_gen_time = datetime.strptime(
                meta_data["timestamp"], "%d-%m-%Y_%H-%M-%S"
            )

            # How long since meta_gen_time has been updated relative to
            time_since_update = mod_time - meta_gen_time.timestamp()
            return time_since_update > 0

        # Symbolizes missing files
        return False

    def gen_gazemaps(self):

        # If directory already exists and is populated, there's no need to (re-)generate these gazemaps
        # if (
        #     os.path.exists(self.gazemap_path) and len(os.listdir(self.gazemap_path)) > 0
        # ) and not self.check_modified():
        #     return

        for img, entry in zip(self.images, self.clips):
            # shape of bbox
            w, h = img["shape"]

            if self.dense_gaze:

                if self.evaluation:
                    gaze_points = [
                        (
                            int((img["gaze_points"][i][0] * self.shape[0])),
                            int((img["gaze_points"][i][1] * self.shape[1])),
                        )
                        for i in range(len(img["gaze_points"]))
                    ]

                else:
                    gaze_points = [
                        (
                            int((entry["gaze_points"][i][0] * w) * (self.shape[0] / w)),
                            int((entry["gaze_points"][i][1] * h) * (self.shape[1] / h)),
                        )
                        for i in range(0, entry["stop"] - entry["start"] + 1)
                    ]

                g = self.point2Heatmap(
                    gaze_points,
                    gaussianSize=self.gaussianSize,
                    heatmapShape=self.shape,
                    normalize=self.normalize,
                )

                gaze_path = os.path.join(self.gazemap_path, entry["v_id"])
                if not os.path.exists(gaze_path):
                    try:
                        subprocess.run(["mkdir", gaze_path])
                    except:
                        print("Already exists")

                gazemap_path = os.path.join(
                    gaze_path,
                    f"gazemap_{entry['v_id']}_{entry['start']}_{entry['stop']}",
                )

                np.save(gazemap_path, g)

            else:
                for frame in entry["frames"]:
                    idx = frame[1] - entry["start"]
                    gaze_points = [
                        (
                            int(entry["gaze_points"][i][0] * self.shape[1]),
                            int(entry["gaze_points"][i][1] * self.shape[0]),
                        )
                        for i in range(idx - self.sample_rate + 1, idx + 1)
                    ]
                    g = self.point2Heatmap(
                        gaze_points,
                        gaussianSize=self.gaussianSize,
                        heatmapShape=self.shape,
                        normalize=self.normalize,
                    )

                    gaze_path = os.path.join(self.gazemap_path, entry["v_id"])
                    if not os.path.exists(gaze_path):
                        try:
                            subprocess.run(["mkdir", gaze_path])
                        except:
                            print("Already exists")

                    gazemap_path = os.path.join(gaze_path, f"gazemap_{frame[1]:010d}")
                    np.save(gazemap_path, g)

        # Generate a meta-data file
        meta = {
            "timestamp": datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
            "dataset": self.dataset,
            "split": self.split,
            "sample_rate": self.sample_rate,
            "guassian_size": self.gaussianSize,
            "shape": self.shape,
            "dense_gaze": self.dense_gaze,
            "evaluation": self.evaluation,
        }

        meta_file = open(os.path.join(self.gazemap_path, "meta.json"), "w")
        json.dump(meta, meta_file, indent=2)
        meta_file.close()

    def __call__(self, frame, **kwargs):
        if isinstance(frame, tuple):
            v_id, f_id = frame
        else:
            v_id = frame["video_id"]
            # dummy value for f_id
            f_id = 0

        # Construction of path must be exactly the same for training-script and evaluation-script
        if kwargs["start"] is not None and kwargs["stop"] is not None:
            file_name = f"gazemap_{v_id}_{kwargs['start']}_{kwargs['stop']}.npy"
            # gazemap_path = os.path.join(self.gazemap_path, v_id, file_name)
        else:
            file_name = f"gazemap_{f_id:010d}.npy"

        gazemap_path = os.path.join(self.gazemap_path, v_id, file_name)

        gazemap = np.load(gazemap_path)
        # Always copy data to avoid errors when manipulating
        gazemap = np.array(gazemap, copy=True)
        return torch.from_numpy(gazemap).permute(2, 0, 1)


class EPICInteractions(VideoInteractions):

    def __init__(self, root, split, max_len, d_name, sample_rate=10, **kwargs):
        super().__init__(root, split, max_len, sample_rate, **kwargs)

        # annots = json.load(open("data/epic/annotations.json"))
        self.d_name = d_name
        annots = json.load(
            open(os.path.join(f"../../data/datasets/{d_name}/annotation.json"))
        )
        self.verbs, self.nouns = annots["verbs"], annots["nouns"]
        self.train_data, self.val_data, self.test_data = (
            annots["train_clips"],
            annots["val_clips"],
            annots["test_clips"],
        )

        self.images = annots[f"{self.split}_images"]
        # self.data = self.train_data if self.split == "train" else self.val_data

        if self.split == "train":
            self.data = self.train_data
        elif self.split == "val":
            self.data = self.val_data
        elif self.split == "test":
            self.data = self.test_data

        print(
            "Train data: %d | Val data: %d" % (len(self.train_data), len(self.val_data))
        )
        # NOTE: First sampling
        # Use every frame. For EPIC sample_rate = 10 --> 6fps
        for entry in self.train_data + self.val_data + self.test_data:
            # Due to the potential drift at the start of an action and towards its end, the first sampling frame,
            # and the last are excluded from being candidate frames
            entry["frames"] = [
                (entry["v_id"], f_id)
                for f_id in range(
                    entry["start"] + self.sample_rate,
                    entry["stop"] - self.sample_rate + 1,
                    self.sample_rate,
                )
            ]
            # NOTE: Due to the typical drift of the gaze at the start and towards the end of an action
            # exlucde the first and last entry in the the sampled frames.

            # Offset the frame indexes
            frame_idxs = [frame[1] - entry["start"] for frame in entry["frames"]]
            frame_nums = [frame[1] for frame in entry["frames"]]

            entry["gaze"] = [
                (entry["v_id"], f_id, entry["gaze_points"][idx])
                for idx, f_id in zip(frame_idxs, frame_nums)
            ]

        self.gazemap_loader = GazeMapLoader(
            self.images,
            self.data,
            d_name,
            self.sample_rate,
            shape=(kwargs["size"], kwargs["size"]),
            gaussianSize=(
                kwargs["gaussianSize"] if kwargs["gaussianSize"] is not None else 33
            ),
            dense_gaze=(
                kwargs["dense_gaze"] if kwargs["dense_gaze"] is not None else False
            ),
            split=self.split,
            evaluation=False,
            normalize=kwargs["normalize"] if kwargs["normalize"] is not None else True,
        )

        self.dense_gaze = (
            kwargs["dense_gaze"] if kwargs["dense_gaze"] is not None else False
        )

        self.rgb_loader = ImageLoader(self.root, d_name, "rgb")
        self.box_cropper = CropPerturb(self.root, d_name)

        self.inactive_images = annots["inactive_images"]
        self._load_detections()

    def _load_detections(self):

        # Directories that should be ignored during loading. If the number of directories increases beyond a certain point,
        # consider changing the list to directories that should be included.
        ignore_dirs = [
            "inactive_images",
            "visualizations",
            "New_Data_Graminor",
            "output",
            "checkpoints",
            "runs",
            "training_metrics",
            "train_heatmaps",
            "val_heatmaps",
            "videos",
            "gaze_heatmaps",
            "evaluation_metrics",
            "annotation_backup",
            "test",
        ]

        participant_dirs = [
            dir_name
            for dir_name in os.listdir(data._DATA_ROOTS[self.d_name])
            if dir_name not in ignore_dirs
            and os.path.isdir(os.path.join(data._DATA_ROOTS[self.d_name], dir_name))
        ]

        dets_dirs = [
            os.path.join(self.root, participant, "hand-objects")
            for participant in participant_dirs
        ]

        dets = [
            os.path.join(det_file, f)
            for det_file in dets_dirs
            for f in os.listdir(det_file)
            if "corrected" not in f
        ]

        video_detections = {}
        for f in dets:
            with open(f, "rb") as det:

                detections = [
                    FrameDetections.from_protobuf_str(s) for s in pickle.load(det)
                ]
                video_id = f.split("/")[-1].replace(".pkl", "").strip("_gaze")

                video_detections[video_id] = detections
        self.detections = video_detections

    @override
    def load_frame(self, frame):
        v_id, f_id = frame
        return self.rgb_loader(v_id, f_id)

    def load_gazemap(self, frame, **kwargs):
        if kwargs["start"] is not None and kwargs["stop"] is not None:
            return self.gazemap_loader(
                frame, start=kwargs["start"], stop=kwargs["stop"]
            )
        else:
            return self.gazemap_loader(frame, start=None, stop=None)

    @override
    def select_inactive_instances(self, entry):

        def select(noun):
            candidates = self.inactive_images[noun]
            img = candidates[np.random.randint(len(candidates))]
            crop = self.box_cropper(img)
            crop = self.img_transform(crop)

            detection = self.detections[img["v_id"]][int(img["f_id"]) - 1]
            gaze_point = [detection.gaze.x, detection.gaze.y]

            hands = detection.hands
            bboxes = [hand.bbox for hand in hands]

            ext_bboxes = []
            for bbox in bboxes:
                bbox = [bbox.left, bbox.top, bbox.right, bbox.bottom]
                ext_bboxes.append(bbox)

            if len(ext_bboxes) == 0:
                ext_bboxes = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            elif len(ext_bboxes) == 1:
                ext_bboxes = [ext_bboxes[0], [0.0, 0.0, 0.0, 0.0]]
            else:
                ext_bboxes = [ext_bboxes[0], ext_bboxes[1]]

            hand_centers = []
            for bbox in ext_bboxes:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                hand_centers.append([center_x, center_y])

            # hand_centers = np.array(hand_centers)

            return (
                crop,
                gaze_point,
                hand_centers,
            )

        pos_noun = self.nouns[entry["noun"]]

        candidate_nouns = list(self.inactive_images.keys())
        neg_noun = candidate_nouns[np.random.randint(len(candidate_nouns))]

        positive, pos_gaze, pos_hands = select(pos_noun)
        negative, neg_gaze, neg_hands = select(neg_noun)
        return positive, negative, pos_gaze, neg_gaze, pos_hands, neg_hands

    # __getitem__ is alreay implemented in VideoInteractions, however due to use of uninitialized
    # variables in VideoInteractions version of this method, it is overriden here.
    #
    def __getitem__(self, index):
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        entry = self.data[index]

        # --------------------------------------------------------------------------#
        # sample frames and load/transform
        frames = self.sample(entry["frames"])
        length = len(frames)

        # print(
        #     "THIS IS THE LENGTH OF FRAMES ONCE LOADED FROM ANNOT DICT : {len(frames)}"
        # )
        gaze_points = np.zeros((self.max_len, 2), dtype=np.float32)
        hand_locations = np.zeros((self.max_len, 2, 2), dtype=np.float32)
        idx = 0
        for frame in frames:

            v_id, f_num = frame[0], frame[1]

            video_dets = self.detections[v_id]
            detection = video_dets[f_num - 1]
            gaze_points[idx] = np.array(
                [detection.gaze.x, detection.gaze.y], dtype=np.float32
            )

            hands = self.detections[v_id][f_num - 1].hands

            bboxes = [hand.bbox for hand in hands]
            ext_bboxes = []

            for bbox in bboxes:
                bbox = [bbox.left, bbox.top, bbox.right, bbox.bottom]
                ext_bboxes.append(bbox)

            # When bounding boxes are not available
            if len(ext_bboxes) == 0:
                ext_bboxes = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            elif len(ext_bboxes) == 1:
                ext_bboxes = [ext_bboxes[0], [0.0, 0.0, 0.0, 0.0]]
            else:
                ext_bboxes = [ext_bboxes[0], ext_bboxes[1]]

            hand_centers = []
            for bbox in ext_bboxes:
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                hand_centers.append(np.array([center_x, center_y], dtype=np.float32))
            hand_centers = np.array(hand_centers)

            hand_locations[idx] = hand_centers
            idx += 1

            # hand_locations.append(hand_centers)

        # padded_hands = np.zeros((self.max_len, 2, 2))
        # hand_locations = np.array(hand_locations, dtype=np.float32)
        # padded_hands[: len(hand_locations), :] = hand_locations

        # gaze_points = np.array(gaze_points, dtype=np.float32)
        # padded_gaze = np.zeros((self.max_len, 2))
        # padded_gaze[: len(gaze_points), :] = gaze_points
        # padded_gaze = np.array(padded_gaze, dtype=np.float32)

        positive, negative, pos_gaze, neg_gaze, pos_hands, neg_hands = (
            self.select_inactive_instances(entry)
        )

        gaze_points_pos = np.array(pos_gaze, dtype=np.float32)
        gaze_points_neg = np.array(neg_gaze, dtype=np.float32)
        hand_points_pos = np.array(pos_hands, dtype=np.float32)
        hand_points_neg = np.array(neg_hands, dtype=np.float32)
        # Gazemaps should only be loaded when using the GazeLSTM model. TODO: Add check for which model is being used
        if self.dense_gaze:
            gazemaps = [
                self.load_gazemap(frame, start=entry["start"], stop=entry["stop"])
                for frame in frames
            ]
        else:
            gazemaps = [
                self.load_gazemap(frame, start=None, stop=None) for frame in frames
            ]

        loaded_frames = [self.load_frame(frame) for frame in frames]

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        loaded_frames = self.clip_transform(loaded_frames)  # (T, 3, 224, 224)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        gazemaps = self.gazemap_transform(gazemaps)
        # Gaze_maps should be zero padded too?

        assert (
            np.array(gazemaps).shape == np.array(loaded_frames).shape
        ), f"gazemaps shape: {np.array(gazemaps).shape}, loaded_frames shape: {np.array(loaded_frames).shape}, Frames: {frames}"

        instance = {
            "frames": loaded_frames,
            "gaze": gaze_points,
            "gazemaps": gazemaps,
            "hands": hand_locations,
            "verb": entry["verb"],
            "noun": entry["noun"],
            "length": length,
        }

        # --------------------------------------------------------------------------#
        # load the positive and negative images for the triplet loss
        # positive, negative = self.select_inactive_instances(entry)
        instance.update(
            {
                "positive": positive,
                "negative": negative,
                "pos_gaze": gaze_points_pos,
                "neg_gaze": gaze_points_neg,
                "pos_hands": hand_points_pos,
                "neg_hands": hand_points_neg,
            }
        )

        # --------------------------------------------------------------------------#
        return instance

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------------#


# NOTE: Remember to adapt paths to the container
class EPICHeatmaps(HeatmapDataset):
    def __init__(self, root, split, d_name, std_norm=True, **kwargs):
        self.d_name = d_name
        self.split = split
        self.size = kwargs["size"]
        base_path = os.path.expanduser("~/")
        hm_file = os.path.join(base_path, f"../../data/datasets/{d_name}/heatmaps.h5")
        # hm_file = "data/epic/heatmaps.h5"
        super().__init__(root, split, hm_file=hm_file, std_norm=std_norm)
        annot_path = os.path.join(
            base_path, f"../../data/datasets/{d_name}/annotation.json"
        )
        annots = json.load(open(annot_path))
        if not os.path.exists(hm_file):
            generate_heatmaps(
                annots,
                kernel_size=3.0,
                out_file=hm_file,
                transpose=True,
                split=self.split,
            )

        self.verbs = annots["verbs"]
        self.train_data, self.val_data, self.test_data = (
            annots["train_images"],
            annots["val_images"],
            annots["test_images"],
        )
        self.train_clips, self.val_clips, self.test_clips = (
            annots["train_clips"],
            annots["val_clips"],
            annots["test_clips"],
        )
        # self.data = self.train_data if self.split == "train" else self.val_data

        if self.split == "train":
            self.data = self.train_data
            self.clips = self.train_clips
        elif self.split == "val":
            self.data = self.val_data
            self.clips = self.val_clips
        elif self.split == "test":
            self.data = self.test_data
            self.clips = self.test_clips

        self.gazemap_loader = GazeMapLoader(
            self.data,
            self.clips,
            d_name,
            kwargs["sample_rate"],
            shape=(kwargs["size"], kwargs["size"]),
            gaussianSize=kwargs["gaussianSize"],
            dense_gaze=kwargs["dense_gaze"],
            split=self.split,
            evaluation=True,
            normalize=kwargs["normalize"],
        )
        print(
            "%d train images, %d val images, %d test imags"
            % (len(self.train_data), len(self.val_data), len(self.test_data))
        )

    def load_gazemap(self, entry, start=None, stop=None):
        return self.gazemap_loader(entry, start=start, stop=stop)

    def load_image_heatmap(self, entry):
        # splitName = entry["image"][0].split("_")
        video_id = entry["video_id"]
        path = os.path.join(
            f"../../data/datasets/{self.d_name}/inactive_images",
            video_id,
            entry["image"][0],
        )
        # path = "data/epic/images/%s" % (
        #     entry["image"][str(0)]
        # )  # P01_17_301_1084_92_1412_462 ...
        crop = util.load_img(path)

        hm_key = tuple(entry["image"]) + (str(entry["verb"]),)

        heatmap = self.heatmaps(hm_key)

        # Org: crop, heatmap. Crop is no longer used. It was removed due to incompatibility with current application
        _, heatmap = self.pair_transform(crop, heatmap, size=self.size)

        return _, heatmap

    def __getitem__(self, index):

        if self.heatmaps is None:
            self.heatmaps = self.init_hm_loader()

        entry = self.data[index]
        clip = self.clips[index]
        # If visible error from pyright, ignore. Only incorrect type reading in the background
        img, heatmap = self.load_image_heatmap(entry)
        # Use start and stop values from the corresponding clip
        gazemap = self.load_gazemap(entry, start=clip["start"], stop=clip["stop"])

        instance = {
            "img": img,
            "verb": entry["verb"],
            "heatmap": heatmap,
            "gazemap": gazemap,
            "image_name": entry["image"][0],
        }
        return instance

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------------#
