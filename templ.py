import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2 as cv
import os
import subprocess
import time

import util_funcs.image_processing as image_processing
from utilities import DATASET_BASE_PATH, read_participant_id

video_base_path = "../data/video/"
video_name = "scenevideo.mp4"
video_id = str(input("Enter video ID: "))
participant_id = read_participant_id(video_id)

frames_path = os.path.join(DATASET_BASE_PATH, participant_id, "rgb_frames", video_id)
save_path_base = os.path.join(DATASET_BASE_PATH, participant_id, "hand-landmarks")

num_files = (
    len(
        [
            f
            for f in os.listdir(frames_path)
            if os.path.isfile(os.path.join(frames_path, f))
        ]
    )
    + 1
)

video_path = os.path.join(video_base_path, video_id, video_name)

cap = cv.VideoCapture(video_path)
number = 1

file_count = len([name for name in os.listdir()])
hand_poses = {}
while cap.isOpened():
    print(
        "["
        + "#" * round((number / num_files) * 100)
        + "-" * round(((num_files - number) / num_files) * 100)
        + "]"
        + f" - Processing frame: {number}",
        end="\r",
    )
    ref, frame = cap.read()
    if not ref:
        break
    h, w, _ = frame.shape

    # BGR to RGB
    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    image_aug = image_processing.image_sharpening(image)
    image_aug = image_processing.gamma_correction(image, 1.5)
    edges_aug = cv.Canny(image_aug, 25, 80)
    # edges = cv.Canny(image, 25, 100)

    width = int(image.shape[0] / 1.5)
    height = int(image.shape[1] / 1.5)

    edges_aug = cv.resize(edges_aug, (height, width), cv.INTER_AREA)
    # edges = cv.resize(edges, (height, width), cv.INTER_AREA)
    # image = cv.resize(image, (height, width), cv.INTER_AREA)
    image_aug = cv.resize(image_aug, (height, width), cv.INTER_AREA)

    color = [255, 0, 0]

    # edges_aug_3d = np.stack([edges_aug] * 3, axis=2)
    mask = (edges_aug > 0).astype(np.uint8)
    # Expand to three color channels
    mask_3d = np.expand_dims(mask, axis=2)
    mask_3d = np.repeat(mask_3d, 3, axis=2)

    colored_mask = np.zeros_like(mask_3d)
    colored_mask[:, :] = color
    colored_mask *= mask_3d
    # colored_mask[mask == 1] = color

    image = cv.resize(frame, (height, width), cv.INTER_AREA)

    # image[mask == 1] = [0, 255, 50]
    # image_aug[mask == 1] = 255
    alpha = 0.85
    beta = 0.15
    image = cv.addWeighted(image, alpha, colored_mask, beta, 0)

    number += 1

    two_imgs = np.hstack([image_aug, edges_aug])
    # two_imgs_2 = np.hstack([image_aug, edges_aug])
    # four_imgs = np.vstack([two_imgs, two_imgs_2])
    cv.imshow("Edge image", image)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    # time.sleep(0.5)
    # Detections

# NOTE: Save all hand pose estimates
cap.release()
cv.destroyAllWindows()
