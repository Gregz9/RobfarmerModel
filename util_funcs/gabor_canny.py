import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2 as cv
import os
import subprocess
import time

import image_processing
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
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image_aug = image_processing.image_sharpening(image)
    image_aug = image_processing.gamma_correction(image_aug, 1.5)

    gabor_image = image_processing.gabor_edge_aug(image_aug)
    edges_gabor = cv.Canny(gabor_image, 25, 80)
    edges_org = cv.Canny(image_aug, 25, 80)
    # edges = cv.Canny(image, 25, 100)

    width = int(image.shape[0] / 2)
    height = int(image.shape[1] / 2)

    edges_org = cv.resize(edges_org, (height, width), cv.INTER_AREA)
    edges_gabor = cv.resize(edges_gabor, (height, width), cv.INTER_AREA)
    gabor_image = cv.resize(gabor_image, (height, width), cv.INTER_AREA)

    number += 1

    gabor_image = cv.cvtColor(gabor_image, cv.COLOR_RGB2BGR)
    edges_gabor = cv.cvtColor(edges_gabor, cv.COLOR_GRAY2RGB)
    two_imgs = np.hstack([gabor_image, edges_gabor])
    cv.imshow("Edge image", two_imgs)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    # time.sleep(0.5)
    # Detections

# NOTE: Save all hand pose estimates
cap.release()
cv.destroyAllWindows()
