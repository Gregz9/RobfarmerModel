import sys
import os
import time
import argparse
from tqdm import tqdm
import cv2 as cv
import subprocess
from utilities import read_participant_id

DATASET_PATH = "data/datasets/Robofarmer"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_size", type=str, default="224x224", help="x separated size of image"
    )
    parser.add_argument(
        "--comp_algorithm",
        type=str,
        default="Lanczos",
        help="ref: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), DATASET_PATH),
        help="Path to directory with trianin images",
    )

    args = parser.parse_args()

    # Create a copy of the dataset folder
    dataset_name = args.path.split("/")[-1]

    dataset_compr = dataset_name + "_compressed"
    strip_part = "/" + dataset_compr
    stripped_base = args.path.strip(dataset_name)

    comp_dataset_path = os.path.join(stripped_base, dataset_compr)

    try:
        if not os.path.exists(comp_dataset_path):
            subprocess.run(["mkdir", comp_dataset_path])
        else:
            print("Compressed directory already exists")
    except:
        print("Could not create directory for the compressed dataset")

    video_id = str(input("Enter video id: "))
    participant_id = read_participant_id(video_id)

    # Creating full paths
    org_images_path = os.path.join(args.path, participant_id, "rgb_frames", video_id)
    compr_images_path = os.path.join(
        comp_dataset_path, participant_id, "rgb_frames", video_id
    )

    print(org_images_path)
    print(compr_images_path)

    image_paths = os.listdir(org_images_path)

    image_dims = args.image_size.split("x")

    for image_name in tqdm(image_paths):
        image_path = os.path.join(org_images_path, image_name)
        new_image_path = os.path.join(compr_images_path, image_name)
        image = cv.imread(image_path)
        image_resized = cv.resize(
            image, (int(image_dims[0]), int(image_dims[1])), cv.INTER_LANCZOS4
        )
        cv.imwrite(new_image_path, image_resized)
    # dirs = os.listdir(args.path)

    # all_paths = []
    # for directory in dirs:
    #     paths = []
    #     for dir_path, dir_names, file_names in os.walk(str(directory)):
    #         for name in file_names:
    #             file_path = os.path.join(dir_path, name)
    #             if os.path.islink(file_path):
    #                 continue
    #             paths.append(file_path)
    #     all_paths.extend(paths)
