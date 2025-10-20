from utilities import DataHandler
import argparse
import os
import json
import subprocess
import pandas as pd
import cv2 as cv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dset", type=str, default="Robofarmer-II")
parser.add_argument("--vid_lvl_split", type=bool, default=False)
parser.add_argument(
    "--create_structure",
    action="store_true",
    help="Tells the program to create the dateset structure",
)
parser.add_argument(
    "--extract_inactive",
    action="store_true",
    help="Tells the program to extract inactive frames using Annotation.csv file and place them in inactive_images directory",
)
parser.add_argument(
    "--split_inactive",
    action="store_true",
    help="Tells the program to organize the inactive images according to the train|val|test split defined in annotation.json",
)

dataset_annots = {
    # "Robofarmer": "../data/Annotation.csv",
    "Robofarmer-II": "../data/datasets/Robofarmer-II/Annotation.csv",
}


def create_dataset_structure():

    # Current directory should be src
    base_path = os.path.dirname(os.getcwd())
    dataset_path = os.path.join(base_path, "data/datasets/Robofarmer-II")
    videos_path = os.path.join(dataset_path, "videos")

    participant_names = []
    names_mapping = {}
    video_participant_map = {}
    video_participant_acronym_map = {}
    idx = 1
    for directory in [
        directory
        for directory in os.listdir(videos_path)
        if os.path.isdir(os.path.join(videos_path, directory))
    ]:

        meta_file = open(
            os.path.join(videos_path, directory, "meta/participant"),
            "r",
        )
        file_dir = json.loads(meta_file.readlines()[0])
        # print(file_dir, directory)

        name = file_dir["name"]
        if "_" in name:
            name = name.split("_")[0]
            participant_names.append(name)
        elif "-" in name:
            name = name.split("-")[0]
            participant_names.append(name)
        elif " " in name:
            name = name.split(" ")[0]
            participant_names.append(name)

        name = "".join(filter(lambda x: not x.isdigit(), name)).capitalize()
        video_participant_map[directory] = name

        if name not in names_mapping:
            names_mapping[name] = "P0" + str(idx)
            idx += 1

        video_participant_acronym_map[directory] = names_mapping[name]

    image_map = {}
    for k, v in video_participant_acronym_map.items():
        image_map[k] = []
    # Create participant directories in Robofarmer-II
    # try:
    for k, v in names_mapping.items():
        if not os.path.exists(os.path.join(dataset_path, v)):
            try:
                subprocess.run(["mkdir", os.path.join(dataset_path, v)])
            except:
                print(
                    f"Error occured during creation of directory: {os.path.join(dataset_path, v)}"
                )
        if not os.path.exists(os.path.join(dataset_path, v, "hand-landmarks")):
            try:
                subprocess.run(
                    ["mkdir", os.path.join(dataset_path, v, "hand-landmarks")]
                )
            except:
                print(
                    f"Error occured during creation of directory: {os.path.join(dataset_path, v, 'hand-landmarks')}"
                )

        if not os.path.exists(os.path.join(dataset_path, v, "hand-objects")):
            try:
                subprocess.run(["mkdir", os.path.join(dataset_path, v, "hand-objects")])
            except:
                print(
                    f"Error occured during creation of directory: {os.path.join(dataset_path, v, 'hand-objects')}"
                )

        if not os.path.exists(os.path.join(dataset_path, v, "rgb_frames")):
            try:
                subprocess.run(["mkdir", os.path.join(dataset_path, v, "rgb_frames")])
            except:
                print(
                    f"Error occured during creation of directory: {os.path.join(dataset_path, v, 'rgb_frames')}"
                )

    for k2, v2 in video_participant_acronym_map.items():
        # if k == v2:
        vid_images_path = os.path.join(dataset_path, v2, "rgb_frames", k2)
        try:
            subprocess.run(["mkdir", vid_images_path])
        except:
            print(f"File {vid_images_path} already exists")

        print(os.path.join(videos_path, k2, "scenevideo.mp4"))
        vidcap = cv.VideoCapture(os.path.join(videos_path, k2, "scenevideo.mp4"))
        success, image = vidcap.read()
        count = 1

        # Get total frame count for tqdm
        total_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))

        # Saving path to each of images
        image_map[k2] = []

        # Extract frames from the video
        with tqdm(total=total_frames, desc=f"Extracting frames from {k2}") as pbar:
            while success:
                file_name = os.path.join(vid_images_path, f"frame_{count:010d}.jpg")
                image_map[k2].append(file_name)

                cv.imwrite(file_name, image)
                print(
                    f"Reading of frame_{count:010d}.jpg was successful: {success}",
                    end="\r",
                )
                count += 1
                success, image = vidcap.read()
                pbar.update(1)

        # Sorting paths
        image_map[k2] = sorted(image_map[k2])

    # Creating convenience files

    participant_path = os.path.join(dataset_path, "participants.json")
    if not os.path.exists(participant_path):
        try:
            with open(participant_path, "w") as f:
                json.dump(names_mapping, f, indent=4)
            f.close()
        except:
            print(f"Error occured during creation of file: {participant_path}")

    video_participant_path = os.path.join(dataset_path, "video_participants.json")
    if not os.path.exists(video_participant_path):
        try:
            with open(video_participant_path, "w") as f:
                json.dump(video_participant_map, f, indent=4)
            f.close()
        except:
            print(f"Error occured during creation of file: {video_participant_path}")

    video_participant_acronym_path = os.path.join(
        dataset_path, "video_participants_acro.json"
    )
    if not os.path.exists(video_participant_acronym_path):
        try:
            with open(video_participant_acronym_path, "w") as f:
                json.dump(video_participant_acronym_map, f, indent=4)
            f.close()
        except:
            print(
                f"Error occured during creation of file: {video_participant_acronym_path}"
            )

    image_path_map = os.path.join(dataset_path, "image_path_map2.json")
    if not os.path.exists(image_path_map):
        try:
            with open(image_path_map, "w") as f:
                json.dump(image_map, f, indent=4)
            f.close()
        except:
            print(f"Error occured during creation of file: {image_path_map}")


def extract_inactive_images():
    inactive_imgs_path = "../data/datasets/Robofarmer-II/inactive_images"
    if not os.path.exists(inactive_imgs_path):
        try:
            subprocess.run(["mkdir", inactive_imgs_path])
        except:
            print("Could not create directory for inactive images")

    annot_df = pd.read_csv(
        "../data/datasets/Robofarmer-II/Annotation.csv",
        dtype={"uid": int, "start_action": int, "stop_action": int},
    )
    for idx, row in annot_df.iterrows():
        frame_name = row["inactive_frame_name"]
        bbox_coords = row["bbox_coords"]

        # Process coords
        bbox_coords = bbox_coords.split(",")
        bbox_coords = [int(coord) for coord in bbox_coords]

        # Find original image
        org_img_path = f"../data/datasets/Robofarmer-II/{row['participant_id']}/rgb_frames/{row['video_id']}/frame_{row['inactive']:010d}.jpg"
        img = cv.imread(org_img_path)

        inactive_img = img[
            bbox_coords[0] : bbox_coords[2] + 1, bbox_coords[1] : bbox_coords[3] + 1, :
        ].copy()

        cv.imwrite(os.path.join(inactive_imgs_path, frame_name), inactive_img)


def split_inactive_imgs():
    annotation_path = f"../data/datasets/{args.dset}/annotation.json"
    dataset_path = f"../data/datasets/{args.dset}"
    annotation = {}
    if os.path.exists(annotation_path):
        annotation_file = open(annotation_path, "r")
        annotation = json.load(annotation_file)

    train_images = annotation["train_images"]
    val_images = annotation["val_images"]
    test_images = annotation["test_images"]

    if not os.path.exists(os.path.join(dataset_path, "inactive_images/train_images")):
        try:
            subprocess.run(
                ["mkdir", os.path.join(dataset_path, "inactive_images/train_images")]
            )
        except:
            print("Error during creation of: inactive_images/train_images")

    if not os.path.exists(os.path.join(dataset_path, "inactive_images/val_images")):
        try:
            subprocess.run(
                ["mkdir", os.path.join(dataset_path, "inactive_images/val_images")]
            )
        except:
            print("Error during creation of: inactive_images/val_images")

    if not os.path.exists(os.path.join(dataset_path, "inactive_images/test_images")):
        try:
            subprocess.run(
                ["mkdir", os.path.join(dataset_path, "inactive_images/test_images")]
            )
        except:
            print("Error during creation of: inactive_images/test_images")

    for img in train_images:

        video_id = img["video_id"]
        img_name = img["image"][0]

        try:
            subprocess.run(
                [
                    "cp",
                    os.path.join(dataset_path, "inactive_images", video_id, img_name),
                    os.path.join(dataset_path, "inactive_images/train_images"),
                ]
            )
        except:
            print(
                f"Error while copying image file: {os.path.join("inactive_images", video_id, img_name)}"
            )

    for img in val_images:

        video_id = img["video_id"]
        img_name = img["image"][0]

        try:
            subprocess.run(
                [
                    "cp",
                    os.path.join(dataset_path, "inactive_images", video_id, img_name),
                    os.path.join(dataset_path, "inactive_images/val_images"),
                ]
            )
        except:
            print(
                f"Error while copying image file: {os.path.join("inactive_images", video_id, img_name)}"
            )

    for img in test_images:

        video_id = img["video_id"]
        img_name = img["image"][0]

        try:
            subprocess.run(
                [
                    "cp",
                    os.path.join(
                        dataset_path,
                        "inactive_images",
                        video_id,
                        img_name,
                    ),
                    os.path.join(dataset_path, "inactive_images/test_images"),
                ]
            )
        except:
            print(
                f"Error while copying image file: {os.path.join("inactive_images", video_id, img_name)}"
            )


if __name__ == "__main__":

    args = parser.parse_args()

    if args.create_structure:
        create_dataset_structure()

    dataHandler = DataHandler(
        dataset_annots[args.dset], args.dset, video_level_split=args.vid_lvl_split
    )
    dataHandler.dumpJsonAnnotation()

    if args.extract_inactive:
        extract_inactive_images()
    if args.split_inactive:
        split_inactive_imgs()
