from utilities import DataHandler
import argparse
import os
import json 
import subprocess 
import pandas as pd 
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("--dset", type=str, default="Robofarmer-II")
parser.add_argument("--vid_lvl_split", type=bool, default=False)

dataset_annots = {
    # "Robofarmer": "../data/Annotation.csv",
    "Robofarmer-II": "../data/datasets/Robofarmer-II/Annotation.csv",
}

def extract_inactive_images():
    inactive_imgs_path = "../data/datasets/Robofarmer-II/inactive_images"
    if not os.path.exists(inactive_imgs_path):
        try:
            subprocess.run(["mkdir", inactive_imgs_path])
        except:
            print("Could not create directory for inactive images")

    annot_df = pd.read_csv("../data/datasets/Robofarmer-II/Annotation.csv", dtype={"uid" : int, "start_action": int, "stop_action" : int})
    for idx, row in annot_df.iterrows():
        frame_name = row["inactive_frame_name"]
        bbox_coords = row["bbox_coords"]

        # Process coords
        bbox_coords = bbox_coords.split(",")
        bbox_coords = [int(coord) for coord in bbox_coords]
        
        # Find original image
        org_img_path = f"../data/datasets/Robofarmer-II/{row['participant_id']}/rgb_frames/{row['video_id']}/frame_{row['inactive']:010d}.jpg"
        img = cv.imread(org_img_path)
        
        inactive_img = img[bbox_coords[0]: bbox_coords[2]+1, bbox_coords[1] : bbox_coords[3]+1, :].copy()

        cv.imwrite(os.path.join(inactive_imgs_path, frame_name), inactive_img)
        

def post_processing():
    annotation_path = f"../data/datasets/{args.dset}/annotation.json"
    annotation = {}
    if os.path.exists(annotation_path):
        annotation_file = open(annotation_path, "r")
        annotation = json.load(annotation_file)

    train_images = annotation["train_images"]
    val_images = annotation["val_images"]
    train_images = annotation["train_images"]
    
    if not os.path.exists("inactive_images/train_images"):
        try:
            subprocess.run(["mkdir", "inactive_images/train_images"])
        except: 
            print("Error during creation of: inactive_images/train_images")

    if not os.path.exists("inactive_images/val_images"):
        try:
            subprocess.run(["mkdir", "inactive_images/val_images"])
        except: 
            print("Error during creation of: inactive_images/val_images")

    for img in train_images:

        video_id = img["video_id"]
        img_name = img["image"][0]

        try: 
            subprocess.run(
                [
                "cp",
                os.path.join("inactive_images", video_id, img_name),
                "inactive_images/train_images",
                ]
                )
        except:
            print(f"Error while copying image file: {os.path.join("inactive_images", video_id, img_name)}")

        for img in val_images:

            video_id = img["video_id"]
            img_name = img["image"][0]

            try: 
                subprocess.run(
                    [
                    "cp",
                    os.path.join("inactive_images", video_id, img_name),
                    "inactive_images/train_images",
                    ]
                    )
            except:
                print(f"Error while copying image file: {os.path.join("inactive_images", video_id, img_name)}")


if __name__ == "__main__":

    args = parser.parse_args()

    dataHandler = DataHandler(dataset_annots[args.dset], args.dset, video_level_split=args.vid_lvl_split)
    # dataHandler.dumpJsonAnnotation()
    extract_inactive_images()
