from utilities import DataHandler
import argparse
import os
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--dset", type=str, default="Robofarmer-II")

dataset_annots = {
    # "Robofarmer": "../data/Annotation.csv",
    "Robofarmer-II": "../data/datasets/Robofarmer-II/Annotation.csv",
}

def post_processing():
    annotation_path = f"../data/datasets/{args.dset}/annotation.json"
    annotation = {}
    if os.path.exists(annotation_path):
        annotation_file = open(annotation_path, "r")
        annotation = json.load(annotation_file)

    train_images = annotation["train_images"]
    val_images = annotation["test_images"]
    
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

    dataHandler = DataHandler(dataset_annots[args.dset], args.dset)
    dataHandler.dumpJsonAnnotation()
    # Move inactive images to correct directories
    post_processing()
