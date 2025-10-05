import os
import sys
import cv2
import subprocess
import json

paths = "../../data/paths.json"
with open(paths, "r") as file:
    data = json.load(file)

print(json.dumps(data, indent=4))

data_base_path = "../../data/"
dataset_base_path = "../../data/dataset"
robofarmer_path = "../../data/datasets/Robofarmer"

video_folder = str(input("Enter video ID: "))
participant_file = open(
    os.path.join(data_base_path, "video", video_folder, "meta/participant.json")
)
participant = json.load(participant_file)["participant"]
# video_file = str(input("Enter name of video file: "))
video_file = "scenevideo.mp4"

data[participant] = os.path.join(robofarmer_path, participant)
if participant not in list(data.keys()):
    data[participant] = os.path.join(robofarmer_path, participant)

    with open(paths, "w") as file:
        json.dump(data, file, indent=4)

    file.close()

while video_folder.lower() != "q" or video_folder != "":

    if not os.path.exists(data[participant]):
        try:
            print(f"Creating data directories for: \n {participant}")
            subprocess.run(["mkdir", os.path.join(data[participant])])
            subprocess.run(["mkdir", os.path.join(data[participant], "rgb_frames")])
            subprocess.run(["mkdir", os.path.join(data[participant], "hand-objects")])
            subprocess.run(["mkdir", os.path.join(data[participant], "labels")])
        except:
            print("Could not create directory")

    video_save_path = os.path.join(data[participant], "rgb_frames", video_folder)
    print(video_save_path)
    if not os.path.exists(video_save_path):
        try:
            print(f"Creating directory: \n {video_save_path}")
            subprocess.run(["mkdir", video_save_path])
        except:
            print(f"Could not create: {video_save_path}")

    # video_path = data["video"] + video_folder + "/scenevideo.mp4"
    video_path = os.path.join(data["video"], video_folder, video_file)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    print(success)
    count = 1
    while success:
        # Save the frame with the new name format
        filename = os.path.join(video_save_path, f"frame_{count:010d}.jpg")
        cv2.imwrite(filename, image)
        print(
            f"Reading of video frame {count:010d}.jpg was successful: {success}",
            end="\r",
        )
        count += 1
        success, image = vidcap.read()
    print("\n")

    video_folder = str(input("Enter video ID: "))
    if video_folder == "q":
        exit()
    participant_file = open(
        os.path.join(data_base_path, "video", video_folder, "meta/participant.json")
    )
    participant = json.load(participant_file)["participant"]
    # video_file = str(input("Enter name of video file: "))
    video_file = "scenevideo.mp4"

    data[participant] = os.path.join(robofarmer_path, participant)
    if participant not in list(data.keys()):
        data[participant] = os.path.join(robofarmer_path, participant)

        with open(paths, "w") as file:
            json.dump(data, file, indent=4)

        file.close()
