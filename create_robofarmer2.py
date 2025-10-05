import os
import subprocess
import json
import cv2 as cv

if __name__ == "__main__":

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
                print(f"Error occured during creation of directory: {os.path.join(dataset_path, v)}")
        if not os.path.exists(os.path.join(dataset_path, v, "hand-landmarks")):
            try: 
                subprocess.run(["mkdir", os.path.join(dataset_path, v, "hand-landmarks")])
            except:
                print(f"Error occured during creation of directory: {os.path.join(dataset_path, v, 'hand-landmarks')}")

        if not os.path.exists(os.path.join(dataset_path, v, "hand-objects")):
            try: 
                subprocess.run(["mkdir", os.path.join(dataset_path, v, "hand-objects")])
            except:
                print(f"Error occured during creation of directory: {os.path.join(dataset_path, v, 'hand-objects')}")

        if not os.path.exists(os.path.join(dataset_path, v, "rgb_frames")):
            try: 
                subprocess.run(["mkdir", os.path.join(dataset_path, v, "rgb_frames")])
            except:
                print(f"Error occured during creation of directory: {os.path.join(dataset_path, v, 'rgb_frames')}")

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

        # Saving path to each of images
        image_map[k2] = []

        # Extract frames from the video
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
            print(f"Error occured during creation of file: {video_participant_acronym_path}")
   
    image_path_map = os.path.join(dataset_path, "image_path_map2.json")
    if not os.path.exists(image_path_map):
        try:
            with open(image_path_map, "w") as f:
                json.dump(image_map, f, indent=4)
            f.close()
        except:
            print(f"Error occured during creation of file: {image_path_map}")
        


    # Extract images from videos
