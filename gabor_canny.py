
import numpy as np
import cv2 as cv
import os
import argparse
import json

import image_processing

# parser = argparse.ArgumentParser()

datasets_path = os.path.expanduser("~/Desktop/MasterThesis/data/datasets")
map_file = open(os.path.join(datasets_path,"Robofarmer-II/video_participants_acro.json"), "r")
video_partic_map = json.load(map_file)
inaactive_path = os.path.join(datasets_path, "Robofarmer-II/inactive_images/val_images")
inactive_val_images = sorted(os.listdir(inaactive_path))
video_list = list(video_partic_map.keys())
# print("Choose video: ") 
# for i, video in enumerate(inactive_videos):
#     print(f"{i}. {video}")
#
# exit()

# choice = int(input("Enter choice: "))

# choosen_video = video_list[choice]
# frames = sorted(os.listdir(os.path.join(datasets_path, "Robofarmer-II", video_partic_map[choosen_video] , "rgb_frames", choosen_video)))
# frames_path = os.path.join(datasets_path, "Robofarmer-II", video_partic_map[choosen_video] , "rgb_frames", choosen_video)
frames = inactive_val_images
frames_path = os.path.join(datasets_path, "Robofarmer-II/inactive_images/val_images")
idx = 0

while(True):
    frame = cv.imread(os.path.join(frames_path, frames[idx]))
    h, w, _ = frame.shape

    # BGR to RGB
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image_aug = image_processing.image_sharpening(image)
    image_aug = image_processing.gamma_correction(image_aug, 1.5)

    gabor_image = image_processing.gabor_edge_aug(image_aug)
    edges_gabor = cv.Canny(gabor_image, 50, 150)
    edges_org = cv.Canny(image_aug, 50, 150)
    # edges = cv.Canny(image, 25, 100)
    
    lines_gabor = cv.HoughLinesP(edges_gabor, 1, np.pi/180, threshold=10, minLineLength=50, maxLineGap=10)

    image_without_lines = edges_gabor.copy()

# Remove the detected lines by drawing over them with a specific color (e.g., white)
    if lines_gabor is not None:
        for line in lines_gabor:
            x1, y1, x2, y2 = line[0]
            # Draw a black line over the detected line to remove it
            cv.line(image_without_lines, (x1, y1), (x2, y2), (0, 0, 0), 5) # 5 is the thickness

    width = 600 #int(image.shape[0] / 2)
    height = 600 #int(image.shape[1] / 2)

    edges_org = cv.resize(edges_org, (height, width), cv.INTER_CUBIC)
    org_frame = cv.resize(frame, (height, width), cv.INTER_CUBIC)
    edges_gabor = cv.resize(edges_gabor, (height, width), cv.INTER_CUBIC)
    gabor_image = cv.resize(gabor_image, (height, width), cv.INTER_CUBIC)
    
    gabor_no_lines = cv.resize(image_without_lines, (height, width), cv.INTER_CUBIC)

    gabor_image = cv.cvtColor(gabor_image, cv.COLOR_RGB2BGR)
    edges_gabor = cv.cvtColor(edges_gabor, cv.COLOR_GRAY2RGB)
    gabor_no_lines = cv.cvtColor(gabor_no_lines, cv.COLOR_GRAY2RGB)
    two_imgs = np.hstack([org_frame, edges_gabor])
    cv.imshow("Edge image", two_imgs)
    key = cv.waitKey(40) & 0xFF
    if key == ord("q"):
        break
        # exit()
    elif key == 83:
        idx = idx + 1
    elif key == 81:
        idx = idx - 1
    # time.sleep(0.5)
    # Detections
# NOTE: Save all hand pose estimates
cv.destroyAllWindows()
