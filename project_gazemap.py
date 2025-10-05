import os
import cv2 as cv
from tqdm import tqdm
import numpy as np
import json
import pandas as pd

from utilities import (
    findTime,
    load_jsonl_data,
    DATASET_BASE_PATH,
)

COLORMAPS = {
    "Autumn" : cv.COLORMAP_AUTUMN,
    "Bone" : cv.COLORMAP_BONE,
    "Winter" : cv.COLORMAP_WINTER,
    "Jet" : cv.COLORMAP_JET,
    "Ocean" : cv.COLORMAP_OCEAN,
    "Magma" : cv.COLORMAP_MAGMA,
    "Plasma" : cv.COLORMAP_PLASMA,
    "Viridis" : cv.COLORMAP_VIRIDIS,
    "Hot" : cv.COLORMAP_HOT
}

def pointToHeatmap(
    pointList, gaussianSize=99, normalize=True, heatmapShape=(900, 900), offset=(0, 0)
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
    return g


def overlay_heatmap(image, heatmap, colormap, alpha=0.3):

    heatmap_colored = cv.applyColorMap(
        (heatmap * 255).astype(np.uint8), colormap
    )
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv.resize(heatmap_colored, (image.shape[1], image.shape[0]))

    result = cv.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return result


def load_gaze(path):

    gaze_data = load_jsonl_data(path)

    gaze_data = list(
        filter(lambda x: len(x["data"]) > 0 and "gaze2d" in x["data"], gaze_data)
    )

    return gaze_data 

def load_annotation(dataset_path: str):
    
    annot = pd.read_csv(
        os.path.join(dataset_path, "Annotation.csv"),
        dtype={"uint": "int", "start_action": "int", "stop_action": "int"},
    )

    annot.sort_values(by="uid", inplace=True)
    return annot


if __name__ == "__main__":

    dataset_path = os.path.expanduser(
        "~/Desktop/MasterThesis/data/datasets/Robofarmer-II"
    )
    annot = load_annotation(dataset_path)
    # unique_vids = annot["video_id"].unique().copy().tolist()
    idx = 0
    # cv.namedWindow("Gaze map", cv.WINDOW_NORMAL)
    while True:

        gaze_path = os.path.join(
            dataset_path, "videos", annot.iloc[idx]["video_id"], "gazedata"
        )
        # Laod gaze data
        gaze_data = load_jsonl_data(gaze_path)

        # Remove from gaze data all empty recording samples
        gaze_data = list(
            filter(lambda x: len(x["data"]) > 0 and "gaze2d" in x["data"], gaze_data)
        )

        # Gather all timestamps from the filtered gaze_data
        timestamps = [data["timestamp"] for data in gaze_data]

        start_action = annot.iloc[idx]["start_action"]
        stop_action = annot.iloc[idx]["stop_action"]
        frame_num = int(annot.iloc[idx]["inactive"])

        img_name = annot.iloc[idx]["inactive_frame_name"]

        inactive_path = os.path.join(
            dataset_path,
            annot.iloc[idx]["participant_id"],
            "rgb_frames",
            annot.iloc[idx]["video_id"],
            f"frame_{frame_num:010d}.jpg",
        )

        inactive_path2 = os.path.join(
            dataset_path,
            "inactive_images",
            annot.iloc[idx]["video_id"],
            img_name,
        )

        img = cv.imread(inactive_path)

        img2 = cv.imread(inactive_path2)

        interval = np.linspace(
            start_action, stop_action, stop_action - start_action, dtype=int
        )

        gaze_points = []
        for fr in interval:
            gaze_points.append(
                gaze_data[findTime(fr / 25, timestamps)]["data"]["gaze2d"]
            )

        h, w, _ = img.shape

        points = []
        for point in gaze_points:
            gaze_x, gaze_y = 0, 0
            gaze_x = int(point[0] * w)
            gaze_y = int(point[1] * h)

            points.append([gaze_x, gaze_y])
            # cv.circle(img, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
            #
        frame2 = img.copy()

        g = pointToHeatmap(points, heatmapShape=(h, w))
        frame = overlay_heatmap(img, g, COLORMAPS["Jet"])
        frame = cv.resize(
            frame, (int(w / 2.6), int(h / 2.4)), interpolation=cv.INTER_CUBIC
        )

        contact_points = annot.iloc[idx]["org_contact_points"]
        contact_str = contact_points.replace(" ", "")
        pairs = contact_str.replace("(", "").replace(")", "").split(",")
        contact_coords = []
        for i in range(0, len(pairs), 2):
            if i + 1 < len(pairs):
                contact_coords.append([int(pairs[i]), int(pairs[i + 1])])

        # print(contact_coords)

        # h2, w2, _ = img2.shape
        # frame2 = frame.copy()

        g2 = pointToHeatmap(contact_coords, heatmapShape=(h, w))
        frame2 = overlay_heatmap(frame2, g2, COLORMAPS["Jet"])
        frame2 = cv.resize(
            frame2, (int(w / 2.6), int(h / 2.4)), interpolation=cv.INTER_CUBIC
        )

        frames = np.concatenate([frame, frame2], axis=1)
        
        # Create third image combinig both heatmaps
        frame3 = overlay_heatmap(frame, g2, COLORMAPS["Jet"])
        h2, w2, c  = frame3.shape
        canvas = np.zeros((h2, w2*2, c), dtype=np.uint8)
        canvas[:, canvas.shape[1]//4 : (canvas.shape[1] // 4)* 3, : ] = frame3
        frames = np.concatenate([frames, canvas], axis=0)
        
        cv.imshow("Gaze map vs Annotation", frames)

        # Delay allowing to view the frames
        key = cv.waitKey(40) & 0xFF
        if key == ord("q"):
            exit()
        elif key == 83:
            idx = idx + 1
            print(
                os.path.join(
                    annot.iloc[idx]["video_id"],
                    f"frame_{frame_num:010d}.jpg",
                )
            )
        elif key == 81:
            idx = idx - 1
            print(
                os.path.join(
                    annot.iloc[idx]["video_id"],
                    f"frame_{frame_num:010d}.jpg",
                )
            )

    cv.destroyAllWindows()
