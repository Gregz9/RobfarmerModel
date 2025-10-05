import os
import cv2 as cv
from tqdm import tqdm
import numpy as np

from utilities import (
    findTime,
    load_jsonl_data,
    DATASET_BASE_PATH,
)


def pointToHeatmap(
    pointList, gaussianSize=99, normalize=True, heatmapShape=(900, 900), offset=(0, 0)
):
    canvas = np.zeros(heatmapShape)
    for p in pointList:
        if p[1] < heatmapShape[0] and p[0] < heatmapShape[1]:
            canvas[p[1]][p[0]] = 255
    g = cv.GaussianBlur(canvas, (gaussianSize, gaussianSize), 0, 0)
    if normalize:
        g = cv.normalize(
            g, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
    return g


def overlay_heatmap(image, heatmap, alpha=0.3):

    heatmap_colored = cv.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv.COLORMAP_JET
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


if __name__ == "__main__":

    # Find the list of all new video files
    new_vids = os.listdir(
        os.path.expanduser("~/Desktop/MasterThesis/data/datasets/Robofarmer-II/videos")
    )

    # Filter out all files from the list, keep only directories with videos
    new_vids = list(
        filter(
            lambda x: not os.path.isfile(x),
            new_vids,
        )
    )

    # Print the list of all directories with videos
    vid_idx = 1
    for video in new_vids:
        print(f"{vid_idx}. {video}")
        vid_idx += 1

    # Choose video for which to project gaze
    choice = int(
        input("Enter video choice (video number preceeding video directory name): ")
    )

    # Full paths to the videos
    base_path = os.path.join(
        os.path.expanduser("~/"), "Desktop/MasterThesis/data/datasets/Robofarmer-II"
    )
    video_path = os.path.join(
        base_path, "videos", new_vids[choice - 1], "scenevideo.mp4"
    )
    gaze_path = os.path.join(base_path, "videos", new_vids[choice - 1], "gazedata")

    # Laod gaze data
    gaze_data = load_jsonl_data(gaze_path)

    # Remove from gaze data all empty recording samples
    gaze_data = list(
        filter(lambda x: len(x["data"]) > 0 and "gaze2d" in x["data"], gaze_data)
    )

    # Gather all timestamps from the filtered gaze_data
    timestamps = [data["timestamp"] for data in gaze_data]

    vid = cv.VideoCapture(video_path)

    # NOTE: NOT USED DUE TO PROBLEMS WHEN SAVING VIDOES
    # Get video properties for the output video
    # fps = vid.get(cv.CAP_PROP_FPS)
    # width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    # frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

    # Starting at 1, otherwise timeconst will be wrong
    idx = 1

    # progress_bar = tqdm(total=frame_count, desc="Processing frames")

    # Iterate over frames in video and project gaze
    while vid.isOpened():

        ret, frame = vid.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gaze_points = []
        if idx < 10:
            for i in range(idx):
                gaze_points.append(
                    gaze_data[findTime(i / 25, timestamps)]["data"]["gaze2d"]
                )
        else:
            for i in range(idx - 10, idx):
                gaze_points.append(
                    gaze_data[findTime(i / 25, timestamps)]["data"]["gaze2d"]
                )

        h, w, _ = frame.shape

        points = []
        for point in gaze_points:
            gaze_x, gaze_y = 0, 0
            gaze_x = int(point[0] * w)
            gaze_y = int(point[1] * h)
            gaze_x_half = int(point[0] * w // 2)
            gaze_y_half = int(point[1] * h // 2)

            points.append([gaze_x_half, gaze_y_half])

        # cv.circle(frame, (gaze_x, gaze_y), 7, (0, 0, 255), 5)
        # except:
        #     print(f"Could not draw gaze point for frame: {idx}")

        resized_frame = cv.resize(
            frame, (w // 2, h // 2), interpolation=cv.INTER_LANCZOS4
        )
        g = pointToHeatmap(points, heatmapShape=(h // 2, w // 2))
        frame = overlay_heatmap(resized_frame, g)
        cv.imshow(new_vids[choice - 1], frame)
        # cv.imshow("Gaze map", g)
        # progress_bar.update(1)

        idx += 1
        # Delay allowing to view the frames
        if cv.waitKey(40) & 0xFF == ord("q"):
            break
    vid.release()
    cv.destroyAllWindows()
