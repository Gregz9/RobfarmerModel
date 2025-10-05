import os
from re import A
import cv2 as cv
import json
from preprocess.dataset_util import FrameDetections
from preprocess.ho_types import BBox, HandDetection, HandSide, FloatVector, HandState
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import pprint
from tabulate import tabulate
from tqdm import tqdm

FLANN_INDEX_TREE = 1
DATASET_BASE_PATH = "../data/datasets/Robofarmer"
VIDEO_BASE_PATH = "../data/video"
VIDEO_FILE_NAME = "scenevideo.mp4"
PARTICIPANT_FILE = "meta/participant.json"


def inactive_image_structure(df: pd.DataFrame):

    inactive_images = {}

    # To keep track of unique values, since inactive bboxes can be reused multiple times
    seen_pairs = set()

    for col3_val, group in df.groupby(df.columns[2]):
        inactive_images[col3_val] = {}
        local_idx = 0

        for _, row in group.iterrows():
            col1_val = row.iloc[0]
            col2_val = row.iloc[1]
            col4_val = row.iloc[3]

            pair = (col1_val, col2_val)
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            if isinstance(col4_val, str):
                bbox_coords = col4_val.split(",")
            else:
                bbox_coords = col4_val

            inactive_images[col3_val][local_idx] = {
                "v_id": col1_val,
                "f_id": col2_val,
                "bbox": {
                    0: bbox_coords[0],
                    1: bbox_coords[1],
                    2: bbox_coords[2],
                    3: bbox_coords[3],
                },
            }
            local_idx += 1

    return inactive_images


def clips_structure(df, verbs, nouns):
    clips = []
    # Remove the outer loop since we want one entry per row
    local_idx = 0
    for idx, row in df.iterrows():
        clips.append(
            {
                "v_id": row.iloc[1],
                "start": row.iloc[2],
                "stop": row.iloc[3],
                "verb": [
                    idx for idx, name in verbs["verbs"].items() if name == row.iloc[4]
                ][0],
                "noun": [
                    idx for idx, name in nouns["nouns"].items() if name == row.iloc[5]
                ][0],
                "uid": row.iloc[0],
            }
        )
        local_idx += 1
    # print(json.dumps(clips, indent=4))
    return clips


def image_structure(df, verbs, nouns):

    images_data = []

    local_idx = 0
    for _, row in df.iterrows():

        if isinstance(row.iloc[4], str):
            bbox_coords = row.iloc[4].split(",")
            bbox_coords = [int(coord) for coord in bbox_coords]
        else:
            bbox_coords = row.iloc[4]

        images_data.append(
            {
                "image": {0: row.iloc[1].replace(" ", "")},
                "shape": {
                    0: bbox_coords[2] - bbox_coords[0],
                    1: bbox_coords[3] - bbox_coords[1],
                },
                "verb": row.iloc[2],
                "noun": row.iloc[3],
                "uid": row.iloc[0],
                "points": {},
            }
        )
        local_idx += 1
    # print(json.dumps(images_data, indent=4))
    return images_data


class DataHandler:
    def __init__(self, filepath, callback=None):
        self.filepath = filepath
        self.last_modified = os.path.getmtime(filepath)
        # self.data = self.read_data()
        self.read_data()
        self._toJson()
        self.callback = callback
        self.running = False
        self.thread = None

    def start_monitoring(self, interval=1):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()

    def stop_monitoring(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def read_data(self):
        df = pd.read_csv(
            self.filepath,
            dtype={
                "uid": "int",
                "start_movement": "int",
                "start_action": "int",
                "stop_action": "int",
            },
        )

        self.data = df.groupby("uid").filter(
            lambda x: x["participant_id"].notna().all()
            and x["participant_id"].ne("").all()
            and x["video_id"].notna().all()
            and x["video_id"].ne("").all()
            and x["bbox_coords"].notna().all()
            and x["bbox_coords"].ne("-, -, -, -").all()
        )

    def dumpJsonAnnotation(self):

        with open("../data/new_annotation.json", "w") as f:
            json.dump(self.annotation, f, indent=4)

    def _toJson(self):
        if self.data.empty:
            print("Cannot write data to a json file, dataframe is empty")
            return

        video_names = self.data["video_id"].unique()

        verbs_list = []
        nouns_list = []
        train_idxs = []
        test_idxs = []
        for v_name in video_names:
            data = self.data[self.data["video_id"] == v_name]
            num_rows = data.shape[0]
            uids = data["uid"].to_list()

            train_uids, temp_uids = train_test_split(
                uids,
                test_size=0.3,
                random_state=737,
                shuffle=False,
            )

            val_uids, test_uids = train_test_split(
                temp_uids,
                test_size=0.25,
                random_state=737,
                shuffle=False,
            )
            test_uids = val_uids
            train_idxs.extend(train_uids)
            test_idxs.extend(test_uids)

            actions = data["action"].unique()
            verbs_list.extend(actions)
            # print(actions)
            plants = data["plant"].unique()
            nouns_list.extend(plants)

        verbs_list = list(set(verbs_list))
        # verbs_list.remove(" ")
        verbs_inner: dict[int, str] = {}
        for i in range(len(verbs_list)):
            verbs_inner[i] = verbs_list[i]
        verbs = {"verbs": verbs_inner}

        nouns_list = list(set(nouns_list))
        nouns_inner = {}
        for j in range(len(nouns_list)):
            nouns_inner[j] = nouns_list[j]
        nouns = {"nouns": nouns_inner}

        inactive_noun = self.data[["video_id", "inactive", "plant", "bbox_coords"]]
        inactive_images = inactive_image_structure(inactive_noun)

        clips = self.data[
            ["uid", "video_id", "start_movement", "stop_action", "action", "plant"]
        ]

        train_df = clips[clips["uid"].isin(train_idxs)]
        test_df = clips[clips["uid"].isin(test_idxs)]

        train_clips = clips_structure(train_df, verbs, nouns)
        test_clips = clips_structure(test_df, verbs, nouns)

        image_data = self.data[
            ["uid", "inactive_frame_name", "action", "plant", "bbox_coords"]
        ]

        train_part = image_data[image_data["uid"].isin(train_idxs)]
        test_part = image_data[image_data["uid"].isin(test_idxs)]

        train_images = image_structure(train_part, verbs, nouns)
        test_images = image_structure(test_part, verbs, nouns)

        self.annotation = {
            "verbs": verbs_list,
            "nouns": nouns_list,
            "inactive_images": inactive_images,
            "train_clips": train_clips,
            "test_clips": test_clips,
            "train_images": train_images,
            "test_images": test_images,
        }

        # print(json.dumps(self.annotation, indent=4))

    def _monitor_loop(self, interval):
        while self.running:
            current_modified = os.path.getmtime(self.filepath)
            if current_modified > self.last_modified:
                self.read_data()
                self.last_modified = current_modified
                if self.callback:
                    self.callback(self.data)
            time.sleep(interval)


def read_participant_id(video_id):

    participant_file = open(os.path.join(VIDEO_BASE_PATH, video_id, PARTICIPANT_FILE))
    return json.load(participant_file)["participant"]


def load_cropped_inactive_frame(df: pd.DataFrame, uid: int):
    frame_name = f"{df.iloc[uid]["inactive_frame_name"]}.jpg"

    video_id = df.iloc[uid]["video_id"]
    participant_id = df.iloc[uid]["participant_id"]

    frame_path = os.path.join(
        DATASET_BASE_PATH, "inactive_images", video_id, frame_name
    )

    return cv.imread(frame_path), frame_name


def load_detections(video_path):
    with open(video_path, "rb") as f:
        video_detections = [
            FrameDetections.from_protobuf_str(s) for s in pickle.load(f)
        ]
    f.close()
    return video_detections


def euclidean_dist(a, b):
    return np.linalg.norm(np.sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)))


def draw_pose(image: np.ndarray, landmarks: dict):

    finger_colors = {
        "thumb": (255, 0, 0),  # Blue
        "index": (0, 255, 0),  # Green
        "middle": (0, 0, 255),  # Red
        "ring": (255, 255, 0),  # Cyan
        "pinky": (255, 0, 255),  # Magenta
    }

    h, w, _ = image.shape
    for idx in range(len(landmarks.keys())):

        landmark = landmarks[idx]
        x_coord = int(landmark[0] * w)
        y_coord = int(landmark[1] * h)

        # Determine finger and color
        if idx in range(1, 5):  # Thumb
            color = finger_colors["thumb"]
        elif idx in range(5, 9):  # Index
            color = finger_colors["index"]
        elif idx in range(9, 13):  # Middle
            color = finger_colors["middle"]
        elif idx in range(13, 17):  # Ring
            color = finger_colors["ring"]
        elif idx in range(17, 21):  # Pinky
            color = finger_colors["pinky"]
        else:  # Wrist (landmark 0)
            color = (255, 255, 255)  # White

        cv.circle(image, (x_coord, y_coord), 5, color, -1)

    connections = [
        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        # Index finger
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        # Middle finger
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        # Ring finger
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        start_point = (
            int(landmarks[start_idx][0] * w),
            int(landmarks[start_idx][1] * h),
        )
        end_point = (
            int(landmarks[end_idx][0] * w),
            int(landmarks[end_idx][1] * h),
        )

        # Get color based on which finger the connection belongs to
        if end_idx in range(1, 5):
            color = finger_colors["thumb"]
        elif end_idx in range(5, 9):
            color = finger_colors["index"]
        elif end_idx in range(9, 13):
            color = finger_colors["middle"]
        elif end_idx in range(13, 17):
            color = finger_colors["ring"]
        else:
            color = finger_colors["pinky"]

        cv.line(image, start_point, end_point, color, 2)

        return image


def load_jsonl_data(file_path):
    """
    Load JSON Lines data from a file into a list of dictionaries.

    Args:
        file_path (str): Path to the file containing JSON Lines data

    Returns:
        list: List of dictionaries, where each dictionary is a parsed JSON object
    """
    data = []

    try:
        with open(file_path, "r") as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    try:
                        json_obj = json.loads(line.strip())
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    return data


# Alternative function if you have the data as a string instead of a file
def parse_jsonl_string(jsonl_string):
    """
    Parse a string containing JSON Lines data into a list of dictionaries.

    Args:
        jsonl_string (str): String containing JSON Lines data

    Returns:
        list: List of dictionaries, where each dictionary is a parsed JSON object
    """
    data = []

    for line in jsonl_string.split("\n"):
        if line.strip():  # Skip empty lines
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue

    return data


# Helper function to denormalize/renormalize bbox
def denormalize_bbox(bbox, img_shape):
    h, w = img_shape
    return [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]


def normalize_bbox(bbox, img_shape):
    h, w = img_shape
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]


def interpolatePoints(
    dets_interval: list[FrameDetections],
    bboxes: dict[int, list[float]],
    side: str,
    estimator="sift",
):
    frames = bboxes.keys()
    frames = list(frames)
    frames.sort()

    homography = None

    if estimator == "sift":
        homography = siftHomography
    elif estimator == "surf":
        homography = surfHomography

    # Handle interpolation for frames before the first bbox frame
    if frames[0] > dets_interval[0].frame_number:
        # NOTE: Estimation backwards
        diff = frames[0] - dets_interval[0].frame_number
        homography_stack = [np.eye(3)]
        for i in range(frames[0] - dets_interval[0].frame_number, 0, -1):
            f1_path = os.path.join(
                DATASET_BASE_PATH,
                read_participant_id(dets_interval[i].video_id),
                "rgb_frames",
                dets_interval[i].video_id,
                f"frame_{dets_interval[i].frame_number:010d}.jpg",
            )
            f2_path = os.path.join(
                DATASET_BASE_PATH,
                read_participant_id(dets_interval[i - 1].video_id),
                "rgb_frames",
                dets_interval[i - 1].video_id,
                f"frame_{dets_interval[i-1].frame_number:010d}.jpg",
            )
            img1 = cv.imread(f1_path, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(f2_path, cv.IMREAD_GRAYSCALE)
            H, _ = homography(img1, img2)
            homography_stack.append(np.dot(homography_stack[-1], H))
            curr_bbox = denormalize_bbox(bboxes[frames[0]], img1.shape)
            new_bbox = interpolate_bbox(H, curr_bbox)
            new_bbox = normalize_bbox(new_bbox, img2.shape)
            new_bbox = BBox(new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3])
            origin_hand = [
                hand
                for hand in dets_interval[i].hands
                if side.upper() in hand.side.name
            ][0]
            new_hand = HandDetection(
                new_bbox,
                origin_hand.score,
                origin_hand.state,
                origin_hand.side,
                origin_hand.object_offset,
            )
            if len(dets_interval[i - 1].hands) < 2:
                dets_interval[i - 1].hands.append(new_hand)

    # Handle interpolation between bbox frames
    for b in tqdm(range(len(frames) - 1)):
        if frames[b + 1] - frames[b] < 2:
            continue  # Skip if frames are consecutive

        offset = frames[b + 1] - frames[b]
        start_idx = frames[b] - dets_interval[0].frame_number
        stop_idx = start_idx + offset

        homography_stack = [np.eye(3)]
        for i in range(start_idx, stop_idx - 1):
            f1_path = os.path.join(
                DATASET_BASE_PATH,
                read_participant_id(dets_interval[i].video_id),
                "rgb_frames",
                dets_interval[i].video_id,
                f"frame_{dets_interval[i].frame_number:010d}.jpg",
            )
            f2_path = os.path.join(
                DATASET_BASE_PATH,
                read_participant_id(dets_interval[i + 1].video_id),
                "rgb_frames",
                dets_interval[i + 1].video_id,
                f"frame_{dets_interval[i+1].frame_number:010d}.jpg",
            )
            img1 = cv.imread(f1_path, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(f2_path, cv.IMREAD_GRAYSCALE)
            H, _ = homography(img1, img2)

            homography_stack.append(np.dot(homography_stack[-1], H))

            curr_bbox = denormalize_bbox(bboxes[frames[b]], img1.shape)
            new_bbox = interpolate_bbox(H, curr_bbox)
            new_bbox = normalize_bbox(new_bbox, img2.shape)
            new_bbox = BBox(new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3])

            origin_hand = [
                hand
                for hand in dets_interval[i].hands
                if side.upper() in hand.side.name
            ][0]
            new_hand = HandDetection(
                new_bbox,
                origin_hand.score,
                origin_hand.state,
                origin_hand.side,
                origin_hand.object_offset,
            )

            if len(dets_interval[i + 1].hands) < 2:
                dets_interval[i + 1].hands.append(new_hand)

    # Handle interpolation for frames after the last bbox frame
    if frames[-1] < dets_interval[-1].frame_number:
        start_idx = frames[-1] - dets_interval[0].frame_number
        stop_idx = len(dets_interval) - 1  # Go to the end of dets_interval

        homography_stack = [np.eye(3)]
        for i in range(start_idx, stop_idx):
            f1_path = os.path.join(
                DATASET_BASE_PATH,
                read_participant_id(dets_interval[i].video_id),
                "rgb_frames",
                dets_interval[i].video_id,
                f"frame_{dets_interval[i].frame_number:010d}.jpg",
            )
            f2_path = os.path.join(
                DATASET_BASE_PATH,
                read_participant_id(dets_interval[i + 1].video_id),
                "rgb_frames",
                dets_interval[i + 1].video_id,
                f"frame_{dets_interval[i+1].frame_number:010d}.jpg",
            )
            img1 = cv.imread(f1_path, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(f2_path, cv.IMREAD_GRAYSCALE)
            H, _ = homography(img1, img2)
            homography_stack.append(np.dot(homography_stack[-1], H))

            curr_bbox = denormalize_bbox(bboxes[frames[-1]], img1.shape)
            new_bbox = interpolate_bbox(H, curr_bbox)
            new_bbox = normalize_bbox(new_bbox, img2.shape)
            new_bbox = BBox(new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3])

            origin_hand = [
                hand
                for hand in dets_interval[i].hands
                if side.upper() in hand.side.name
            ][0]
            new_hand = HandDetection(
                new_bbox,
                origin_hand.score,
                origin_hand.state,
                origin_hand.side,
                origin_hand.object_offset,
            )

            if len(dets_interval[i + 1].hands) < 2:
                dets_interval[i + 1].hands.append(new_hand)

    return dets_interval


def matchIdx2Frame(dets_list: list[FrameDetections], frame_bbox_dict: int):
    idxs = [det.frame_number for det in dets_list]

    return idxs.index(frame_bbox_dict)


def interpolate_bbox(H: np.ndarray, bbox: list[float]) -> list[float]:

    point_top = np.array((bbox[0], bbox[1], 1), dtype=np.float32)
    point_bottom = np.array((bbox[2], bbox[3], 1), dtype=np.float32)

    new_top = np.dot(H, point_top)
    new_bottom = np.dot(H, point_bottom)

    return [
        float(new_top[0] / new_top[2]),
        float(new_top[1] / new_top[2]),
        float(new_bottom[0] / new_bottom[2]),
        float(new_bottom[1] / new_bottom[2]),
    ]


def findIntervalBboxes(dets_interval: list[FrameDetections], side="right"):

    # hand_bbox = None
    # Represents lack of frame dtection
    # frame_number = -1
    bbox_frames = {}
    for det in dets_interval:
        for hand in det.hands:
            if side.upper() in hand.side.name:
                hand_bbox = [
                    hand.bbox.left,
                    hand.bbox.top,
                    hand.bbox.right,
                    hand.bbox.bottom,
                ]
                bbox_frames[det.frame_number] = hand_bbox

    if not bbox_frames:
        print(f"Could not find any bounding box detection for {side} hand")

    return bbox_frames


def siftHomography(img1, img2, lowe_ratio=0.3):

    H, mask = None, None
    sift = cv.SIFT_create()
    index_params = dict(algorithm=FLANN_INDEX_TREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Compute the
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Find matches between keypoints from the first and second image
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter for the best descriptors on the relative distance of
    # descriptors with regard to each other.
    good_matches = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise Exception("Cannot estimate homography, too few good matches")

    query_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    train_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        # Estimate homography
        H, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        if H is None or mask is None:
            raise Exception("Could not estimate homography between images")
    except:
        raise Exception("Could not estimate homography between images")

    return H, mask


def surfHomography(img1, img2, lowe_ratio=0.7):

    H, mask = None, None
    surf = cv.xfeatures2d.SURF_create(400)
    matcher = cv.DescriptorMatcher_create("BruteForce")

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise Exception("Cannot estiamte homography, too few good matches")

    query_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    train_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        H, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        if H is None or mask is None:
            raise Exception("Failed to estimate homography")
    except:
        raise Exception("Failed to estimate homography")

    return H, mask


def correct_sides(video_detections):

    # action_interval = []
    print("Post interpolation correction of sides")
    for i in range(len(video_detections)):
        if len(video_detections[i].hands) == 2:
            hands = [hand for hand in video_detections[i].hands]
            hands[0].side = (
                HandSide(1)
                if (hands[0].bbox.right + hands[0].bbox.left) / 2
                > (hands[1].bbox.left + hands[1].bbox.right) / 2
                else HandSide(0)
            )
            hands[1].side = HandSide(1) if hands[0].side != HandSide(1) else HandSide(0)
            video_detections[i].hands = hands
        # action_interval.append(video_detections[i])

    return video_detections


def findTime(curr_time: float, times: list[float]) -> int:

    return int(np.argmin(np.array([abs(curr_time - time) for time in times])))


def findGazePoint(frame_number, gaze_data):

    t_const = 1 / 25
    current_time = frame_number // 2 * t_const
    time_idx = 0
    times = []
    coords = []
    if frame_number < 50:
        times = [
            gaze_data[j]["timestamp"]
            for j in range(frame_number, frame_number + 100)
            if len(gaze_data[j]["data"]) > 0 and "gaze2d" in gaze_data[j]["data"]
        ]
        coords = [
            gaze_data[j]["data"]["gaze2d"]
            for j in range(frame_number, frame_number + 100)
            if len(gaze_data[j]["data"]) > 0 and "gaze2d" in gaze_data[j]["data"]
        ]
        time_idx = findTime(current_time, times)

    else:
        times = [
            gaze_data[j]["timestamp"]
            for j in range(frame_number - 50, frame_number + 50)
            if len(gaze_data[j]["data"]) > 0 and "gaze2d" in gaze_data[j]["data"]
        ]
        coords = [
            gaze_data[j]["data"]["gaze2d"]
            for j in range(frame_number - 50, frame_number + 50)
            if len(gaze_data[j]["data"]) > 0 and "gaze2d" in gaze_data[j]["data"]
        ]
        time_idx = findTime(current_time, times)

    gaze_x = coords[time_idx][0]
    gaze_y = coords[time_idx][1]

    return gaze_x, gaze_y


def generate_gaze_heatmap(image, gaze_coord, sigma=30):
    """
    Generate a Gaussian heatmap based on gaze coordinates for a single image.

    Args:
        image (numpy.ndarray): Input image array with shape (H, W, C)
        gaze_coord (tuple): Gaze coordinates (x, y) normalized between 0 and 1
        sigma (int): Standard deviation of Gaussian kernel (controls spread)

    Returns:
        tuple: (heatmap, visualization) where both are uint8 numpy arrays
    """
    # Get image dimensions
    h, w = image.shape[:2]

    # Make a copy of the original image to avoid modifying it
    original = image.copy()

    # Convert normalized coordinates to pixel coordinates
    x, y = int(gaze_coord[0] * w), int(gaze_coord[1] * h)

    # Create empty heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Draw Gaussian at gaze point
    cv.circle(heatmap, (x, y), 0, 1.0, -1)
    heatmap = cv.GaussianBlur(heatmap, (0, 0), sigma)

    # Normalize heatmap to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Convert heatmap to uint8 (0-255)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Create colored heatmap for visualization (red-yellow heat)
    heatmap_colored = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_HOT)

    # Make sure we're operating on uint8 images
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)

    # Create visualization with heatmap overlay
    alpha = 0.5  # Lower alpha for more subtle overlay
    visualization = cv.addWeighted(original, 1.0, heatmap_colored, alpha, 0)

    # Convert single-channel heatmap to 3-channel for CUDA compatibility
    heatmap_3ch = cv.merge([heatmap_uint8, heatmap_uint8, heatmap_uint8])

    return heatmap_3ch, visualization
