import cv2 as cv
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import subprocess
from utilities import load_cropped_inactive_frame
from utilities import read_participant_id
from typing import Optional
import math

_BASE_PATH = os.path.expanduser("~/Desktop/MasterThesis/data/datasets")

# NOTE: The old dataset does not have paths configured optimally
_DATA_ROOTS = {
    "dataset": os.path.join(_BASE_PATH, "Robofarmer"),
    "annotation": os.path.expanduser("~/Desktop/MasterThesis/data/Annotation.csv"),
    "videos": os.path.expanduser("~/Desktop/MasterThesis/data/video"),
}

_DATA_ROOTS_II = {
    "dataset": os.path.join(_BASE_PATH, "Robofarmer-II"),
    "annotation": os.path.join(_BASE_PATH, "Robofarmer-II/Annotation.csv"),
    "videos": os.path.join(_BASE_PATH, "Robofarmer-II/videos"),
    "images": os.path.join(_BASE_PATH, "Robofarmer-II/image_path_map2.json"),
    "inactive_images": os.path.join(_BASE_PATH, "Robofarmer-II/inactive_images"),
    "video_participants": os.path.join(
        _BASE_PATH, "Robofarmer-II/video_participants_acro.json"
    ),
    "participants": os.path.join(_BASE_PATH, "Robofarmer-II/participants.json"),
}


class CombinedAnnotator:
    def __init__(self):

        # self.save_annot_thread = None
        # self.save_img_thread = None
        # Upon init, write the menu and allow the user to input arguments interactively
        print("Choose which dataset to use: ")
        print("1. Robofarmer")
        print("2. Robofarmer-II")

        # Take input from user at the start
        choice = int(input("Enter choice (1 or 2): "))
        self.dataset_path = (
            _DATA_ROOTS["dataset"] if choice == 1 else _DATA_ROOTS_II["dataset"]
        )
        self.data_paths = _DATA_ROOTS if choice == 1 else _DATA_ROOTS_II

        self.data = self.load_annotation()

        self.video_participant_map = {}
        f = open(os.path.join(self.dataset_path, "video_participants_acro.json"), "r")
        self.video_participant_map = json.load(f)
        f.close()

        image_paths_map = {}
        f = open(os.path.join(self.dataset_path, "image_path_map2.json"), "r")
        # NOTE: Should be a member variable?
        self.image_paths_map = json.load(f)
        f.close()

        # self.image_paths = image_list_path
        self.dataset_name = self.dataset_path.split("/")[-1]
        self.video_dirs = list(self.image_paths_map.keys())

        print("Choose video to annotate: ")
        for i in range(len(self.video_dirs)):
            print(f"{i+1}. {self.video_dirs[i]}")

        choice2 = int(input("Enter choice (or -1 for all videos): "))
        self.video_idx = choice2 - 1
        print(self.video_dirs[self.video_idx])

        # Choosing only one directory
        self.video_idx = choice2 - 1
        self.image_list = sorted(self.image_paths_map[self.video_dirs[self.video_idx]])
        # Grab paths to all images across the dataset
        # for k, v in self.image_paths_map.items():
        # NOTE: List must be re-sorted after reading
        # self.image_list.extend(sorted(v))

        self.img = None
        self.inactive_frame = None
        self.video_name = None
        self.display_img = None
        self.right_img = None
        self.display_right_img = None

        self.last_annot_row = 0

        self.start_action = 0
        self.stop_action = 0

        # Bounding box drawing state
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1

        # Bbox coordinate copy
        self.ix_copy = self.iy_copy = self.fx_copy = self.fy_copy = -1

        # Point annotation state
        self.display_points = []
        self.annot_points = []
        self.bbox_points = []
        self.max_points = 10

        # Navigation state
        self.current_frame = 0
        self.scale_factor = 0.5

        # NOTE: Read image to extract dimensions
        self.img_height, self.img_width, _ = cv.imread(self.image_list[0]).shape
        # print(self.img_height, "x", self.img_width)

        if not os.path.exists(self.data_paths["dataset"]):
            self.createDatasetDirs(self.dataset_name)

        # NOTE: Fill image used in display
        self.blank_img = np.zeros(
            (
                int(self.img_height * self.scale_factor),
                int(self.img_width * self.scale_factor),
                3,
            ),
            dtype=np.uint8,
        )
        self.diplay_right_img = self.blank_img

    def load_annotation(self):

        data = None
        if self.checkIfAnnotExists():
            data = self.read_annotation()
        else:
            data = self.createAnnotationTemplate()

        return data

    def checkIfAnnotExists(self):
        return os.path.exists(self.data_paths["annotation"])

    def find_current_row(self):

        replace = False
        row_idx = 0
        video_name = self.video_dirs[self.video_idx]

        if len(self.data) == 0:
            row_idx = 0
            return 0, replace

        # NOTE: Find all fully and partially overlapping lines
        mask = (
            (self.data["video_id"] == video_name)
            & (self.data["stop_action"].astype(int) >= self.start_action)
            & (self.stop_action > self.data["start_action"].astype(int))
        )

        query_result = self.data[mask]
        print(query_result)
        if len(query_result) > 0:
            # NOTE: Remove overlapping rows
            row_idx = int(query_result.iloc[0]["uid"])
            replace = True
            self.data = self.data[~mask].copy()
            return row_idx, replace

        # NOTE: if no overlapping rows, find the closest one
        if len(query_result) == 0:

            video_rows = self.data[self.data.video_id == video_name].sort_values(
                "start_action"
            )

            if len(video_rows) == 0:
                row_idx = len(self.data)

            else:
                insertion_idx = None
                for idx, row in video_rows.iterrows():
                    if self.start_action < row["start_action"]:
                        insertion_idx = self.data.iloc[idx]["uid"]
                        break

                if insertion_idx is None:
                    row_idx = len(self.data)
                else:
                    row_idx = insertion_idx

        return row_idx, replace

    def save_annotation_row(self, annot_row, uid, replace):

        new_row = pd.DataFrame([annot_row])
        if not replace and uid < len(self.data) - 1:
            first_part = self.data.iloc[:uid].copy()
            second_part = self.data.iloc[uid:].copy()
            second_part["uid"] += 1

            self.data = pd.concat([first_part, new_row, second_part]).reset_index(
                drop=True
            )
        elif replace:
            self.data.loc[uid] = new_row.iloc[0]
            print("Replacing")
        else:
            self.data = pd.concat([self.data, new_row])

        self.data.to_csv(self.data_paths["annotation"], index=False)

    def createAnnotationTemplate(self):

        annot_headers = [
            "uid",
            "participant_id",
            "video_id",
            "start_timestamp",
            "stop_timestamp",
            "start_action",
            "stop_action",
            "inactive",
            "inactive_frame_name",
            "action",
            "plant",
            "bbox_coords",
            "contact_points",
            "org_contact_points",
        ]

        annot_df = pd.DataFrame(columns=annot_headers)
        # Create the empty csv file
        annot_df.to_csv(self.data_paths["annotation"], sep=",")

        return annot_df

    def read_annotation(self):

        df = pd.read_csv(
            _DATA_ROOTS_II["annotation"],
            dtype={
                "uid": "int",
                # "start_movement": "int",
                "start_action": "int",
                "stop_action": "int",
            },
        )
        data = df.sort_values(by="uid")
        return data

    def read_data(self):
        """Read data from CSV file (from original point.py)"""
        df = pd.read_csv(
            self.data_paths["annotation"],
            dtype={
                "uid": "int",
                # "start_movement": "int",
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

    def nextFrame(self):

        if self.current_frame == len(self.image_list) - 1:
            self.video_idx = (
                min(self.video_idx + 1, len(self.video_dirs) - 1)
                if self.video_idx < len(self.video_dirs) - 1
                else 0
            )
            self.current_frame = 0

        else:
            self.current_frame = min(self.current_frame + 1, len(self.image_list) - 1)

        # NOTE: Do not update unless video_idx is updated
        if self.video_dirs[self.video_idx] not in self.image_list[0]:
            self.image_list = sorted(
                self.image_paths_map[self.video_dirs[self.video_idx]]
            )

    def next10thFrame(self):
        if self.current_frame == len(self.image_list) - 1:
            self.video_idx = (
                min(self.video_idx + 1, len(self.video_dirs) - 1)
                if self.video_idx < len(self.video_dirs) - 1
                else 0
            )
            self.current_frame = 0

        else:
            self.current_frame = min(self.current_frame + 10, len(self.image_list) - 1)

        # NOTE: Do not update unless video_idx is updated
        if self.video_dirs[self.video_idx] not in self.image_list[0]:
            self.image_list = sorted(
                self.image_paths_map[self.video_dirs[self.video_idx]]
            )

    def previousFrame(self):
        # self.current_frame = max(self.current_frame - 1, 0)

        if self.current_frame == 0:
            self.video_idx = (
                max(self.video_idx - 1, 0)
                if self.video_idx > 0
                else len(self.video_dirs) - 1
            )
            self.current_frame = (
                len(self.image_paths_map[self.video_dirs[self.video_idx]]) - 1
            )
        else:
            self.current_frame = max(self.current_frame - 1, 0)

        # NOTE: Same here, unless update to video_idx, do not change the image list
        if self.video_dirs[self.video_idx] not in self.image_list[0]:
            self.image_list = sorted(
                self.image_paths_map[self.video_dirs[self.video_idx]]
            )

    def previous10thFrame(self):
        if self.current_frame == 0:
            self.video_idx = (
                max(self.video_idx - 1, 0)
                if self.video_idx > 0
                else len(self.video_dirs) - 1
            )
            self.current_frame = (
                len(self.image_paths_map[self.video_dirs[self.video_idx]]) - 10
            )
        else:
            self.current_frame = max(self.current_frame - 10, 0)

        # NOTE: Same here, unless update to video_idx, do not change the image list
        if self.video_dirs[self.video_idx] not in self.image_list[0]:
            self.image_list = sorted(
                self.image_paths_map[self.video_dirs[self.video_idx]]
            )

    def nextVideo(self):

        if self.video_idx == len(self.video_dirs) - 1:
            self.video_idx = 0
        else:
            self.video_idx += 1

        self.image_list = sorted(self.image_paths_map[self.video_dirs[self.video_idx]])

        self.current_frame = 0

    def previousVideo(self):

        if self.video_idx == 0:
            self.video_idx = len(self.video_dirs) - 1
        else:
            self.video_idx -= 1

        self.image_list = sorted(self.image_paths_map[self.video_dirs[self.video_idx]])

        self.current_frame = 0

    # NOTE: One window can not have two dedicated callback functions. A combined function must be created
    def mouse_callback_left(self, event, x, y, flags, param):
        """Mouse callback for left pane (bounding box drawing)"""

        if x < self.display_img.shape[1] and y < self.display_img.shape[0]:
            if event == cv.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x, y
                self.fx, self.fy = x, y

                self.inactive_frame = self.current_frame

            elif event == cv.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.fx, self.fy = x, y

            elif event == cv.EVENT_LBUTTONUP:
                self.drawing = False
                self.fx, self.fy = x, y
                self.ix_copy = self.ix
                self.iy_copy = self.iy
                self.fx_copy = self.fx
                self.fy_copy = self.fy
                # When bbox is completed, display the image in right pane
                self.update_right_pane(x1=self.ix, y1=self.iy, x2=self.fx, y2=self.fy)

        elif (
            x >= self.display_img.shape[1]
            and x <= self.display_img.shape[1] * 2
            and y < self.display_img.shape[0]
        ):
            if event == cv.EVENT_LBUTTONDOWN:
                if len(self.display_points) < self.max_points:
                    # NOTE: The point must be offset
                    self.display_points.append((x - self.display_img.shape[1], y))

                    # Scaling factor for the bbox
                    bbox_width = self.fx_copy - self.ix_copy
                    bbox_height = self.fy_copy - self.iy_copy

                    relative_x = (
                        x - self.display_img.shape[1]
                    ) / self.display_img.shape[1]

                    relative_y = y / self.display_right_img.shape[0]

                    scaled_x = (relative_x * bbox_width) + self.ix_copy
                    scaled_y = self.iy_copy + (relative_y * bbox_height)

                    self.annot_points.append((scaled_x, scaled_y))

                    # NOTE: Saving coordinates scaled to the bounding box
                    scaled2_x = relative_x * bbox_width
                    scaled2_y = relative_y * bbox_height

                    self.bbox_points.append((scaled2_x, scaled2_y))

                    print(f"Point {len(self.annot_points)}: ({scaled_x}, {scaled_y})")

                    if len(self.annot_points) == self.max_points:
                        formatted_points = ", ".join(
                            [
                                f"({int(point[0])}, {int(point[1])})"
                                for point in self.annot_points
                            ]
                        )
                        print(f"All points marked: {formatted_points}")

        """Mouse callback for right pane (point annotation)"""

    def update_right_pane(self, **kwargs):
        """Update right pane with the current image when bbox is drawn"""
        # self.img is assigned below in "run loop"
        if self.img is not None:

            # Handling of drawing from right to left
            x1, x2, y1, y2 = kwargs["x1"], kwargs["x2"], kwargs["y1"], kwargs["y2"]
            if kwargs["y1"] > kwargs["y2"]:
                y1 = kwargs["y2"]
                y2 = kwargs["y1"]

            if kwargs["x1"] > kwargs["x2"]:
                x1 = kwargs["x2"]
                x2 = kwargs["x1"]

            self.bbox_coords = [x1, y1, x2, y2]

            # NOTE: When saving annotation, the order of points must be inverted, i.e. x becomes y and vice versa
            self.right_img = self.extract_img[y1:y2, x1:x2, :].copy()
            # self.inactive_frame = self.extract_img[y1:y2, x1:x2, :].copy()
            # Reset points when new image is loaded in right pane
            self.display_points = []
            self.annot_points = []
            self.bbix_points = []

    def create_info_panel(self):
        """Create info panel with controls and status"""
        panel = np.zeros((700, 350, 3), dtype=np.uint8)

        # Title
        cv.putText(
            panel,
            "Combined Annotator",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Current frame info
        cv.putText(
            panel,
            f"Image: {self.current_frame + 1}/{len(self.image_list)}",
            (10, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Bounding box info
        orig_ix = int(self.ix / self.scale_factor) if self.ix != -1 else -1
        orig_iy = int(self.iy / self.scale_factor) if self.iy != -1 else -1
        orig_fx = int(self.fx / self.scale_factor) if self.fx != -1 else -1
        orig_fy = int(self.fy / self.scale_factor) if self.fy != -1 else -1

        cv.putText(
            panel,
            "Bounding Box:",
            (10, 110),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv.putText(
            panel,
            f"({orig_ix}, {orig_iy}) to ({orig_fx}, {orig_fy})",
            (10, 140),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Points info
        cv.putText(
            panel,
            f"Points: {len(self.display_points)}/{self.max_points}",
            (10, 180),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        for i, point in enumerate(self.display_points):
            orig_x = int(point[0])  # / self.scale_factor)
            orig_y = int(point[1])  # / self.scale_factor)
            cv.putText(
                panel,
                f"  {i+1}: ({orig_x}, {orig_y})",
                (10, 210 + i * 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Controls
        y_offset = 320
        cv.putText(
            panel,
            "Controls:",
            (10, y_offset),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv.putText(
            panel,
            "Left: Draw bbox",
            (10, y_offset + 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "Right: Click points",
            (10, y_offset + 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "n/p: Next/Prev",
            (10, y_offset + 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "c: Clear",
            (10, y_offset + 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "r: reload images",
            (10, y_offset + 110),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "s: Save",
            (10, y_offset + 130),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "q: Quit",
            (10, y_offset + 150),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "v: Mark start of action",
            (10, y_offset + 170),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "b: Mark end of action",
            (10, y_offset + 190),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv.putText(
            panel,
            "g : Print formatted",
            (10, y_offset + 210),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return panel

    def write_annotation_row(self):
        """Save data collected in the dataframe from the annotation process"""

        uid, replace = self.find_current_row()
        start_action = self.start_action
        stop_action = self.stop_action
        participant_id = self.video_participant_map[self.video_dirs[self.video_idx]]
        video_id = self.video_dirs[self.video_idx]

        time_constant = 1 / 25.0  # 25 Hz is the sampling rate
        start_timestamp = start_action * time_constant
        stop_timestamp = stop_action * time_constant
        image_name = os.path.basename(self.image_list[self.current_frame])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_number = self.current_frame + 1
        plant = "strawberry"

        # min_x, min_y, max_x, max_y = 0, 0, 0, 0
        # if self.ix != -1 and self.iy != -1 and self.fx != -1 and self.fy != -1:

        # Convert coordinates back to original scale (1920x1080)
        orig_ix = int(self.bbox_coords[0] / self.scale_factor)
        orig_iy = int(self.bbox_coords[1] / self.scale_factor)
        orig_fx = int(self.bbox_coords[2] / self.scale_factor)
        orig_fy = int(self.bbox_coords[3] / self.scale_factor)

        # Ensure coordinates are in correct order
        min_x = min(orig_ix, orig_fx)
        max_x = max(orig_ix, orig_fx)
        min_y = min(orig_iy, orig_fy)
        max_y = max(orig_iy, orig_fy)

        bbox_coords = f"{min_x}, {min_y}, {max_x}, {max_y}"
        # Inactive frame name
        inactive_name = (
            str(participant_id)
            + "_"
            + str(self.inactive_frame + 1)
            + "_"
            + str(min_x)
            + "_"
            + str(min_y)
            + "_"
            + str(max_x)
            + "_"
            + str(max_y)
            + ".jpg"
        )

        self.inactive = self.right_img.copy()

        self.save_inactive_frame(inactive_name, video_id)

        # Add points if any
        org_points = []
        points = []
        print(self.annot_points)
        for i, point in enumerate(self.annot_points):
            org_points.append(
                f"{(int(point[0] / self.scale_factor), int(point[1] / self.scale_factor))}"
            )

            points.append(
                f"{(int(self.bbox_points[i][0]), int(self.bbox_points[i][1]))}"
            )

        points_str = ", ".join(points)
        org_points_str = ", ".join(org_points)

        action = str(input("Enter action performed by participant: "))

        df_dict = {
            "uid": uid,
            "participant_id": str(participant_id),
            "video_id": str(video_id),
            "start_timestamp": float(start_timestamp),
            "stop_timestamp": float(stop_timestamp),
            "start_action": int(start_action),
            "stop_action": int(stop_action),
            "inactive": str(self.inactive_frame),
            "inactive_frame_name": str(inactive_name),
            "action": str(action),
            "plant": str(plant),
            "bbox_coords": bbox_coords,
            "contact_points": str(points_str),
            "org_contact_points": str(org_points_str),
        }
        self.save_annotation_row(df_dict, uid, replace)

    def save_inactive_frame(self, inactive_name, video_id):

        dir_path = os.path.join(self.data_paths["inactive_images"], video_id)
        img_path = os.path.join(dir_path, inactive_name)

        if not os.path.exists(self.data_paths["inactive_images"]):
            try:
                subprocess.run(["mkdir", self.data_paths["inactive_images"]])
            except:
                print("Directory already exsists or and error occured")

        if not os.path.exists(dir_path):
            try:
                subprocess.run(["mkdir", dir_path])
            except:
                print("Directory already exsists or and error occured")

        try:
            cv.imwrite(img_path, self.inactive)
        except:
            print("Could not save inactive_image")

    def run(self, start_frame=0):
        """Main execution loop"""
        if not self.image_list:
            print("No images found! Check the image list or CSV file.")
            return

        self.current_frame = start_frame

        # Create windows
        cv.namedWindow("Annotation Tool", cv.WINDOW_NORMAL)
        cv.namedWindow("Info Panel", cv.WINDOW_NORMAL)

        # Set mouse callbacks
        cv.setMouseCallback("Annotation Tool", self.mouse_callback_left)

        # Position windows
        cv.moveWindow("Annotation Tool", 0, 0)
        cv.moveWindow("Info Panel", 1600, 0)

        while True:
            # Load current image
            if self.current_frame < len(self.image_list):
                self.img = cv.imread(self.image_list[self.current_frame])
                if self.img is None:
                    print(
                        f"Could not read image: {self.image_list[self.current_frame]}"
                    )
                    break

                # Create display image for left pane
                self.display_img = cv.resize(
                    self.img,
                    (
                        int(self.img.shape[1] * self.scale_factor),
                        int(self.img.shape[0] * self.scale_factor),
                    ),
                )

                self.extract_img = self.display_img.copy()

                # Draw bounding box if exists
                if self.ix != -1 and self.iy != -1 and self.fx != -1 and self.fy != -1:
                    cv.rectangle(
                        self.display_img,
                        (self.ix, self.iy),
                        (self.fx, self.fy),
                        (0, 255, 0),
                        2,
                    )

                # Add frame info
                cv.putText(
                    self.display_img,
                    f"Frame: {self.current_frame + 1}/{len(self.image_list)}",
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                comb_img = np.concatenate(
                    [self.display_img, self.diplay_right_img], axis=1
                )

                # Show left pane
                cv.imshow("Annotation Tool", comb_img)

                # Update right pane if image exists
                if self.right_img is not None:
                    if 0 in self.right_img.shape:
                        self.display_right_img = self.blank_img
                    else:
                        self.display_right_img = cv.resize(
                            self.right_img,
                            (
                                int(self.display_img.shape[1]),
                                int(self.display_img.shape[0]),
                            ),
                            interpolation=cv.INTER_CUBIC,
                        )

                    # Draw points
                    for point in self.display_points:
                        # offset_point = (point[0] + self.display_img.shape[1], point[1])
                        cv.circle(self.display_right_img, point, 3, (255, 0, 0), -1)

                    # Add points info
                    cv.putText(
                        self.display_right_img,
                        f"Points: {len(self.display_points)}/{self.max_points}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    comb_img = np.concatenate(
                        [self.display_img, self.display_right_img], axis=1
                    )
                    cv.imshow("Annotation Tool", comb_img)

                # TODO: Check whether a default value is needed for keeping the image open
                # else:
                #     self.display_right_img = self.display_right_img

                # Show info panel
                info_panel = self.create_info_panel()
                cv.imshow("Info Panel", info_panel)

            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                self.next10thFrame()
                self.ix = self.iy = self.fx = self.fy = -1
            elif key == 83:
                self.nextFrame()
                self.ix = self.iy = self.fx = self.fy = -1
            elif key == ord("k"):
                self.nextVideo()
            elif key == ord("j"):
                self.previousVideo()
            elif key == ord("p"):
                self.previous10thFrame()
                self.ix = self.iy = self.fx = self.fy = -1
            elif key == 81:
                self.previousFrame()
                self.ix = self.iy = self.fx = self.fy = -1
            elif key == ord("c"):
                self.ix = self.iy = self.fx = self.fy = -1
                self.ix_copy = self.iy_copy = self.fx_copy = self.fy_copy = -1
                self.display_points = []
                self.annot_points = []
                self.bbox_points = []
                self.right_img = None
                self.start_action = 0
                self.stop_action = 0
                print("Annotations cleared.")
            # TODO: Re-implement functionality for saving annotation
            elif key == ord("s"):
                self.write_annotation_row()
                self.ix = self.iy = self.fx = self.fy = -1
                self.ix_copy = self.iy_copy = self.fx_copy = self.fy_copy = -1
                self.display_points = []
                self.annot_points = []
                self.bbox_points = []
                self.right_img = None
                self.start_action = 0
                self.stop_action = 0
            elif key == ord("v"):
                self.start_action = self.current_frame
            elif key == ord("b"):
                self.stop_action = self.current_frame
            elif key == ord("t"):
                self.write_annotation_row()

        cv.destroyAllWindows()


if __name__ == "__main__":
    # NOTE: Ignore warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    annotator = CombinedAnnotator()

    if annotator.image_list:
        start_frame = int(input("Enter starting frame number (0-based): "))
        annotator.run(start_frame)
