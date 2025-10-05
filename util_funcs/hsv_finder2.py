import cv2
import mediapipe as mp
import os
import json
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

window_title = "Recognition result"
control_window = "Controls"  # Separate window for controls


class HSVHelper:
    def __init__(self):
        self.mp_hands = mp_hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
        )
        self.img_bgr = None
        self.original_size = None

    def apply_hsv_and_contrast(self, image, h_offset, s_scale, v_scale, contrast):
        # Apply contrast adjustment first
        contrast_image = cv2.addWeighted(image, 1.0 + (contrast / 100.0), image, 0, 0)

        # Then apply HSV adjustments
        img_hsv = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2HSV)

        h = cv2.add(img_hsv[:, :, 0], h_offset)
        s = cv2.multiply(img_hsv[:, :, 1], s_scale)
        v = cv2.multiply(img_hsv[:, :, 2], v_scale)

        img_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def on_trackbar_change(self, *args):
        # Get current trackbar values
        h_offset = cv2.getTrackbarPos("H", control_window)
        s_scale = cv2.getTrackbarPos("S", control_window) / 100.0
        v_scale = cv2.getTrackbarPos("V", control_window) / 100.0
        contrast = cv2.getTrackbarPos("C", control_window) - 100  # Center at 0

        # Process image with current adjustments
        img_bgr_modified = self.recognize(
            self.img_bgr.copy(), h_offset, s_scale, v_scale, contrast
        )

        # Resize to half size
        display_size = (self.original_size[1] // 2, self.original_size[0] // 2)
        img_display = cv2.resize(img_bgr_modified, display_size)

        cv2.imshow(window_title, img_display)

    def recognize(self, img_bgr, h_offset, s_scale, v_scale, contrast):
        # Apply adjustments
        img_bgr = self.apply_hsv_and_contrast(
            img_bgr, h_offset, s_scale, v_scale, contrast
        )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        else:
            print("no hands were found")
        return img_bgr

    def run(self, img_path):
        self.img_bgr = cv2.imread(img_path)
        if self.img_bgr is None:
            print("Image was not found!")
            return

        self.original_size = self.img_bgr.shape[:2]

        # Create separate windows
        cv2.namedWindow(window_title)
        cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)

        # Set control window size and position
        cv2.resizeWindow(control_window, 400, 100)

        # Create compact trackbars with short names
        cv2.createTrackbar("H", control_window, 0, 179, self.on_trackbar_change)  # Hue
        cv2.createTrackbar(
            "S", control_window, 100, 200, self.on_trackbar_change
        )  # Saturation
        cv2.createTrackbar(
            "V", control_window, 100, 200, self.on_trackbar_change
        )  # Value
        cv2.createTrackbar(
            "C", control_window, 100, 200, self.on_trackbar_change
        )  # Contrast

        # Set initial values
        self.on_trackbar_change(0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TFLITE_LOG_SILENT"] = "4"
    h = HSVHelper()
    video_base_path = "../data/video"
    robo_path = "../data/datasets/Robofarmer"

    print("Enter the following: ")
    video_id = str(input("Enter video ID: "))
    frame_number = int(input("Enter frame number for inspection"))

    participant_file = open(
        os.path.join(video_base_path, video_id, "meta/participant.json")
    )
    participant_id = json.load(participant_file)["participant"]

    h.run(
        os.path.join(
            robo_path,
            participant_id,
            "rgb_frames",
            video_id,
            f"frame_{frame_number:010d}.jpg",
        )
    )
