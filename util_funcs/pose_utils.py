import cv2 as cv
import numpy as np
import mediapipe as mp


def transform_to_skin_tone(frame):
    """
    Transform white gloves to appear more like natural skin tone
    """
    # Convert to LAB color space (better for color manipulation)
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    # Adjust 'a' channel towards red/pink (typical for skin)
    a = cv.add(a, 10)  # Add slight red tint

    # Adjust 'b' channel towards yellow (typical for skin)
    b = cv.add(b, 15)  # Add yellow tint

    # Reduce brightness of very bright areas
    l = cv.multiply(l, 0.85)  # Darken bright whites

    # Merge channels and convert back
    lab = cv.merge([l, a, b])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def add_hue_offset(image, offset=93):
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Add offset to Hue channel
    # OpenCV uses H values from 0-179 (not 0-360)
    # So we need to scale our offset
    h_offset = int(offset * 179 / 360)

    # Add offset and handle wraparound
    hsv[:, :, 0] = (hsv[:, :, 0] + h_offset) % 180

    # Convert back to BGR
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
