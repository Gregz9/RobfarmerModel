import os
import cv2 as cv
import cupy as cp
import numpy as np
import matplotlib.pyplot as pyplot
import torch
import open_clip
from open_clip import tokenizer


def show_mask(image, mask, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0, 0, 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    roi_img = None
    if borders:
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        cnt = max(contours, key=cv.contourArea)

        M = cv.moments(cnt)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        x, y, w, h = cv.boundingRect(cnt)

        roi_img = image[x - 100 : x + 100, y - 100 : y + 100].copy()

        # Try to smooth mask contours
        contours = [
            cv.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv.drawContours(
            image, contours, -1, (255, 0, 255, 0.5), thickness=2
        )

        # cv.rectangle(
        #     mask_image, (x - 100, y - 100), (x + w + 100, y + h + 100), (0, 255, 255), 1
        # )
        # cv.circle(mask_image, (center_x, center_y), 5, (255, 0, 0), -1)
    return mask_image, roi_img, (x, y, w, h), (center_x, center_y)


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    # Index of the highest score
    const_masks = [(i, mask) for i, mask in enumerate(masks) if mask.sum() <= 50000]
    mask_areas = [mask.sum() for mask in masks]
    const_scores = [scores[idx[0]] for idx in const_masks]
    b_idx = np.argmax(scores)
    # const_b_idx = const_scores.index(max(const_scores))
    mask, score = masks[b_idx], scores[b_idx]
    # mask, score = const_masks[const_b_idx][1], const_scores[const_b_idx]
    mask_image = show_mask(image, mask, borders=True)
    return mask_image


def show_mask_gpu(image, mask, random_color=False, borders=True):
    """GPU-accelerated version of show_mask function"""
    # Transfer data to GPU
    if isinstance(image, np.ndarray):
        image_gpu = cv.cuda_GpuMat()
        image_gpu.upload(image)

    if isinstance(mask, np.ndarray):
        mask_gpu = cv.cuda_GpuMat()
        mask_gpu.upload(mask.astype(np.uint8))

    if random_color:
        color = cp.concatenate([cp.random.random(3), cp.array([0.6])], axis=0)
    else:
        color = cp.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]

    # Reshape operations on GPU using CuPy
    mask_reshaped = cp.asarray(mask).reshape(h, w, 1)
    color_reshaped = color.reshape(1, 1, -1)
    print(f"Mask reshaped shape: {mask_reshaped.shape}")
    print(f"Color reshaped shape: {color_reshaped.shape}")
    print(color)
    mask_image = mask_reshaped * color_reshaped

    if borders:
        # Download mask for contour operations (OpenCV contour operations are CPU-only)
        mask_cpu = mask_gpu.download() if isinstance(mask_gpu, cv.cuda_GpuMat) else mask

        # Find contours on CPU (OpenCV doesn't support GPU contours)
        contours, _ = cv.findContours(
            mask_cpu.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )

        # Smooth contours
        contours = [
            cv.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]

        # Draw contours using GPU
        mask_image_gpu = cv.cuda_GpuMat()
        mask_image_gpu.upload(cp.asnumpy(mask_image))

        # Since drawContours isn't GPU-accelerated, we'll do this on CPU
        mask_image = cv.drawContours(
            cp.asnumpy(image.download()), contours, -1, (1, 1, 1), thickness=2
        )

        # Transfer back to GPU if needed for further processing
        mask_image = cp.asarray(mask_image, dtype=cp.float32)
        if mask_image.shape[-1] == 4:
            print("Wrong number of dimesions")
            print("\n", mask_image.shape)

    return cp.asnumpy(mask_image)  # Convert back to CPU for return


def show_masks_gpu(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    """GPU-accelerated version of show_masks function"""
    # Transfer scores to GPU for argmax
    scores_gpu = cp.asarray(scores)
    b_idx = cp.argmax(scores_gpu).get()

    mask, score = masks[b_idx], scores[b_idx]
    mask_image = show_mask_gpu(image, mask, borders=borders)

    return mask_image


def initialize_gpu():
    """Initialize GPU context and check for CUDA availability"""
    if not cv.cuda.getCudaEnabledDeviceCount():
        raise RuntimeError("No CUDA-capable device detected")

    # Initialize CUDA context
    cv.cuda.setDevice(0)
    stream = cv.cuda.Stream()

    return stream


def cleanup_gpu():
    """Clean up GPU resources"""
    cv.cuda.streamDestroy()
    cv.cuda.resetDevice()


# Example usage:
# if __name__ == "__main__":
#     try:
#         # Initialize GPU
#         stream = initialize_gpu()
#
#         # Your mask processing code here
#         # Example:
#         # image = cv.imread("image.jpg")
#         # masks = ... # Your masks array
#         # scores = ... # Your scores array
#         # result = show_masks_gpu(image, masks, scores)
#
#     finally:
#         # Clean up GPU resources
#         cleanup_gpu()
