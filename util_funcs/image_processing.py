from warnings import warn
import cv2 as cv
import numpy as np
import copy


def invert_colors(image: np.ndarray) -> np.ndarray:

    return cv.bitwise_not(image)


# Sharpening using unsharp masking
def image_sharpening(image: np.ndarray) -> np.ndarray:

    blur = cv.GaussianBlur(image, (7, 7), 0)
    sharp_features = cv.subtract(image, blur)

    sharp_img = cv.addWeighted(image, 2, blur, -1, 0)

    return sharp_img  # image + sharp_features


def bicubic_upsamling(image: np.ndarray, height: int, width: int) -> np.ndarray:

    gpu_image = cv.cuda.GpuMat()
    gpu_image.upload(image)

    gpu_upsampled_image = cv.cuda.resize(
        gpu_image, (width, height), interpolation=cv.INTER_LANCZOS4
    )

    return gpu_upsampled_image.download()


# Using EDSR
def super_resolution(image):
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    path = "models/EDSR_x4.pb"

    sr.readModel(path)
    sr.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    sr.setModel("edsr", 4)

    # upsample image 4x
    result_img = sr.upsample(image)

    return result_img


def super_resolution_tiles(image, tile_size=256):
    h, w = image.shape[:2]
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    path = "models/EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)
    sr.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # Create output image (4x larger)
    result = np.zeros((h * 4, w * 4, 3), dtype=np.uint8)

    # Add small overlap to avoid seams (padding of 8 pixels)
    overlap = 8

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Get tile coordinates (with overlap)
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)

            # Extract tile
            tile = image[y1:y2, x1:x2]

            # Process tile
            sr_tile = sr.upsample(tile)

            # Calculate positions in output image
            out_x1 = x1 * 4
            out_y1 = y1 * 4
            out_x2 = x2 * 4
            out_y2 = y2 * 4

            # Remove overlap edges except for border tiles
            if x > 0:
                out_x1 += overlap * 4
                sr_tile = sr_tile[:, overlap * 4 :]
            if y > 0:
                out_y1 += overlap * 4
                sr_tile = sr_tile[overlap * 4 :, :]

            # Place the tile in the result image
            result[out_y1:out_y2, out_x1:out_x2] = sr_tile

            # Let Python handle the cleanup
            sr_tile = None

    return result


def enhanced_edge_detection(
    image,
    bilateral_d=9,
    bilateral_sigma_color=75,
    bilateral_sigma_space=75,
    canny_low_threshold=50,
    canny_high_threshold=150,
):
    """
    Enhanced edge detection using bilateral filtering and Canny
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply bilateral filter for noise reduction while preserving edges
    bilateral = cv.bilateralFilter(
        gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    )

    # Apply Canny edge detection
    edges = cv.Canny(bilateral, canny_low_threshold, canny_high_threshold)

    # Convert edges to 3-channel if original image was color
    if len(image.shape) == 3:
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        return cv.addWeighted(image, 0.7, edges, 0.3, 0)

    return cv.addWeighted(gray, 0.7, edges, 0.3, 0)


def extract_roi(image, gaze_point, patch_size=200):
    h, w, _ = image.shape
    x_min = gaze_point[0][0] - patch_size if gaze_point[0][0] - patch_size > 0 else 0
    x_max = gaze_point[0][0] + patch_size if gaze_point[0][0] + patch_size < h else h
    y_min = gaze_point[0][1] - patch_size if gaze_point[0][1] - patch_size > 0 else 0
    y_max = gaze_point[0][1] + patch_size if gaze_point[0][1] + patch_size < w else w

    roi_image = copy.deepcopy(image[x_min:x_max, y_min:y_max])

    return roi_image


def local_contrast_enhancement(image, tile_size=(16, 16), clip_limit=3.0):
    """
    Local contrast enhancement using CLAHE with custom tile size
    """
    # Convert to LAB color space if image is BGR
    if len(image.shape) == 3:
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
    else:
        l = image.copy()

    # Create CLAHE object
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    # Apply CLAHE to luminance channel
    enhanced_l = clahe.apply(l)

    # Merge back if original was color
    if len(image.shape) == 3:
        enhanced_lab = cv.merge([enhanced_l, a, b])
        return cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

    return enhanced_l


def multiscale_edge_enhancement(image, scales=[1, 2, 4], weights=[0.5, 0.3, 0.2]):
    """
    Multi-scale edge enhancement using Gaussian pyramids
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Initialize result
    enhanced = np.zeros_like(gray, dtype=np.float32)

    # Process each scale
    for scale, weight in zip(scales, weights):
        # Gaussian blur for current scale
        kernel_size = int(2 * scale * 3 + 1)  # Ensure odd kernel size
        blurred = cv.GaussianBlur(gray, (kernel_size, kernel_size), scale)

        # Compute edges at current scale using Laplacian
        edges = cv.Laplacian(blurred, cv.CV_64F)

        # Normalize edges to 0-255 range
        edges = np.uint8(np.absolute(edges))

        # Add weighted edges to result
        enhanced += weight * edges.astype(np.float32)

    # Normalize final result
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Convert enhanced edges to 3-channel if original image was color
    if len(image.shape) == 3:
        enhanced = cv.cvtColor(enhanced, cv.COLOR_GRAY2BGR)
        return cv.addWeighted(image, 0.7, enhanced, 0.3, 0)

    return cv.addWeighted(gray, 0.7, enhanced, 0.3, 0)


def gamma_correction(image, gamma=1.0):

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv.LUT(image, table)


def egde_aug(image, color, thr1=25, thr2=80, alpha=0.85, beta=0.15):

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Unsharp masking
    image_sharp = image_sharpening(image_gray)

    # Contrast enchancement
    image_sharp = gamma_correction(image_sharp, gamma=1.5)

    # Edge detection
    edges = cv.Canny(image_sharp, thr1, thr2)

    mask = (edges > 0).astype(np.uint8)

    mask_3d = np.expand_dims(mask, axis=2)
    mask_3d = np.repeat(mask_3d, 3, axis=2)

    colored_mask = np.zeros_like(mask_3d)
    colored_mask[:, :] = color
    colored_mask *= mask_3d

    image = cv.addWeighted(image, alpha, colored_mask, beta, 0)

    return image


def gabor_edge_aug(
    image, num_filters=16, ksize=35, sigma=3.0, lambd=10.0, gamma=0.5, psi=0
):

    filters = []
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kernel = cv.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_32F
        )
        # Brightness normalization
        kernel /= 1.0 * kernel.sum()
        filters.append(kernel)

    new_image = np.zeros_like(image)

    depth = -1

    for kernel in filters:
        image_filter = cv.filter2D(image, depth, kernel)

        np.maximum(new_image, image_filter, new_image)

    return new_image
