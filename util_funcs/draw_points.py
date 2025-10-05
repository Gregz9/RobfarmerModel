import numpy as np
import cv2

# Your coordinates
coordinates = np.array([
    [782.2722168, 782.57672119],
    [789.88189697, 784.28503418],
    [793.55645752, 785.45617676],
    [800.49298096, 784.50909424],
    [813.10290527, 778.47821045],
    [832.34143066, 752.59301758],
    [837.48858643, 751.20635986],
    [841.93438721, 756.81884766],
    [844.00994873, 754.25488281],
    [851.32214355, 750.74890137]
])

# Create a blank image
# The size should be large enough to contain all points
# img_size = (1000, 1000)  # Adjust based on your coordinate range
# img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
img = cv2.imread("../../data/datasets/Robofarmer/P01/rgb_frames/STBR_01/frame_0000000106.jpg")
original_height, original_width = img.shape[:2]
# If you want a white background instead:
# img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
new_size = (1250, 1080)

# Resize image
img_resized = cv2.resize(img, new_size)

# Scale coordinates
scale_x = 750 / original_width
scale_y = 750 / original_height

# Apply scaling to coordinates
scaled_coordinates = coordinates.copy()
scaled_coordinates[:, 0] *= scale_x  # Scale x coordinates
scaled_coordinates[:, 1] *= scale_y  # Scale y coordinates

# Convert to integers for drawing
scaled_coordinates = scaled_coordinates.astype(np.int32)

# Draw points on resized image
for point in scaled_coordinates:
    cv2.circle(img_resized, (point[0], point[1]), 3, (0, 0, 255), -1)  # Red dots
    # Made point size smaller (radius=3) for the smaller image

# Display the image
cv2.imshow('Points on Resized Image', img_resized)
cv2.waitKey()
cv2.destroyAllWindows()
