import numpy as np
import cv2

# Load the segmentation label matrix
file_path = "/home/yousef/shared_for_test/test_results_EMSGC/echo_bird_combinations/seg_labels/0.489271167.txt"

import numpy as np
import cv2

# Load the text file

with open(file_path, "r") as f:
    lines = f.readlines()

# Extract pixel coordinates and labels
pixels = [list(map(int, line.split())) for line in lines]

# Determine image size (assuming max x, y from data)
max_x = max(p[0] for p in pixels)
max_y = max(p[1] for p in pixels)

# Create a blank black image
binary_image = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)

for x, y, label in pixels:
    if label == 1:
        binary_image[y, x] = 255  # Correct order is (y, x)

# Save the binary image
cv2.imwrite("output.png", binary_image)

print("Binary image saved as output.png")

