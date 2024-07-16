import matplotlib.pyplot as plt
import numpy as np

# Your data
srgb_colors = np.array([
    [255, 0, 0],     # Red
    [0, 255, 0],     # Green
    [0, 0, 255],     # Blue
    [255, 255, 0],   # Yellow
    [255, 165, 0],   # Orange
    [255, 255, 255], # White
    [0, 0, 0]        # Black
])

# Create 800x480 image
image = np.zeros((480, 800, 3), dtype=np.uint8)

# Fill image with colors
for i in range(7):
    color = srgb_colors[i]
    x_start = i * 100
    x_end = (i + 1) * 100
    image[:, x_start:x_end, :] = color.astype(np.uint8)

# Display the image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis('off')
plt.title('Color Representation')
plt.show()
