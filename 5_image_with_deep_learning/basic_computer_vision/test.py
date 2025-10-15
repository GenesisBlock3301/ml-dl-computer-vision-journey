import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generate a random high-contrast image (simulate edges everywhere)
random_state = 42
rng = np.random.default_rng(random_state)

# Generate random integer image
img = rng.integers(low=0, high=256, size=(256, 256), dtype=np.uint8)

# Apply edge detection kernel (simple Laplacian-like)
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]], dtype=np.float32)
feature_map = cv2.filter2D(img, -1, kernel)

# Define max pooling function
def max_pooling(feature_map, pool_size=2, stride=2):
    h, w = feature_map.shape
    new_h = h // stride
    new_w = w // stride
    pooled = np.zeros((new_h, new_w), dtype=feature_map.dtype)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            window = feature_map[i:i+pool_size, j:j+pool_size]
            pooled[i//stride, j//stride] = np.max(window)
    return pooled

# Apply max pooling
pooled_map = max_pooling(feature_map, pool_size=2, stride=2)

# Visualize
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.title("Original random image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Feature map (edges)")
plt.imshow(feature_map, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Max pooled (2x2)")
plt.imshow(pooled_map, cmap='gray')
plt.axis('off')

plt.show()
