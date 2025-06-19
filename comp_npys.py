# compare_npy.py
import numpy as np
import os

def l2_diff(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Load the two .npy files
file1 = "./preprocessed_images/i1.npy"
file2 = "./preprocessed_images/i2.npy"  # or another source of the same image

image1 = np.load(file1)
image2 = np.load(file2)
print("image 1 is")
print(image1)
print("image 2 is")
print(image2)
# Ensure shapes match
if image1.shape != image2.shape:
    raise ValueError(f"Shape mismatch: {image1.shape} vs {image2.shape}")

# Compute L2 difference
diff = l2_diff(image1, image2)
print(f"L2 Difference: {diff}")
