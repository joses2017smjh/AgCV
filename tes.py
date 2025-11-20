import os
import cv2
import numpy as np

path = r"C:\Users\joses\Desktop\tree_dataset\depth\depth_0001.exr"

depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
print("dtype:", depth.dtype)
print("min:", depth.min(), "max:", depth.max())

# If it has multiple channels, take the first
if depth.ndim == 3:
    depth = depth[:, :, 0]

print("after squeeze -> min:", depth.min(), "max:", depth.max())
