import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

FRAME = 1
BASE = Path("/mnt/c/Users/joses/Desktop/tree_dataset")   # WSL path
rgb_path = BASE / "rgb" / f"rgb_{FRAME:04d}.png"
ann_path = BASE / "ann" / f"frame_{FRAME:04d}.json"

img = cv2.imread(str(rgb_path))
if img is None:
    raise FileNotFoundError(f"Could not read image at {rgb_path}. Please check that the file exists and the BASE path is correct.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if not ann_path.exists():
    raise FileNotFoundError(f"Could not find annotation file at {ann_path}. Please check that the file exists and the BASE path is correct.")

with open(ann_path, "r") as f:
    ann = json.load(f)

K = np.array(ann["camera"]["intrinsics"]["K"])
width = ann["camera"]["intrinsics"]["width"]
height = ann["camera"]["intrinsics"]["height"]

# Project world point to image pixel
def project_point(world_point):
    X = np.array(world_point).reshape(3, 1)
    P = X  # no extrinsics here because Blender annotation is already world coords
    uvw = K @ P
    u = uvw[0, 0] / uvw[2, 0]
    v = uvw[1, 0] / uvw[2, 0]
    return int(u), int(v)

# Draw each point on image
for p in ann["points"]:
    u, v = project_point(p["pos_world"])
    if 0 <= u < width and 0 <= v < height:
        cv2.circle(img, (u, v), 5, (255, 0, 0), -1)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title(f"Projected 10cm Points on Frame {FRAME}")
plt.axis("off")
plt.show()
