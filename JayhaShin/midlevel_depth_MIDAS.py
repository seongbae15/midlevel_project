# !pip install timm

IMG_DIR_2 = 'JayhaShin/sample_depth.jpeg'

import cv2
import torch
import time
import numpy as np

# Load a MiDas model for depth estimation
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(IMG_DIR_2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)
# Prediction and resize to original resolution
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
depth_map = prediction.cpu().numpy()
depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
depth_map = (depth_map*255).astype(np.uint8)
depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)



import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img)
plt.subplot(1,2,2)
plt.title('Target(Mid-Level)')
plt.imshow(depth_map)
# plt.colorbar()
plt.show()
# print(depth_map.shape)



# 3D plot
(y,x) = depth_map.shape
x = range(x)
y = range(y)
data = depth_map
ax1 = plt.figure(figsize=(15,10))
ax1 = ax1.add_subplot(111, projection='3d')
ax1.view_init(30, -70)  # (elevation, azimuth) of the axes in degrees
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.set_ylim(max(data.flatten()),0)     # z and y switch
ax1.set_zlim(max(y),0)                  # z and y switch
X, Y = numpy.meshgrid(x, y)
ax1.plot_surface(X, data, Y)            # z and y switch
plt.show()