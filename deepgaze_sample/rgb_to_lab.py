from skimage import color
import torch
from torchvision import datasets, transforms
import numpy as np
import cv2

image = cv2.imread(
    "/Users/krc/Documents/retail/retail_gh/data_sample/can_intu/0922c000181.jpg"
)

imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
lab = color.rgb2lab(image)

cv2.imshow("image", image)
cv2.imshow("labimage", lab)
cv2.waitKey()

cv2.destroyAllWindows()
