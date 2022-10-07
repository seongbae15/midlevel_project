import numpy as np
import matplotlib.pyplot as plt
import cv2

sample_path = "/Users/krc/Documents/retail/retail_gh/data_sample/"
image = cv2.imread(sample_path + "can_intu/0922c000001.jpg")

roi_sample = np.genfromtxt(
    "/Users/krc/Documents/retail/retail_gh/data_sample/can_intu/0922c000001.txt"
)

mask = roi_sample[0]
print(type(roi_sample))
print(mask[1])

"""
cv2.imshow("image", image)
cv2.waitKey()

cv2.destroyAllWindows()
"""
