import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

file_path = "./data_sample"
file_list = os.walk(file_path)  # OS로 불러오기
img_files = [file for file in file_list if file[-1][-1].endswith(".jpg")]


if not img_files:  # if empty folder
    print("there are no jpg files")
    sys.exit()

""" # check of printed path 
for f, g, h in img_files:
    print(f,g,h)
    print(f+'/'+h[0])
"""

for i, f in enumerate(img_files[0][2]):
    # t.ly/zgLP
    img = cv2.imread(file_path + "/" + f)
    img = cv2.resize(
        img, dsize=(640, 480), interpolation=cv2.INTER_LINEAR
    )  # image resize
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
    K = 12  # N of colors
    attempts = 20
    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    center = np.uint8(center)

    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    cv2.imwrite(f"./image_result/{f}_cluster.jpg", result_image)

    cv2.imshow("Original Image", img)
    cv2.imshow("Segmented Image when K = %i" % K, result_image)
    cv2.waitKey()  # 키보드 입력이 들어올 때까지 기다리기

    cv2.destroyAllWindows()
