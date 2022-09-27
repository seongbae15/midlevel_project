import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

file_path = './data_sample'
file_list = os.walk(file_path) # OS로 불러오기 
img_files = [file for file in file_list if file[-1][-1].endswith('.jpg')]


if not img_files: # if empty folder
    print('there are no jpg files')
    sys.exit()

''' check of printed path 
for f, g, h in img_files:
    print(f+'/'+h[0])
'''


for f, g, h  in img_files:
    # t.ly/zgLP
    img = cv2.imread(f+'/'+h[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5 # N of colors
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    cv2.imwrite(f'{h[0]}_cluster.jpg', result_image)

    cv2.imshow('Original Image',img)
    cv2.imshow('Segmented Image when K = %i' % K, result_image)
    cv2.waitKey()  # 키보드 입력이 들어올 때까지 기다리기

    cv2.destroyAllWindows()
