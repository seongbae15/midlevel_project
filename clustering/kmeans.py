import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

file_list = os.listdir('../data_sample') # OS로 불러오기 
img_files = [file for file in file_list if file.endswith('.jpg')]

if not img_files: # if empty folder
    print('there are no png files')
    sys.exit()

for f in img_files:
    print(f)