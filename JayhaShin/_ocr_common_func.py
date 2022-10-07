import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def _img_ocr_result(img, ocr_result, FONT_SIZE=10):
    for _, row in ocr_result.iterrows():
        if row['conf']>0:       # threshold
            x, y, w, h = row['left'], row['top'], row['width'], row['height']
            plt.text(x, (y-10), row['text'], fontsize=FONT_SIZE, color='red')
            cv2.rectangle(img, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (255,0,0), 1)
    return img

# radial heatmap
from scipy.interpolate.rbf import Rbf  # radial basis functions
def _heatmap_2(img, ocr_result):
    if len(ocr_result) <= 2 :
        ocr_result = pd.DataFrame(data=[], columns=['left','top','width','height','conf','text'])
        ocr_result.loc[len(ocr_result)] = [0, 0, 0, 0, 0., '1']
        ocr_result.loc[len(ocr_result)] = [0, 1, 0, 0, 0., '1']
        # avoid [ValueError: zero-size array to reduction operation maximum which has no identity]
    x = ocr_result['left']+(ocr_result['width']//2)
    y = ocr_result['top']+(ocr_result['height']//2)
    z = ocr_result['text'].astype(np.int64)
    # https://stackoverflow.com/questions/51647590/2d-probability-distribution-with-rbf-and-scipy
    rbf_adj = Rbf(x, y, z, function='gaussian')
    dh, dw, _ = img.shape
    x_fine = np.linspace(0, dw, num=81)  # (start, stop, step)
    y_fine = np.linspace(0, dh, num=82)
    x_grid, y_grid = np.meshgrid(x_fine, y_fine)
    z_grid = rbf_adj(x_grid.ravel(), y_grid.ravel()).reshape(x_grid.shape)
    return x_fine, y_fine, z_grid