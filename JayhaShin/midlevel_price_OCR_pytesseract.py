'''
pytesseract
oem psm 모드 설명
https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc

!sudo apt install tesseract-ocr
!pip install pytesseract
!pip install tesseract
!sudo apt-get install tesseract-ocr-eng         # optional
!sudo apt-get install tesseract-ocr-kor         # optional
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
from _ocr_common_func import _img_ocr_result, _heatmap_2
def _ocr_pytesseract(img):
    traindata = None       # 'kor' 'eng' 'kor+eng' None
    psm = 6
    oem = 3
    ocr_result = pytesseract.image_to_data(image=img, lang=traindata, config='--oem '+str(oem)+' --psm '+str(psm), output_type='data.frame')
    ocr_result = ocr_result.dropna(axis=0).reset_index(drop=True)
    for idx, str_item in enumerate(ocr_result['text']):
        for char in str_item:
            if not char.isdigit():
                str_item = str_item.replace(char,"")
                ocr_result.loc[idx, 'text'] = str_item       # remove non-numeric char from DataFrame
            if len(str_item)<3:
                ocr_result.loc[idx, 'text'] = ""
    ocr_result = ocr_result[ocr_result['text']!=""]
    ocr_result = ocr_result.reset_index(drop=True)
    ocr_result = ocr_result[['left','top','width','height','conf','text']]
    return ocr_result


IMG_DIR = 'JayhaShin/sample_price.jpeg'
img = plt.imread(IMG_DIR)
dh, dw, _ = img.shape

plt.figure(figsize=(18,3))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
ocr_result = _ocr_pytesseract(img)
img_with_ocr = _img_ocr_result(img, ocr_result)   # plt.text
plt.imshow(img_with_ocr)
plt.subplot(1, 3, 3)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()
plt.show()

print(ocr_result)
