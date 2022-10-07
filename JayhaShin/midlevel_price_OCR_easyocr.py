'''
EasyOCR
https://github.com/JaidedAI/EasyOCR/blob/master/README.md
https://www.jaided.ai/easyocr/documentation/

!pip install easyocr
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _ocr_common_func import _img_ocr_result, _heatmap_2
import easyocr
def _ocr_easyocr(IMGDIR_or_IMAGE, CONFIDENCE=0.95):
    reader = easyocr.Reader(['ko'])  # ['en', 'ko']
    bounds = reader.readtext(IMGDIR_or_IMAGE )   # optional (allowlist ='0123456789')
    ocr_result = pd.DataFrame(data=[], columns=['left','top','width','height','conf','text'])
    if len(bounds) != 0 :
        # EacyOCR result to dataframe
        coordinates, ocr_result_text, ocr_result_conf = map(list, zip(*bounds))  # Splitting nested list
        tl, _, br, _ = map(list, zip(*coordinates))
        tl_x, tl_y = map(list, zip(*tl))
        br_x, br_y = map(list, zip(*br))
        ocr_result['left'] = tl_x
        ocr_result['top'] = tl_y
        ocr_result['width'] = list(np.subtract(br_x, tl_x))
        ocr_result['height'] = list(np.subtract(br_y, tl_y))
        ocr_result['conf'] = ocr_result_conf
        ocr_result['text'] = ocr_result_text

        # remove non-numeric char from DataFrame
        ocr_result = ocr_result.dropna(axis=0).reset_index(drop=True)
        for idx, str_item in enumerate(ocr_result['text']):
            for char in str_item:
                if not char.isdigit():
                    str_item = str_item.replace(char,"")
                    ocr_result.loc[idx, 'text'] = str_item       # remove non-numeric char from DataFrame
                if len(str_item)<3:
                    ocr_result.loc[idx, 'text'] = ""
        ocr_result = ocr_result[ocr_result['text']!=""].reset_index(drop=True)

        # OCR 후처리 - 숫자 오역 처리
        # 주로 일의 자리에는 "0"을 제외하고 다른 숫자가 들어가지 않는다. (반례 1234, 예 1230 )
        # 주로 가격에 "0"을 제외하고 숫자 4개 이상 들어가지 않는다. (반례 1_234_500, 예 1_230_000 )
        for idx, str_item in enumerate(ocr_result['text']):
            if ocr_result.loc[idx, 'conf'] < CONFIDENCE:       # threshold
                ocr_result.loc[idx, 'text'] = ""
                continue
            if len(str_item) < 3 :
                ocr_result.loc[idx, 'text'] = ""
                continue
            if int(str_item) == 0 :
                ocr_result.loc[idx, 'text'] = ""
                continue
            if str_item[-1] != "0":
                str_item = str_item[:-1]
                ocr_result.loc[idx, 'text'] = str_item
        #     if len(str_item) > 5 :
        #         if len(str_item[:-2].replace("0","")) > 3:
        #             ocr_result.loc[idx, 'text'] = ""
        ocr_result = ocr_result[ocr_result['text']!=""].reset_index(drop=True)

        # removing outlier over quantile
        # ocr_result['text'] = ocr_result['text'].astype(int)
        # q_low = ocr_result['text'].quantile(0.05)
        # q_hi  = ocr_result['text'].quantile(0.90)
        # ocr_result = ocr_result[(ocr_result['text'] < q_hi) & (ocr_result['text'] > q_low)]
        # ocr_result = ocr_result[ocr_result['text']!=""].reset_index(drop=True)

    return ocr_result


IMG_DIR = 'JayhaShin/sample_price.jpeg'
img = plt.imread(IMG_DIR)
dh, dw, _ = img.shape

plt.figure(figsize=(18,3))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
ocr_result = _ocr_easyocr(img, CONFIDENCE=0.5)
img_with_ocr = _img_ocr_result(img, ocr_result)   # plt.text
plt.imshow(img_with_ocr)
plt.subplot(1, 3, 3)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()
plt.show()

print(ocr_result)
