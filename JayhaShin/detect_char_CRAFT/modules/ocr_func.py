import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# type result and draw bbox over image
def _img_ocr_result(img, ocr_result, FONT_SIZE=10):
    for _, row in ocr_result.iterrows():
        if row["conf"] > 0:  # threshold
            x, y, w, h = row["left"], row["top"], row["width"], row["height"]
            plt.text(x, (y - 10), row["text"], fontsize=FONT_SIZE, color="red")
            cv2.rectangle(img,(int(x), int(y)),(int(x) + int(w), int(y) + int(h)),(255, 0, 0),1 )
    return img


# K-means Clustering
def _kmeanclustered(img, KNUM=2):
    img_2 = img.copy().reshape(-1, 3)
    img_2 = np.float32(img_2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = KNUM
    ret, label, center = cv2.kmeans(
        img_2, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((img.shape))
    return img


# radial heatmap
def _heatmap_2(img, ocr_result):
    from scipy.interpolate.rbf import Rbf  # radial basis functions
    if len(ocr_result) <= 2:
        ocr_result = pd.DataFrame(data=[], columns=["left", "top", "width", "height", "conf", "text"])
        ocr_result.loc[len(ocr_result)] = [0, 0, 0, 0, 0.0, "1"]
        ocr_result.loc[len(ocr_result)] = [0, 1, 0, 0, 0.0, "1"]
        # avoid [ValueError: zero-size array to reduction operation maximum which has no identity]
    x = ocr_result["left"] + (ocr_result["width"] // 2)
    y = ocr_result["top"] + (ocr_result["height"] // 2)
    z = ocr_result["text"].astype(np.int64)
    # https://stackoverflow.com/questions/51647590/2d-probability-distribution-with-rbf-and-scipy
    rbf_adj = Rbf(x, y, z, function="gaussian")
    dh, dw, _ = img.shape
    x_fine = np.linspace(0, dw, num=81)  # (start, stop, step)
    y_fine = np.linspace(0, dh, num=82)
    x_grid, y_grid = np.meshgrid(x_fine, y_fine)
    z_grid = rbf_adj(x_grid.ravel(), y_grid.ravel()).reshape(x_grid.shape)
    return x_fine, y_fine, z_grid


# pytesseract
def _ocr_pytesseract(img):
    import pytesseract
    traindata = None  # 'kor' 'eng' 'kor+eng' None
    psm = 6
    oem = 3
    ocr_result = pytesseract.image_to_data(
        image=img,
        lang=traindata,
        config="--oem " + str(oem) + " --psm " + str(psm),
        output_type="data.frame",
    )
    ocr_result = ocr_result.dropna(axis=0).reset_index(drop=True)
    for idx, str_item in enumerate(ocr_result["text"]):
        for char in str_item:
            if not char.isdigit():
                str_item = str_item.replace(char, "")
                ocr_result.loc[
                    idx, "text"
                ] = str_item  # remove non-numeric char from DataFrame
            if len(str_item) < 3:
                ocr_result.loc[idx, "text"] = ""
    ocr_result = ocr_result[ocr_result["text"] != ""]
    ocr_result = ocr_result.reset_index(drop=True)
    ocr_result = ocr_result[["left", "top", "width", "height", "conf", "text"]]
    return ocr_result



# EasyOCR
def _ocr_easyocr_whole_img(IMGDIR_or_IMAGE, CONFIDENCE=0.95):
    import easyocr
    reader = easyocr.Reader(["en", "ko"])  # ['en', 'ko']
    bounds = reader.readtext(
        IMGDIR_or_IMAGE,
        blocklist="DoOㅇ이미aSsLlIiUu그",  # I need number, not alphabet
        min_size=5,
        # rotation_info=[-20, 0, 20],
        low_text=0.1,
        link_threshold=30.0,
        canvas_size=np.inf,  # Image bigger than this value will be resized down
        ycenter_ths=0.0,  # maximum vertical shift
        height_ths=0.0,  # maximum different height
        width_ths=0.0,  # maximum horizontal shift
        add_margin=0.0,
        x_ths=0.0,
        y_ths=0.0,
    )  # optional (allowlist ='0123456789')
    ocr_result = pd.DataFrame(
        data=[], columns=["left", "top", "width", "height", "conf", "text"]
    )
    if len(bounds) != 0:
        # EacyOCR result to dataframe
        coordinates, ocr_result_text, ocr_result_conf = map(
            list, zip(*bounds)
        )  # Splitting nested list
        tl, _, br, _ = map(list, zip(*coordinates))
        tl_x, tl_y = map(list, zip(*tl))
        br_x, br_y = map(list, zip(*br))
        ocr_result["left"] = tl_x
        ocr_result["top"] = tl_y
        ocr_result["width"] = list(np.subtract(br_x, tl_x))
        ocr_result["height"] = list(np.subtract(br_y, tl_y))
        ocr_result["conf"] = ocr_result_conf
        ocr_result["text"] = ocr_result_text
        # remove non-numeric char from DataFrame
        # ocr_result.to_csv("temp_ocr_result_before_preprocess.csv")  # optional
        ocr_result = ocr_result.dropna(axis=0).reset_index(drop=True)
        for idx, str_item in enumerate(ocr_result["text"]):
            for char in str_item:
                if not char.isdigit():
                    str_item = str_item.replace(char, "")
                    ocr_result.loc[
                        idx, "text"
                    ] = str_item  # remove non-numeric char from DataFrame, such as comma
                if len(str_item) < 3:
                    ocr_result.loc[idx, "text"] = ""
        ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
        # OCR 후처리 - 숫자 오역 처리
        # 백 원 미만 무시
        # 주로 백의 자리 미만에는 "0"을 제외하고 다른 숫자가 들어가지 않는다. (반례 123, 예 100 )
        # 주로 가격에 "0"을 제외하고 숫자 4개 이상 들어가지 않는다. (반례 1_234_500, 예 1_230_000 )
        for idx, str_item in enumerate(ocr_result["text"]):
            if ocr_result.loc[idx, "conf"] < CONFIDENCE:  # threshold
                ocr_result.loc[idx, "text"] = ""
                continue
            if len(str_item) < 3:
                ocr_result.loc[idx, "text"] = ""
                continue
            if int(str_item) == 0:
                ocr_result.loc[idx, "text"] = ""
                continue
            if (str_item[-2] + str_item[-1]) != "00":
                ocr_result.loc[idx, "text"] = ""
                continue
            if len(str_item) > 5:
                if len(str_item[:-2].replace("0", "")) > 3:
                    ocr_result.loc[idx, "text"] = ""
        ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
        # removing outlier over quantile
        # ocr_result["text"] = ocr_result["text"].astype(int)
        # q_low = ocr_result['text'].quantile(0.05)
        # q_hi = ocr_result["text"].quantile(0.90)
        # print("upper_quantile:", q_hi)  # optional
        # ocr_result = ocr_result[ocr_result['text'] > q_low]
        # ocr_result = ocr_result[ocr_result["text"] <= q_hi]
        # ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)
    return ocr_result
#