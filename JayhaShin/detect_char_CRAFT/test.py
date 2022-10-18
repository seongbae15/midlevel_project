"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT


# Download weight "craft_mlt_25k.pth" from Google drive
import gdown
url = 'https://drive.google.com/file/d/1I0hnVhHHLKBGOrqYXHc8x19bssqzpsva/view?usp=sharing'
output = './weights/craft_mlt_25k.pth'
gdown.download(url=url, output=output, quiet=False, fuzzy=True)


from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=-np.inf, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=np.inf, type=float, help='link confidence threshold')                  # 0.4  낮으면 전부 합친다
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=np.inf, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1., type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./input/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



""" Detection """
# load net
net = CRAFT()     # initialize

print('Loading weights from checkpoint (' + args.trained_model + ')')
if args.cuda:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
else:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

if args.cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()

# LinkRefiner
refine_net = None
if args.refine:
    from refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    refine_net.eval()
    args.poly = True

t = time.time()

# load data
for k, image_path in enumerate(image_list):
    print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_2_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

print("elapsed time : {}s".format(time.time() - t))




""" Detection Result --> Extraction """
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIR = './result/'

TEMP_LIST = list(os.listdir(DIR))
RES_IMG_JPG = [x for x in TEMP_LIST if x.endswith("_img.jpg")]
RES_MASK_JPG = [x for x in TEMP_LIST if x.endswith("_mask.jpg")]
RES_TXT = [x for x in TEMP_LIST if x.endswith(".txt")]
ORIGINAL_IMG = [x for x in TEMP_LIST if x.endswith("_orig.jpg")]
print(RES_IMG_JPG[0])
print(RES_MASK_JPG[0])
print(RES_TXT[0])


# plt.savefig(DIR+"res_4_subplot.jpg")

res_txt = pd.read_csv(DIR+RES_TXT[0], sep=",", header=None)
ocr_result = pd.DataFrame(
    data=[], columns=["left", "top", "width", "height", "conf", "text"]
)
if len(res_txt) != 0:
    ocr_result["left"] = res_txt.loc[:,0]
    ocr_result["top"] = res_txt.loc[:,1]
    ocr_result["width"] = list(np.subtract(res_txt.loc[:,4], res_txt.loc[:,0]))
    ocr_result["height"] = list(np.subtract(res_txt.loc[:,5], res_txt.loc[:,1]))
    ocr_result["conf"] = 0.
    ocr_result["text"] = "nan"
# print(ocr_result)




""" OCR each character """
import easyocr
reader = easyocr.Reader(["en", "ko"])  # ['en', 'ko']
img = plt.imread(DIR+ORIGINAL_IMG[0])
dh, dw, _ = img.shape
MARGIN = 5
for idx, row in ocr_result.iterrows():
    x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
    img_portion = img[max(0,y-MARGIN):min(dh,(y+h)+MARGIN), max(0,x-MARGIN):min(dw,(x+w)+MARGIN), :].copy()
    bounds = reader.readtext(
        img_portion,
        blocklist="DoOㅇaSsLlIi",  # alphabet
        min_size=0,
        low_text=0.1,
        canvas_size=np.inf
    )  # optional (allowlist ='0123456789')
    # print(bounds)
    if len(bounds) != 0 :
        ocr_result.loc[idx, "text"] = str(bounds[0][1])
        ocr_result.loc[idx, "conf"] = float(bounds[0][2])
    # cv2.rectangle(img, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (255, 0, 0), 1 ) # -1 은 채운다
#
print("nan/len =",np.sum(ocr_result['text']=="nan"),"/",len(ocr_result) )
# print(ocr_result)



""" Before masking non-numeric character """
# type result and draw bbox over image
import _hangul_font_for_pyplot
def _img_ocr_result(img, ocr_result, FONT_SIZE=10):
    for idx, row in ocr_result.iterrows():
        if row["conf"] >= 0:  # threshold
            x, y, w, h = row["left"], row["top"], row["width"], row["height"]
            plt.text(x, (y - 10), row["text"], fontsize=FONT_SIZE, color="red")
            cv2.rectangle(
                img,
                (int(x), int(y)),
                (int(x) + int(w), int(y) + int(h)),
                (255, 0, 0),
                1,
            )
    return img
import cv2
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
img_with_ocr = plt.imread(DIR+ORIGINAL_IMG[0])
img_with_ocr = _img_ocr_result(img_with_ocr, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.savefig(DIR+"res_5_img_with_ocr.jpg")
# plt.show()




""" Remove/Mask non-numeric character """
# numeric
for idx, str_item in enumerate(ocr_result["text"]):
    for char in str_item:
        if not char.isdigit():
            str_item = str_item.replace(char, "")
            ocr_result.loc[idx, "text"] = str_item  # remove non-numeric char from DataFrame, such as comma
# return ocr_result
# mask the non-numerics in img
img_masked = img.copy()
for _, row in ocr_result.iterrows():
    if not row["text"].isdigit():
        x, y, w, h = row["left"], row["top"], row["width"], row["height"]
        cv2.rectangle(img_masked, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), -1 )
# return img
ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)

cv2.imwrite(DIR+"res_6_digit_only.jpg", img_masked)

# plt.figure(figsize=(20,5))
# plt.imshow(img_masked)
# plt.show()