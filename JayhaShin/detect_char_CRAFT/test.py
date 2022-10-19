"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import time
starttime=time.time()

import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import zipfile
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from skimage import io

import modules.craft_utils as craft_utils
import modules.imgproc as imgproc
import modules.file_utils as file_utils
from modules.craft import CRAFT
import modules.hangul_font_for_pyplot
from modules.ocr_func import _img_ocr_result, _heatmap_2, _kmeanclustered, _ocr_easyocr_whole_img


# Download weight "craft_mlt_25k.pth" from Google drive
if not os.path.isdir('./weights/'):
    os.mkdir('./weights/')
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
parser.add_argument('--link_threshold', default=np.inf, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=np.inf, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1., type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./input/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()





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
    heatmapimg_ndarray = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, heatmapimg_ndarray



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
    from modules.refinenet import RefineNet
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


""" Detection """
# load data
image_list, _, _ = file_utils.get_files(args.test_folder)  # receive only first image file
img_file = image_list[0]
original_img_ndarray = plt.imread(img_file)
dh, dw, _ = original_img_ndarray.shape
original_img_ndarray = _kmeanclustered(original_img_ndarray, 64)


RESULT_DIR='./result/'
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)

plt.imsave(RESULT_DIR+"res_1_original.jpg", original_img_ndarray)


print("Test image: {:s}".format(img_file), end='\r')
image = imgproc.loadImage(img_file)
bboxes, polys, heatmapimg_ndarray = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)


# save detection heatmap
filename, file_ext = os.path.splitext(os.path.basename(img_file))
mask_file = RESULT_DIR + "/res_2_heatmapimg.jpg"
cv2.imwrite(mask_file, heatmapimg_ndarray)


# polys(ndarray) --> ocr_result(dataframe)
tl, _, br, _ = map(list, zip(*polys))
tl_x, tl_y = map(list, zip(*tl))
br_x, br_y = map(list, zip(*br))
ocr_result = pd.DataFrame(data=[], columns=["left", "top", "width", "height", "conf", "text"])
ocr_result["left"] = tl_x
ocr_result["top"] = tl_y
ocr_result["width"] = list(np.subtract(br_x, tl_x))
ocr_result["height"] = list(np.subtract(br_y, tl_y))
ocr_result["conf"] = 0.       # update after OCR
ocr_result["text"] = "nan"    # update after OCR


""" Detection Result Check """
plt.figure(figsize=((dw/max(dw,dh)*20)*3,(dh/max(dw,dh)*20)) )
plt.subplot(131)
plt.imshow(original_img_ndarray)
plt.subplot(132)
img_mask = cv2.resize(heatmapimg_ndarray, dsize=(dw*2, dh), interpolation=cv2.INTER_CUBIC)
plt.imshow(original_img_ndarray)
plt.imshow(img_mask[:dh,:dw//2], alpha=0.5)
plt.subplot(133)
plt.imshow(img_mask[:dh,dw//2:])
# plt.show()      # optional
plt.savefig(RESULT_DIR+"res_3_subplot.jpg")


""" Detection Result --> Extract(OCR) each character """
import easyocr
reader = easyocr.Reader(["en", "ko"])  # ['en', 'ko']
img = original_img_ndarray.copy()
MARGIN = 5
for idx, row in ocr_result.iterrows():
    x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
    img_portion = img[max(0,y-MARGIN):min(dh,(y+h)+MARGIN), max(0,x-MARGIN):min(dw,(x+w)+MARGIN), :].copy()
    bounds = reader.readtext(
        img_portion,
        blocklist="DoOㅇ이미aSsLlIiUu그",  # I need number, not alphabet
        min_size=0,
        low_text=0.1,
        # rotation_info=[-10,0,10],   # Not good, it gets creative. It reads 3 as 9.
        canvas_size=np.inf
    )  # optional (allowlist ='0123456789')
    # print(bounds)
    if len(bounds) != 0 :
        ocr_result.loc[idx, "text"] = str(bounds[0][1])
        ocr_result.loc[idx, "conf"] = float(bounds[0][2])
print("Quick OCR performance check (null/len) :",np.sum(ocr_result['text']=="nan"),"/",len(ocr_result) )


""" Before masking non-numeric character """
# type result and draw bbox over image
plt.figure(figsize=((dw/max(dw,dh)*10),(dh/max(dw,dh)*10)) )
img_with_ocr = original_img_ndarray.copy()
img_with_ocr = _img_ocr_result(img_with_ocr, ocr_result, FONT_SIZE=8)  # plt.text
plt.imshow(img_with_ocr)
plt.savefig(RESULT_DIR+"res_4_img_with_ocr.jpg")
# plt.show()


""" Remove/Mask non-numeric character """
# numeric
for idx, str_item in enumerate(ocr_result["text"]):
    for char in str_item:
        if not char.isdigit():
            str_item = str_item.replace(char, "")
            ocr_result.loc[idx, "text"] = str_item  # remove non-numeric char from DataFrame, such as comma
# mask the non-numerics in img
img_masked = original_img_ndarray.copy()
for _, row in ocr_result.iterrows():
    if not row["text"].isdigit():
        x, y, w, h = row["left"], row["top"], row["width"], row["height"]
        cv2.rectangle(img_masked, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), -1 )
# return img
ocr_result = ocr_result[ocr_result["text"] != ""].reset_index(drop=True)

plt.imsave(RESULT_DIR+"res_5_digit_only.jpg", img_masked)



""" Extract(OCR) from masked image """
ocr_result = _ocr_easyocr_whole_img(img_masked, CONFIDENCE=0.0)    # ocr_result overwrite
ocr_result.to_csv(f'{RESULT_DIR}res_6_ocr_result.csv', sep=',')
img = original_img_ndarray.copy()
plt.figure(figsize=((dw/max(dw,dh)*30)*2,(dh/max(dw,dh)*30)) )
plt.subplot(121)
img_with_ocr = _img_ocr_result(img, ocr_result, FONT_SIZE=16)  # plt.text
plt.imshow(img_with_ocr)
plt.subplot(122)
x_fine, y_fine, z_grid = _heatmap_2(img, ocr_result)
dh, dw, _ = img.shape
plt.ylim(dh, 0)
plt.pcolor(x_fine, y_fine, z_grid)
plt.colorbar()
# plt.show()      # optional
plt.savefig(RESULT_DIR+"res_7_subplot.jpg")




print('Run Duration: {:.0f}m {:.0f}s'.format( (time.time()-starttime)//60, (time.time()-starttime)%60) )
