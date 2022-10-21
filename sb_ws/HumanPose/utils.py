import os
from pathlib import Path
import cv2
import numpy as np
import shutil

from CONFIG import *


def convert_abs_path(local_path):
    local_path = str(Path(local_path))  # os-agnostic
    abs_path = os.path.abspath(local_path)  # absolute path
    return abs_path


def draw_keypoint(outputs, img_save=True):
    if img_save:
        out = convert_abs_path(POSE_IMG_SAVE_DIR)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    for output in outputs:
        filepath = output["file_path"][0]
        filename = os.path.basename(filepath)
        out_filepath = os.path.join(out, filename)
        if filename in os.listdir(out):
            img = cv2.imread(out_filepath)
        else:
            img = cv2.imread(filepath)
        pose_kps = output["pose_keys"][0]

        for pose_pair in POSE_PAIRS:
            cv2.circle(
                img,
                pose_kps[pose_pair[0]].astype(np.int32),
                KEY_POINT_CIRCLE_RADIUS,
                color=KEY_POINT_CIRCLE_COLOR,
                thickness=-1,
            )
            cv2.circle(
                img,
                pose_kps[pose_pair[1]].astype(np.int32),
                KEY_POINT_CIRCLE_RADIUS,
                color=KEY_POINT_CIRCLE_COLOR,
                thickness=-1,
            )
            cv2.line(
                img,
                pose_kps[pose_pair[0]].astype(np.int32),
                pose_kps[pose_pair[1]].astype(np.int32),
                color=(255, 0, 0),
                thickness=2,
            )
        cv2.imwrite(out_filepath, img)
