from torch.utils.data import Dataset
from CONFIG import *

from copy import deepcopy
import cv2
import numpy as np


class CustomPoseEstDataset(Dataset):
    def __init__(self, det_out_list, transform=None):
        super().__init__()
        self.det_out = det_out_list
        self.transform = transform
        self.in_img_height = POSE_IMG_HEIGHT
        self.in_img_width = POSE_IMG_WIDTH
        self.aspect_ratio = self.in_img_height * 1.0 / self.in_img_width
        self.pixel_std = 200
        self.num_joint = POSE_CFG.MODEL.NUM_JOINTS
        self.db = self._get_db(det_out_list)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        db = deepcopy(self.db[index])
        image_file = db["file_path"]
        # np_image = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        np_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if np_image is None:
            raise ValueError("Fail to read {}".format(image_file))

        c = db["center"]
        s = db["scale"]
        r = 0

        trans = get_affine_transform(c, s, r, (int(self.in_img_width), int(self.in_img_height)))
        input = cv2.warpAffine(np_image, trans, (int(self.in_img_width), int(self.in_img_height)), flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        return input, db

    def _get_db(self, det_out_list):
        pose_est_input_list = deepcopy(det_out_list)
        db = []
        for pose_est_input in pose_est_input_list:
            img = cv2.imread(pose_est_input["file_path"])
            height = img.shape[0]
            width = img.shape[1]
            x1, y1, x2, y2 = pose_est_input["bbox_xyxy"]
            x1 = np.max((0, x1))
            y1 = np.max((0, y1))
            x2 = np.min((width - 1, np.max((x1, x2 - 1))))
            y2 = np.min((height - 1, np.max((y1, y2 - 1))))
            # x1y1wh = []
            # if x2 >= x1 and y2 >= y1:
            #     x1y1wh = [x1, y1, x2 - x1, y2 - y1]
            x1y1wh = [x1, y1, x2 - x1, y2 - y1]
            # if len(x1y1wh) != 0:
            center, scale = self._box2cs(x1y1wh[:4])
            pose_est_input["center"] = center
            pose_est_input["scale"] = scale

            db.append(pose_est_input)
        return db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
