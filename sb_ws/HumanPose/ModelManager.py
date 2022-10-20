from HumanDetector.YOLOv4.models.models import Darknet
from HumanDetector.YOLOv4.models.models import *
from HumanDetector.YOLOv4.utils.torch_utils import time_synchronized
from HumanDetector.YOLOv4.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from HumanDetector.YOLOv4.utils.plots import plot_one_box

from KeypointsEstimator.SimpleBaseline.lib.models.pose_resnet import get_pose_net
from KeypointsEstimator.SimpleBaseline.lib.core.inference import get_final_preds

from utils import *
from CONFIG import *

import torch
from numpy import random
import time
import cv2
import shutil
import platform


def load_human_detector_model(cfg, imgsz=DET_IMG_SIZE):
    model = Darknet(cfg, imgsz)
    return model


def load_pose_estimator_model(cfg, is_train=False):
    model = get_pose_net(cfg, is_train)
    return model


def load_human_detection_pretrain_weight(human_det_model, det_model_weight_path):
    if det_model_weight_path:
        try:
            human_det_model.load_state_dict(torch.load(det_model_weight_path)["model"])
        except:
            load_darknet_weights(human_det_model, det_model_weight_path)
    return human_det_model


def load_pose_estimation_pretrain_weight(pose_est_model, pose_est_model_weight_path):
    if pose_est_model_weight_path:
        pose_est_model.load_state_dict(torch.load(pose_est_model_weight_path))
    return pose_est_model


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, "r") as f:
        names = f.read().split("\n")
    return list(filter(None, names))  # filter removes empty strings (such as last line)


@torch.no_grad()
def detect_human(input_dataset, human_det_model, is_augment=False, classes=None):
    print("-----Start Detection Human: BBox-----")

    out = convert_abs_path(DET_OUT_DIR)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"  # half precision only

    human_det_model.to(device).eval()

    if half:
        human_det_model.half()  # to FP16

    # Get names and colors
    names = convert_abs_path(DET_NAME_DIR)
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, DET_IMG_SIZE, DET_IMG_SIZE), device=device)  # init img
    _ = human_det_model(img.half() if half else img) if device != "cpu" else None  # run once

    # Dictionary to Save
    result_det = {}
    det_datas = []

    for path, img, im0s, vid_cap in input_dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = human_det_model(img, augment=is_augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, DET_CONF_THRES, DET_IOU_THRES, classes=classes, agnostic=DET_AGNOSTIC)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, "", im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ("_%g" % input_dataset.frame if input_dataset.mode == "video" else "")
            s += "%gx%g " % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += "%g %ss, " % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if DET_SAVE_TXT:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * 5 + "\n") % (cls, *xywh))  # label format

                    if DET_SAVE_IMG or DET_VIEW_IMG:  # Add bbox to image
                        label = "%s %.2f" % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    temp = []
                    for xy in xyxy:
                        temp.append(int(xy.item()))
                    det_element = {"file_path": p, "bbox_xyxy": temp}
                    det_datas.append(det_element)

            # Print time (inference + NMS)
            print("%sDone. (%.3fs)" % (s, t2 - t1))

            # Stream results
            if DET_VIEW_IMG:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if DET_SAVE_IMG:
                if input_dataset.mode == "images":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = "mp4v"  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if DET_SAVE_TXT or DET_SAVE_IMG:
        print("Results saved to %s" % Path(out))
    print("Done. (%.3fs)" % (time.time() - t0))
    result_det["data"] = det_datas
    return result_det


@torch.no_grad()
def estimate_pose(input_datloader, pose_est_ds, pose_est_model):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pose_est_model.to(device).eval()

    num_samples = len(pose_est_ds)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 5))

    outputs = []
    idx = 0
    for input, db in input_datloader:
        output_total = {}
        x = input.to(device)
        output = pose_est_model(x)

        num_images = input.size(0)

        c = db["center"].numpy()
        s = db["scale"].numpy()

        preds, maxvals = get_final_preds(POSE_CFG, output.clone().cpu().numpy(), c, s)
        all_preds[idx : idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx : idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx : idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx : idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx : idx + num_images, 4] = np.prod(s * 200, 1)
        idx += 1

        xyxy_temp = []
        for xy in db["bbox_xyxy"]:
            xyxy_temp.append(int(xy.item()))
        output_total = {
            "file_path": db["file_path"],
            "pose_keys": preds[:, :, 0:2],
            "max_p": maxvals,
            "bbox_xyxy": xyxy_temp,
            "center": [c[:, 0:2]],
            "scale": [s[:, 0:2]],
        }
        outputs.append(output_total)

    return outputs
