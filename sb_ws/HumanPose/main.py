from DataManager import load_test_input_data_det, load_test_input_data_pose_est
from ModelManager import (
    load_human_detector_model,
    load_pose_estimator_model,
    load_human_detection_pretrain_weight,
    load_pose_estimation_pretrain_weight,
    detect_human,
    estimate_pose,
)
from utils import *
from CONFIG import *
from KeypointsEstimator.SimpleBaseline.lib.utils.utils import create_logger

import cv2
import numpy as np


def release():
    print("Release")


def main():
    # init Human detection path
    input_data_path = "data/test_sample"
    det_model_cfg_path = convert_abs_path("HumanDetector/YOLOv4/cfg/yolov4.cfg")
    det_model_weight_path = convert_abs_path("HumanDetector/YOLOv4/weights/yolov4.weights")

    # init pose estimation path
    pose_est_model_cfg_path = convert_abs_path(
        "KeypointsEstimator/SimpleBaseline/experiments/coco/resnet152/256x192_d256x3_adam_lr1e-3.yaml"
    )
    pose_est_model_weight_path = convert_abs_path(POSE_CFG.TEST.MODEL_FILE)
    logger, final_output_dir, tb_log_dir = create_logger(config, pose_est_model_cfg_path, "valid")

    # Load test data for human detection
    input_dataset_det = load_test_input_data_det(input_data_path)
    print(f"-----Complete input data : {len(input_dataset_det)} ea init work-----")

    # Load model for human detection
    human_det_model = load_human_detector_model(det_model_cfg_path)
    print(f"-----Complete human detection model init work-----\n {human_det_model}")

    # Load Model for pose estimation
    pose_est_model = load_pose_estimator_model(POSE_CFG, is_train=False)
    print(f"-----Complete pose estimation model init work-----\n {pose_est_model}")

    human_det_model = load_human_detection_pretrain_weight(human_det_model, det_model_weight_path)
    print(f"-----Complete Update Weights human detection model -----\n {human_det_model}")

    logger.info("=> loading model from {}".format(pose_est_model_weight_path))
    pose_est_model = load_pose_estimation_pretrain_weight(pose_est_model, pose_est_model_weight_path)
    print(f"-----Complete Update pose estimation model-----\n {pose_est_model}")

    # Huamn Detection
    det_result = detect_human(input_dataset_det, human_det_model, classes=0)
    print(f"-----Complete Detection Human: BBox-----")

    pose_est_ds, pose_est_dataloader = load_test_input_data_pose_est(det_result["data"])
    print("-----Complete to create pose estimation dataloader-----")

    # Pose Keypoint Estimation
    outputs = estimate_pose(pose_est_dataloader, pose_est_ds, pose_est_model)
    draw_keypoint(outputs)
    print("-----Complete to run pose estimation-----")


if __name__ == "__main__":
    main()
