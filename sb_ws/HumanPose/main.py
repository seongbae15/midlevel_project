from DataManager import load_test_input_data_det
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


def release():
    print("Release")


def main():
    # init Human detection path
    input_data_path = "data/test_sample"
    det_model_cfg_path = convert_abs_path("HumanDetector/YOLOv4/cfg/yolov4.cfg")
    det_model_weight_path = convert_abs_path("HumanDetector/YOLOv4/weights/yolov4.weights")

    # init pose estimation path
    pose_est_model_cfg_path = convert_abs_path(
        "KeypointsEstimator/SimpleBaseline/experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml"
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
    print(f"-----Complete human detection model init work-----\n {pose_est_model}")

    human_det_model = load_human_detection_pretrain_weight(human_det_model, det_model_weight_path)
    logger.info("=> loading model from {}".format(pose_est_model_weight_path))
    pose_est_model = load_pose_estimation_pretrain_weight(pose_est_model, pose_est_model_weight_path)

    # Huamn Detection
    detect_human(input_dataset_det, human_det_model, classes=0)

    # To Do~~~~
    # 1. Set Input fo Pose Estimation Data
    # 2. Run(Test) Pose Estimation
    # 3. Result Visualize (Adjust Image Size)

    # Set

    # # Pose Keypoint Estimation
    input_dataset_pose = ""
    estimate_pose(input_dataset_pose, pose_est_model)

    # Release
    release()
    pass


if __name__ == "__main__":
    main()
