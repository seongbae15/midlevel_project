from KeypointsEstimator.SimpleBaseline.lib.core.config import config

DET_IMG_SIZE = 640
DET_CONF_THRES = 0.4
DET_IOU_THRES = 0.5
DET_AGNOSTIC = False
DET_OUT_DIR = "output/HumanDetection"
DET_NAME_DIR = "HumanDetector/YOLOv4/data/coco.names"
DET_SAVE_TXT = True
DET_SAVE_IMG = True
DET_VIEW_IMG = False

POSE_CFG = config

POSE_IMG_HEIGHT = 256
POSE_IMG_WIDTH = 256

KEY_POINT_CIRCLE_RADIUS = 6
KEY_POINT_CIRCLE_COLOR = (0, 0, 255)

POSE_PAIRS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

POSE_IMG_SAVE_DIR = "output/KeypointsEstimation"
