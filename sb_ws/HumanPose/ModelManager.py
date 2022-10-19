from HumanDetector.YOLOv4.models.models import Darknet
from HumanDetector.YOLOv4.models.models import *

import torch


def load_human_detector_model(cfg, model_weight_path, is_pretrained=False, imgsz=640):
    model = Darknet(cfg, imgsz)
    if is_pretrained:
        try:
            model.load_state_dict(torch.load(model_weight_path)["model"])
        except:
            load_darknet_weights(model, model_weight_path)
    return model
