from torch.utils.data import DataLoader
from torchvision import transforms

from CONFIG import *

from HumanDetector.YOLOv4.utils.dataset import LoadImages


def load_test_input_data_det(path, img_size=DET_IMG_SIZE, auto_size=32):
    intput_dataset = LoadImages(path, img_size=img_size, auto_size=auto_size)
    return intput_dataset
