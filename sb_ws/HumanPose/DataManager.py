from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob

from CONFIG import *
from utils import *

from HumanDetector.YOLOv4.utils.dataset import LoadImages
from CustomDataset import CustomPoseEstDataset


pose_est_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def load_test_input_data_det(path, img_size=DET_IMG_SIZE, auto_size=32):
    intput_dataset = LoadImages(path, img_size=img_size, auto_size=auto_size)
    return intput_dataset


def load_test_input_data_pose_est(det_list):
    # Det Ouputs is converted to Pose Est Inputs
    print("-----Det Ouputs is converted to Pose Est Inputs-----")
    pose_est_ds = CustomPoseEstDataset(det_list, transform=pose_est_transform)
    pose_est_dataloader = DataLoader(pose_est_ds)
    return pose_est_ds, pose_est_dataloader
