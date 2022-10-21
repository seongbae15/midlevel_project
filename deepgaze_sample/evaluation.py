#%%
from logging import root
from cv2 import DFT_COMPLEX_INPUT
from pandas import DataFrame
from skimage import color
from cmath import log
from hmac import trans_36
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tfu
import warnings
import deepgaze_pytorch
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

warnings.filterwarnings("ignore")
DEVICE = "cpu"

# image = face() #racoon face image load
train_path = "./data_evm"


# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

# image transform
resize_trans = transforms.Compose(
    [
        transforms.Resize((900, 900)),
        transforms.ToTensor(),
    ]
)

# with folder:
dataloader = torchvision.datasets.ImageFolder(root=train_path, transform=resize_trans)


pics = dataloader[0]
image_rgb = pics[0]


# print(len(pics), type(pics), image_rgb.size(), image_rgb.dim())
image = color.rgb2lab(rgb=image_rgb.T, channel_axis=-1)  # color space to lab

# result = 800, 1200, 3 -> 바꿔줘야
image = torch.tensor(image.transpose(2, 1, 0)).to(DEVICE)

image_rgb = image_rgb.unsqueeze(dim=0)
image_unsq = image.unsqueeze(dim=0)

centerbias_template = np.load("deepgaze_sample/centerbias_mit1003.npy")


# rescale to match image size
centerbias = zoom(
    centerbias_template,
    (
        image_unsq.shape[2] / centerbias_template.shape[0],
        image_unsq.shape[3] / centerbias_template.shape[1],
    ),
    order=0,
    mode="nearest",
)
# renormalize log density
centerbias -= logsumexp(centerbias)
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)


log_density_prediction_base = model(image_rgb, centerbias_tensor)
log_density_prediction_eval = model(image_unsq, centerbias_tensor)

base_torch = log_density_prediction_base.squeeze().detach().numpy()
np.save("./base_probability.npy", base_torch)
eval_torch = log_density_prediction_eval.squeeze().detach().numpy()
np_dif = base_torch - eval_torch

print((image_rgb).shape[:], type(np_dif))

print(np_dif.sum() / (image_rgb.shape[2] * image_rgb.shape[3]))


# log_density_prediction_eval

# f, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 12))
# axs[0].matshow(
#     np.exp(log_density_prediction_base.detach().cpu().numpy()[0, 0]),
#     alpha=0.5,
#     cmap=plt.cm.RdBu,
# )
# axs[1].matshow(
#     np.exp(log_density_prediction_eval.detach().cpu().numpy()[0, 0]),
#     alpha=0.5,
#     cmap=plt.cm.RdBu,
# )
# axs[1].imshow(torch.transpose(image_rgb, 2, 0).transpose(0, 1), alpha=0.4)
# axs[1].axis("off")


# df_result = pd.DataFrame(np.exp(log_density_prediction.detach().cpu().numpy()[0, 0]))
# print(df_result.sum().sum())

# %%
