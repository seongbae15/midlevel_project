#%%
from logging import root
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


warnings.filterwarnings("ignore")
DEVICE = "cpu"

train_path = "../data_evm"  # on notebook : ../data_sample


# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
print(model.eval())

# image transform
resize_trans = transforms.Compose(
    [
        transforms.Resize((900, 900)),
        transforms.ToTensor(),
    ]
)

# with folder:
dataloader = torchvision.datasets.ImageFolder(root=train_path, transform=resize_trans)

ig_base = []
ls_ig = []
for idx, pics in enumerate(dataloader):
    image_rgb = pics[0]
    image_orig = image_rgb.clone().detach()
    # image_tf = tfu.gaussian_blur(image_rgb, [25, 25])
    # image_tf = tfu.adjust_brightness(image_rgb, 1.2)
    image_tf = tfu.adjust_gamma(image_rgb, 2)

    # print(len(pics), type(pics), image_rgb.size(), image_rgb.dim())
    image = color.rgb2lab(rgb=image_tf.T, channel_axis=-1)  # color space to lab

    # result = 800, 1200, 3 -> 바꿔줘야
    image = torch.tensor(image.transpose(2, 1, 0)).to(DEVICE)

    # image_rgb = image_rgb.unsqueeze(dim=0)
    image_unsq = image.unsqueeze(dim=0)

    # centerbias_template = np.zeros((900, 900))
    centerbias_template = np.ones((900, 900)) * -16
    # centerbias_template = np.load("centerbias_mit1003.npy")

    # rescale to match image size
    centerbias = zoom(
        centerbias_template,
        (
            image_unsq.shape[2] / centerbias_template.shape[0],
            image_unsq.shape[3] / centerbias_template.shape[1],
        ),
        order=5,
        mode="nearest",
    )

    # centerbias_rgb = zoom(
    #     centerbias_template,
    #     (
    #         image_rgb.shape[2] / centerbias_template.shape[0],
    #         image_rgb.shape[3] / centerbias_template.shape[1],
    #     ),
    #     order=5,
    #     mode="nearest",
    # )
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    # log_density_prediction_base = model(image_rgb, centerbias_rgb)
    log_density_prediction = model(image_unsq, centerbias_tensor)
    print(f"image number {idx} predicition is done")
    base_prob = np.load(f"../data_result/{idx}_prob.npy")
    eval_prob = log_density_prediction.squeeze().detach().numpy()
    ig_result = (base_prob - eval_prob).sum() / (
        image_unsq.shape[2] * image_unsq.shape[3]
    )
    ls_ig.append(ig_result)
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 12))
    axs[0].imshow(torch.transpose(image_orig, 2, 0).transpose(0, 1))
    axs[0].axis("off")
    axs[1].matshow(
        np.exp(log_density_prediction.detach().cpu().numpy()[0, 0]),
        alpha=0.5,
        cmap=plt.cm.tab20b,
    )
    axs[1].imshow(torch.transpose(image_orig, 2, 0).transpose(0, 1), alpha=0.4)
    axs[1].axis("off")
    axs[2].matshow(
        np.exp(log_density_prediction.detach().cpu().numpy()[0, 0]), cmap=plt.cm.tab20b
    )
    axs[2].set_xlabel(ig_result)

    plt.savefig(f"../data_result/{idx}_result.png")
print(f"total IG = {np.array(ls_ig).mean()}")

# %%
