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
from torchvision.transforms.functional import to_pil_image
import warnings
import deepgaze_pytorch
from PIL import Image

warnings.filterwarnings("ignore")
DEVICE = "cpu"

# image = face() #racoon face image load
train_path = "../data_sample"  # on notebook : ../data_sample


# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
# print(model)

# image resize
resize_trans = transforms.Compose(
    [
        transforms.Resize((900, 900)),
        transforms.GaussianBlur(kernel_size=(25, 25), sigma=(1.0, 10.0)),
        transforms.ToTensor(),
    ]
)

# with folder:
dataloader = torchvision.datasets.ImageFolder(root=train_path, transform=resize_trans)


for idx, pics in enumerate(dataloader):
    image_rgb = pics[0]
    # print(len(pics), type(pics), image_rgb.size(), image_rgb.dim())

    # image preprocess

    image = color.rgb2lab(rgb=image_rgb.T, channel_axis=-1)  # color space to lab
    # result = 800, 1200, 3 -> 바꿔줘야
    image = torch.tensor(image.transpose(2, 1, 0)).to(DEVICE)

    image_unsq = image.unsqueeze(dim=0)
    # print(    type(image_unsq),  image_unsq.dim(),  image_unsq.shape, )
    # break
    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.

    centerbias_template = np.load(
        "centerbias_mit1003.npy"
    )  # on vsc ./deepgaze_sample/centerbias_mit1003.npy
    # centerbias_template = np.ones((900, 900)) * (-16)

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
    # centerbias -= logsumexp(centerbias)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_unsq, centerbias_tensor)
    if idx % 10 == 0:
        print(f"number {idx} image is processed")
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 12))
    axs[0].imshow(torch.transpose(image_rgb, 2, 0).transpose(0, 1))
    axs[1].matshow(
        np.exp(log_density_prediction.detach().cpu().numpy()[0, 0]),
        alpha=0.5,
        cmap=plt.cm.RdBu,
    )
    axs[1].imshow(torch.transpose(image_rgb, 2, 0).transpose(0, 1), alpha=0.4)
    axs[1].axis("off")
    axs[2].matshow(
        np.exp(log_density_prediction.detach().cpu().numpy()[0, 0]), cmap=plt.cm.RdBu
    )
    axs[2].axis("off")
    plt.savefig(f"../data_result/{idx}_result.png")

# %%
