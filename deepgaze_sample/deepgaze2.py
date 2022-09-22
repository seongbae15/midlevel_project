#%%
from logging import root
import os
from cmath import log
from hmac import trans_36
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import deepgaze_pytorch

DEVICE = 'cpu'

# image = face() #racoon face image load 
train_path = '../data_sample/'

def deepgaze_2(train_path):
    # you can use DeepGazeI or DeepGazeIIE
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
    #image resize
    resize_trans = transforms.Compose([transforms.Resize((800,1024)),
                                    transforms.ToTensor()  ])
    resize_train = torchvision.datasets.ImageFolder(root = train_path, transform=resize_trans)
    image = resize_train[0][0]
    print('loaded shape:', image.shape)

    image_unsq = image.unsqueeze(dim=0)
    print('unsq shape:', image_unsq.shape, type(image_unsq))
    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.

    #centerbias_template = np.load('centerbias_mit1003.npy')
    centerbias_template = np.zeros((800,1024))

    # rescale to match image size
    centerbias = zoom(centerbias_template, (image.shape[1]/centerbias_template.shape[0], image.shape[2]/centerbias_template.shape[1]), order=0, mode='nearest')
    # renormalize log density
    centerbias -= logsumexp(centerbias)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_unsq, centerbias_tensor)
    print(log_density_prediction.shape)
    f, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
    axs[0].imshow(torch.transpose(image, 2, 0).transpose(0,1))
    axs[1].matshow(np.exp(log_density_prediction.detach().cpu().numpy()[0,0]), alpha=0.5, cmap=plt.cm.RdBu)
    axs[1].imshow(torch.transpose(image, 2, 0).transpose(0,1), alpha=0.4)
    axs[1].axis('off')
    axs[2].matshow(np.exp(log_density_prediction.detach().cpu().numpy()[0,0]), cmap=plt.cm.RdBu)
    axs[2].axis('off')

# %%