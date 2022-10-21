#%%
from pyexpat import model
import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import deepgaze_pytorch
from logging import root
from skimage import color
from cmath import log
from hmac import trans_36
import numpy as np
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
from scipy.misc import face

a = np.exp(100)
b = np.log(a)
print(a, b)


image = face()
image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to("cpu")

print(type(image_tensor), image_tensor.dim(), image_tensor.size())


### model sequence check ###

# you can use DeepGazeI or DeepGazeIIE
# model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to("cpu")
# print(model)

#########CB num check #####

# train_path = "./data_sample"  # on local

# dataset = datasets.ImageFolder(root=train_path, transform=transforms.ToTensor())
# array_cb = np.load(
#     "/Users/krc/Documents/retail/retail_gh/deepgaze_sample/centerbias_mit1003.npy"
# )
# df_cb = pd.DataFrame(array_cb)
# print(df_cb[300:700].describe())

############# means and std ###########
# mean = 0.0
# meansq = 0.0
# count = 0

# print(dataset[0][0].shape)


# for index, data in enumerate(dataset):
#     mean = data[1][i].sum()
#     meansq = meansq + (data[1][i] ** 2).sum()
#     count += np.prod(data[1][i].shape)

# total_mean = mean / count
# total_var = (meansq / count) - (total_mean**2)
# total_std = torch.sqrt(total_var)
# print(i)
# print("mean: " + str(total_mean))
# print("std: " + str(total_std))
