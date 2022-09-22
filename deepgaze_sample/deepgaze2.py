#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from torchvision.transforms.functional import to_pil_image

import deepgaze_pytorch

DEVICE = 'cpu'

# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

image = face() #racoon face image load 

# load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
# you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
# alternatively, you can use a uniform centerbias via `centerbias_template = np.zeros((1024, 1024))`.

# centerbias_template = np.load('centerbias_mit1003.npy')
centerbias_template = np.zeros((1024, 768))

# rescale to match image size
centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
# renormalize log density
centerbias -= logsumexp(centerbias)

image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
print(image_tensor.size())
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
print(centerbias_tensor.size())

log_density_prediction = model(image_tensor, centerbias_tensor)
f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
axs[0].imshow(image)
axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
axs[1].set_axis_off()

# %%
