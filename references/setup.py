from genericpath import exists
import os

try:
    import cv2
    from PIL import Image
    import torchvision.transforms.functional as TF
    import visualpriors
    import subprocess
except ImportError:
    os.system('pip install required packages')



# Download a test image
subprocess.run("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)

# Load image and rescale/resize to [-1,1] and 3x256x256
image = Image.open('test.png')
x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
x = x.unsqueeze_(0)

# Transform to normals feature
representation = visualpriors.representation_transform(x, 'normal', device='cpu')

# Transform to normals feature and then visualize the readout
pred = visualpriors.feature_readout(x, 'normal', device='cpu')

# Save it
TF.to_pil_image(pred[0] / 2. + 0.5).save('test_normals_readout.png')

# Citation
'''@inproceedings{midLevelReps2018,
 title={Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies.},
 author={Alexander Sax and Bradley Emi and Amir R. Zamir and Leonidas J. Guibas and Silvio Savarese and Jitendra Malik},
 year={2018},
}'''