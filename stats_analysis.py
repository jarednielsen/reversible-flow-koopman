from glob import glob
from skimage import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from skimage.measure import compare_ssim as ssim
import cv2

import utils

np.set_printoptions(precision=3)


def get_images(pattern='../Adversarial_Video_Generation/images_final/dissolving_5/*'):
    files = sorted(glob(pattern))
    imgs = [ Image.open(f) for f in files]
    return imgs

def to_tensor(images):
    compose = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    tensors = [compose(img) for img in images]

    return tensors

def optical_flow(images, n_out=5):
    pass



# imgs = get_images()
# tensors = to_tensor(imgs)
# save_image(tensors, '../Adversarial_Video_Generation/images_final/dissolving_5_grid.png', padding=1, pad_value=tensors[0].max())

for i in range(1, 14):
    gen_imgs = get_images('../Adversarial_Video_Generation/Save/Images/bouncing_mnist/Tests/Step_{}000/7/gen_*'.format(i))
    gt_imgs = get_images('../Adversarial_Video_Generation/Save/Images/bouncing_mnist/Tests/Step_{}000/7/gt_*'.format(i))

    print(i)
    psnr_scores = np.array([utils.sklearn_psnr(gen, gt) for gen, gt in zip(gen_imgs, gt_imgs)])
    ssim_scores = np.array([utils.sklearn_ssim(gen, gt) for gen, gt in zip(gen_imgs, gt_imgs)])
    print("PSNR: {}".format(psnr_scores))
    print("SSIM: {}".format(ssim_scores))

# prev_imgs = get_images('../Adversarial_Video_Generation/Save/Images/bouncing_mnist/Tests/Step_13000/7/input_*')
# gt_imgs = get_images('../Adversarial_Video_Generation/Save/Images/bouncing_mnist/Tests/Step_13000/7/gt_*')
# gen_imgs = get_images('../Adversarial_Video_Generation/Save/Images/bouncing_mnist/Tests/Step_13000/7/gen_*')

# tensors = to_tensor(gen_imgs)
# save_image(tensors, '../Adversarial_Video_Generation/images_final/bouncing-mnist-adversarial.png', padding=1, pad_value=tensors[0].max())

