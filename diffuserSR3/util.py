import os
import torch
import torchvision
import random
import numpy as np

from accelerate import Accelerator


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


def create_accelerator(opt,path):
    accelerator = Accelerator(
        mixed_precision=opt['mixed_presition'],
        gradient_accumulation_steps=opt['gradient_accumulation_steps'],
        log_with="wandb",
        project_dir=path,
    )   
    return accelerator

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_image(image,path):
    image = Image.fromarray(tensor2img(image))
    image.save(path)
    


import torch
import torchvision.transforms as transforms
from PIL import Image

def tensors_to_grid_image(tensors_list,low_res, output_file):
    assert len(tensors_list) == 8, "The list should contain exactly 10 tensors"
    
    
    # Convert tensors to PIL images
    images = [Image.fromarray(tensor2img(tensor)) for tensor in tensors_list]
    low_res = Image.fromarray(tensor2img(low_res))
        
    # Create an empty grid
    grid_size = 3
    grid_image = Image.new('RGB', (512 * grid_size, 512 * grid_size))
    
    # Arrange images in the grid
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx == 0:
                grid_image.paste(low_res, (j * 512, i * 512))
            else:
                grid_image.paste(images[idx-1], (j * 512, i * 512))
            idx += 1
    
    # Save the grid image as a PNG file
    grid_image.save(output_file)
    
    