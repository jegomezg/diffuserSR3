import os
import random
import numpy as np
import torch
import torchvision
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)

def get_paths_from_images(path):
    assert os.path.isdir(path), f'{path} is not a valid directory'
    return sorted(
        os.path.join(dirpath, fname)
        for dirpath, _, fnames in sorted(os.walk(path))
        for fname in sorted(fnames)
        if is_image_file(fname)
    )

def augment(img_list, hflip=True, rot=True, split='val'):
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def transform2numpy(img):
    img = np.array(img).astype(np.float32) / 255.
    return np.expand_dims(img, 2) if img.ndim == 2 else img[:, :, :3]

def transform2tensor(img, min_max=(0, 1)):
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img * (min_max[1] - min_max[0]) + min_max[0]

totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()

def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    return [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    img_np = np.transpose(((tensor - min_max[0]) / (min_max[1] - min_max[0])).numpy(), (1, 2, 0))
    return (img_np * 255.0).round().astype(out_type) if out_type == np.uint8 else img_np

def save_image(image, path):
    Image.fromarray(tensor2img(image)).save(path)


def tensors_to_grid_image(tensors_list,low_res, output_file):
    assert len(tensors_list) == 8, "The list should contain exactly 10 tensors"
    
    images = [Image.fromarray(tensor2img(tensor)) for tensor in tensors_list]
    low_res = Image.fromarray(tensor2img(low_res))
        
    grid_size = 3
    grid_image = Image.new('RGB', (512 * grid_size, 512 * grid_size))
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx == 0:
                grid_image.paste(low_res, (j * 512, i * 512))
            else:
                grid_image.paste(images[idx-1], (j * 512, i * 512))
            idx += 1
    grid_image.save(output_file)
    
    