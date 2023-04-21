import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime

import diffuserSR3.util as Utils


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args):
    phase = args.phase
    opt_path = args.config
    
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # set log directory
    if args.debug:
        opt['name'] = f'debug_{opt["name"]}'
    experiments_root = os.path.join('experiments', f'{opt["name"]}_{ get_timestamp()}')
    
    opt['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        opt['path'][key] = os.path.join(experiments_root, path)
        mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase


    # debug
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3


    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, f'{phase}.log')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


import numpy as np
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, normalized_root_mse

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))



def calculate_ssim(img1,img2):
    # Convert RGB images to grayscale
    image1_gray = rgb2gray(img1)
    image2_gray = rgb2gray(img2)

    # Calculate SSIM
    ssim_value = ssim(image1_gray, image2_gray, data_range=image1_gray.max() - image1_gray.min())
    return ssim_value



def evaluate_metrics(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate metrics
    img1=Utils.tensor2img(img1)
    img2=Utils.tensor2img(img2)
    
    psnr = calculate_psnr(img1, img2)
    ssim_value = calculate_ssim(img1, img2)
    
    mse_value = mean_squared_error(img1.flatten(), img2.flatten())
    mae_value = normalized_root_mse(img1.flatten(), img2.flatten())

    # Return results as a dictionary
    metrics = {
        'PSNR': psnr,
        'SSIM': ssim_value,
        'MSE': mse_value,
        'MAE': mae_value,
    }

    return metrics