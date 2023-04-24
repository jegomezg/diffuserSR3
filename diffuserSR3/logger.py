import os
import json
from collections import OrderedDict
from datetime import datetime
import logging
import numpy as np
import wandb

import diffuserSR3.util as Utils


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        [os.makedirs(path, exist_ok=True) for path in paths]


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args):

    phase = args.phase
    opt_path = args.config

    with open(opt_path, 'r') as f:
        json_str = ''.join(line.split('//')[0] + '\n' for line in f)

    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    if args.debug:
        opt['name'] = f'debug_{opt["name"]}'
    experiments_root = os.path.join(
        'experiments', f'{opt["name"]}_{get_timestamp()}')

    opt['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        opt['path'][key] = os.path.join(experiments_root, path)
        mkdirs(opt['path'][key])

    opt['phase'] = phase

    if 'debug' in opt['name']:
        opt.update({
            'train': {'val_freq': 2, 'print_freq': 2, 'save_checkpoint_freq': 3},
            'datasets': {'train': {'batch_size': 2, 'data_len': 6}, 'val': {'data_len': 3}},
            'model': {
                'beta_schedule': {
                    'train': {'n_timestep': 10},
                    'val': {'n_timestep': 10}
                }
            }
        })

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        return NoneDict({key: dict_to_nonedict(sub_opt) for key, sub_opt in opt.items()})
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        indent = ' ' * (indent_l * 2)
        msg += f"{indent}{k}: {dict2str(v, indent_l + 1) if isinstance(v, dict) else v}\n"
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
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


def log_metrics_image(logger, index, metrics, metrics_sum):
    logger.info(f'Metrics for image {index}:')
    logger.info(metrics)
    for key in metrics.keys():
        metrics_sum[key] += metrics[key]


def log_mean_metrics(logger, epoch, metrics_sum, count, accelerator, global_step=None):
    metrics_mean = {}
    for key in metrics_sum.keys():
        metrics_mean[key] = metrics_sum[key] / (count + 1)
    if epoch is not None:
        logger.info(f'Mean metrics for epoch {epoch}:')
    else:
        logger.info(f'Mean metrics for test')
    logger.info(metrics_mean)
    accelerator.log(metrics_mean, step=global_step)


def log_images_to_wandb(accelerator, low_res_image, samples, high_res_image, index):
    wandb_tracker = accelerator.get_tracker("wandb")
    wandb_tracker.log_images({f"validation_{index[0]}": [np.concatenate([Utils.tensor2img(low_res_image),
                                                                        Utils.tensor2img(
                                                                            samples[-1]),
                                                                        Utils.tensor2img(high_res_image)], axis=1)]})


def log_test_table_wandb(accelerator, data):
    wandb_tracker = accelerator.get_tracker("wandb")
    columns = ["id", "low_res", "super_res",
               "high_res", "PSNR", "SSIM", "MSE", "MAE"]
    wandb_tracker.log_table(table_name = 'test', columns=columns,data=data)
