from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import logging

from . import util as Util


class SR3Dataset(Dataset):
    """
    Custom dataset class for super-resolution tasks, inheriting from PyTorch's Dataset class.
    """
    def __init__(self, dataroot, l_resolution=16, r_resolution=128, split='train', data_len=-1):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split

        self.sr_path = Util.get_paths_from_images(f'{dataroot}/sr_{l_resolution}_{r_resolution}')
        self.hr_path = Util.get_paths_from_images(f'{dataroot}/hr_{r_resolution}')
        self.dataset_len = len(self.hr_path)
        self.data_len = min(self.data_len, self.dataset_len) if self.data_len > 0 else self.dataset_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))
        return (torch.cat([img_SR, img_HR]), os.path.splitext(os.path.basename(self.hr_path[index]))[0])

def create_dataset(dataset_opt, phase):
    """
    Wrapper function for creating an instance of the SR3Dataset class.
    """
    return SR3Dataset(dataroot=dataset_opt['train_dataroot'] if phase == "train" else dataset_opt['test_dataroot'],
                      l_resolution=dataset_opt['l_resolution'],
                      r_resolution=dataset_opt['r_resolution'],
                      split=phase,
                      data_len=dataset_opt['train_data_len'] if phase == "train" else dataset_opt['test_data_len'])

def create_dataloader(dataset, dataset_opt, phase):
    """
    Function for creating a PyTorch DataLoader for the given dataset and phase (train or test).
    """
    
    if phase in ['train', 'test']:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'] if phase == "train" else 1,
            shuffle=dataset_opt['use_shuffle'] if phase == "train" else False,
            num_workers=dataset_opt['num_workers'] if phase == "train" else 1,
            pin_memory=False)
    else:
        raise NotImplementedError(f'Dataloader [{phase}] is not found.')
    
def setup_data_loaders(opt):
    """
    Function for setting up data loaders for train and test/validation phases based on the provided options.
    """
    logger = logging.getLogger('base')

    if opt['phase'] == 'train':
        train_set = create_dataset(opt['data'], 'train')
        train_loader = create_dataloader(train_set, opt['training'], 'train')
        val_set = create_dataset(opt['data'], 'test')
        val_loader = create_dataloader(val_set, opt['training'], 'test')
        logger.info('Initial Dataset Finished')
        return train_loader, val_loader
    elif opt['phase'] == 'test':
        val_set = create_dataset(opt['data'], 'test')
        val_loader = create_dataloader(val_set, opt['training'], 'test')
        logger.info('Initial Dataset Finished')
        return val_loader
    else:
        raise ValueError(f"Invalid phase: {opt['phase']}")
