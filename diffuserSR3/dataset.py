from PIL import Image
import torch
from torch.utils.data import Dataset
import os

from . import util as Util


class SR3ataset(Dataset):
    def __init__(self, dataroot, l_resolution=16, r_resolution=128, split='train', data_len=-1):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split

        self.sr_path = Util.get_paths_from_images(
            f'{dataroot}/sr_{l_resolution}_{r_resolution}')
        self.hr_path = Util.get_paths_from_images(
            f'{dataroot}/hr_{r_resolution}')
        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None

        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        [img_SR, img_HR] = Util.transform_augment(
            [img_SR, img_HR], split=self.split, min_max=(-1, 1))
        return (torch.cat([img_SR,img_HR]),os.path.splitext(os.path.basename(self.hr_path[index]))[0])

def create_dataset(dataset_opt, phase):
    '''create dataset'''
    dataset = SR3ataset(dataroot=dataset_opt['train_dataroot'] if phase=="train" else dataset_opt['test_dataroot'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['train_data_len'] if phase=="train" else dataset_opt['test_data_len'],
                )
    #logger = logging.getLogger('base')
    #logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,dataset_opt['name']))
    return dataset


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=False)
    elif phase == 'test':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False) #Change pinning
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))

