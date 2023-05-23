import argparse
import logging
import wandb
import numpy as np
import torch

import diffuserSR3.dataset as Data
import diffuserSR3.logger as Logger
import diffuserSR3.trainner as Train
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from transformers.debug_utils import DebugUnderflowOverflow


def main():
    args = parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    Logger.setup_logger(None, opt['path']['log'],'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(f'Configuration:\n{Logger.dict2str(opt)}')

    model = Train.setup_model(opt)

    if opt['phase'] == 'train':
        train_loader, val_loader = Data.setup_data_loaders(opt)
        optimizer, lr_scheduler = Train.setup_optimization(model, train_loader, opt)
        accelerator = Train.setup_accelerator(opt)
        model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
        logger.info('Model and accelerator initialized')
        
    elif opt['phase'] == 'test':
        val_loader = Data.setup_data_loaders(opt)
        accelerator = Train.setup_accelerator(opt)
        model= accelerator.prepare(model)
        logger.info('Model and accelerator initialized')

    device = torch.device(f"cuda:0")
    model.to(device)
    #model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    debug_overflow = DebugUnderflowOverflow(model)
    
    if opt['phase'] == 'train':
        Train.train_and_evaluate(model, optimizer, lr_scheduler, accelerator, train_loader, val_loader, opt, logger)
    if opt['phase'] == 'test':
        Train.test(model, accelerator, val_loader, opt, logger)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    return parser.parse_args()



















if __name__ == "__main__":
    main()


