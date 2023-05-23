import logging
import torch
import os
from tqdm import tqdm
import wandb

from accelerate import Accelerator
from diffusers.optimization import get_constant_schedule_with_warmup

import torch.nn.functional as F


import diffuserSR3.unet2D as Unet
import diffuserSR3.pipeline as Pipeline
import diffuserSR3.metrics as Metrics
import diffuserSR3.logger as Logger
import diffuserSR3.util as Utils




def setup_model(opt):
    logger = logging.getLogger('base')
    model = Unet.create_unet2D(opt['model'])
    logger.info(f'Model created:\n{model}')
    num_params = sum(p.numel() for p in model.parameters())  # get the total number of parameters
    logger.info(f'Model nnumber of parameters:{num_params}')
    return model

def setup_accelerator(opt):
    logger = logging.getLogger('base')
    accelerator = Accelerator(
                mixed_precision=opt['accelerator']['mixed_presition'],
                gradient_accumulation_steps=opt['accelerator']['gradient_accumulation_steps'],
                log_with="wandb",
                project_dir=opt['path']['accelerator']
                )        
    accelerator.init_trackers(f'{opt["name"]}')
    logger.info('Accelerator created.')
    return accelerator

def setup_optimization(model, train_loader, opt):
    logger = logging.getLogger('base')
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['training']['learning_rate'])
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=opt['training']['lr_warmup_steps'],
    )
    logger.info('Learning rate and optimizer created')
    return optimizer, lr_scheduler


def train_and_evaluate(model, optimizer, lr_scheduler, accelerator, train_loader, val_loader, opt, logger):
    global_step = 0
    logger.info('Starting training:')
    start_step=0
    device=model.device
    if opt['training']['resume_path'] is not "":
        logger.info(f'Loading pretrained model from {opt["training"]["resume_path"]}')
        start_step = load_checkpoint(model, optimizer, opt['training']['resume_path'])
    progress_bar = tqdm(total=opt["training"]['steps'], disable=not accelerator.is_local_main_process)

    for epoch in range(int((opt["training"]['steps']-start_step)/len(train_loader))+1):
        
        scheduler = Pipeline.create_SR3scheduler(opt['scheduler'], 'train')

        for step, (batch, index) in enumerate(train_loader):
            progress_bar.set_description(f"Epoch {global_step}")
            train_step(model, optimizer, lr_scheduler, accelerator, batch, epoch, global_step, scheduler, progress_bar, logger, device)
            global_step += 1
            
            if (global_step+1) % opt['training']['val_freq'] == 0:
                evaluate(model, accelerator, val_loader, opt, logger, epoch, global_step)
                save_checkpoint(model, optimizer, epoch, opt['path']['checkpoint'])


def train_step(model, optimizer, lr_scheduler, accelerator, batch, epoch, global_step, scheduler, progress_bar, logger, device):
    noise = torch.randn(batch.shape).to(device)
    bs = batch.shape[0]
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()
    noisy_images = scheduler.add_noise(batch, timesteps=timesteps, noise=noise)

    with accelerator.accumulate(model):
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.l1_loss(noise_pred, noise[:, 3:, :, :])
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "epoch": epoch}
    logger.info(logs)
    progress_bar.set_postfix(**logs)
    accelerator.log(logs, step=global_step)
    del batch, noise, noisy_images, noise_pred


def evaluate(model, accelerator, val_loader, opt, logger, epoch, global_step):
    logger.info(f'Starting validation at epoch: {epoch}')
    if accelerator.is_main_process:
        sampler = Pipeline.create_SR3Sampler(accelerator.unwrap_model(model), opt['scheduler'])
        metrics_sum = {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
        for count, (images, index) in enumerate(val_loader):
            low_res_image, high_res_image, samples = evaluate_image(sampler, images)
            metrics = Metrics.evaluate_metrics(high_res_image, samples[-1])
            Logger.log_metrics_image(logger, index, metrics, metrics_sum)
            accelerator.log(metrics, step=global_step)
            Utils.save_image(samples[-1], f'{opt["path"]["results"]}/validation_{index[0]}_{epoch}.png')
            Logger.log_images_to_wandb(accelerator, low_res_image, samples, high_res_image, index)
            del low_res_image, high_res_image, samples

        Logger.log_mean_metrics(logger, epoch, metrics_sum, count, accelerator, global_step)
    logger.info(f'Saving model for epoch {epoch}')


def evaluate_image(sampler, images):
    low_res_image = images[:, :3, :, :]
    high_res_image = images[:, 3:, :, :]
    samples = sampler.sample_high_res(low_res_image)
    return low_res_image, high_res_image, samples

def test(model,accelerator, val_loader, opt, logger):
    try:
        logger.info(f'Loading pretrained model from {opt["training"]["resume_path"]}')
        load_checkpoint(model, None, opt['training']['resume_path'])  
    except FileNotFoundError:
        raise FileNotFoundError(f'Pre trained model not found in {opt["training"]["resume_path"]}') 
    
    sampler = Pipeline.create_SR3Sampler((accelerator.unwrap_model(model)), opt['scheduler'])
    metrics_sum = {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
    test_table_data = []
    for count, (images, index) in enumerate(val_loader):
        low_res_image, high_res_image, samples = evaluate_image(sampler, images)
        metrics = Metrics.evaluate_metrics(high_res_image, samples[-1])
        Logger.log_metrics_image(logger, index, metrics, metrics_sum)
        Utils.save_image(samples[-1], f'{opt["path"]["results"]}/test_{index[0]}_sr.png')
        Utils.tensors_to_grid_image(samples,low_res_image,f'{opt["path"]["results"]}/test_{index[0]}_process.png')

        test_table_data.append([index, wandb.Image(Utils.tensor2img(low_res_image)), wandb.Image(Utils.tensor2img(
        samples[-1])), wandb.Image(Utils.tensor2img(high_res_image)), metrics['PSNR'], metrics['SSIM'],metrics['MSE'],metrics['MAE']])
        
        del low_res_image, high_res_image, samples

    Logger.log_mean_metrics(logger, None, metrics_sum, count, accelerator)    
    Logger.log_test_table_wandb(accelerator, test_table_data)

    logger.info(f'Testing finished')

def save_checkpoint(model, optimizer, globap_step, save_path):
    """
    Save model and optimizer state dictionaries to a checkpoint file.
    
    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        save_path (str): The directory path to save the checkpoint.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    checkpoint_file = os.path.join(save_path, f'checkpoint_step_{globap_step}.pth')
    torch.save({
        'step': globap_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_file)

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state dictionaries from a checkpoint file.
    
    Args:
        model (nn.Module): The model to load.
        optimizer (optim.Optimizer): The optimizer to load.
        checkpoint_path (str): The path to the checkpoint file.
    
    Returns:
        int: The epoch number from the loaded checkpoint.
    """

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        pass # Add debug logg for optimizer loded
    step = checkpoint['step']
    return step