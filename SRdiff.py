import diffuserSR3.dataset as Data 
import diffuserSR3.pipeline as Pipeline
import diffuserSR3.unet2D as Unet
import diffuserSR3.util as Utils
import diffuserSR3.logger as Logger

from diffusers.optimization import get_cosine_schedule_with_warmup

import logging
import argparse

from tqdm import tqdm
import wandb
import numpy as np

import torch
import torch.nn.functional as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')


    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)

    logger = logging.getLogger('base')
    logger.info('You are running the model with the following configuration')
    logger.info(Logger.dict2str(opt))
    
    # dataset
    if opt['phase'] == 'train' and args.phase != 'val':
        train_set = Data.create_dataset(opt['data'], 'train')
        train_loader = Data.create_dataloader(train_set, opt['trainning'], 'train')
        val_set = Data.create_dataset(opt['data'], opt['phase'])
        val_loader = Data.create_dataloader(val_set, opt['trainning'], 'test')
    elif opt['phase'] == 'test':
        val_set = Data.create_dataset(opt['data'], 'test')
        val_loader = Data.create_dataloader(val_set, opt['trainning'], 'test')
    logger.info('Initial Dataset Finished')

    #Model 
    model=Unet.crreate_unet2D(opt['model'])
    logger.info('Model created')
    logger.info('The architecture of the denoising model is:')
    logger.info(model)
    
    #Accelerator
    accelerator = Utils.create_accelerator(opt['accelerator'],opt['path']['accelerator']) 
    accelerator.init_trackers(f'{opt["name"]}')
    wandb_tracker = accelerator.get_tracker("wandb")

    logger.info('Aceelerator created')
    
    #Learning rate and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['trainning']['learning_rate'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=opt['trainning']['lr_warmup_steps'],
        num_training_steps=(len(train_loader) * opt['trainning']['num_epochs']),
    )
    logger.info('Learning rate and optimizer created')
    
    #Prepare under accelerator
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, lr_scheduler
    )
    logger.info('Model and accelerator initialized')
    
    global_step = 0
    
    #Set device
    device = torch.device(f"cuda:0")
    model.to(device)
    
    if opt['phase'] == 'train':
        logger.info('Starting trainning:')
        for epoch in range(opt["trainning"]['num_epochs']):

            progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            scheduler = Pipeline.create_SR3scheduler(opt['scheduler'],'train')    
            
            for step, (batch,index) in enumerate(train_loader):

                noise = torch.randn(batch.shape).to(device)   
                bs = batch.shape[0]            
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,),device=device).long()  
                noisy_images = scheduler.add_noise(batch,timesteps=timesteps,noise=noise)
                
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise[:,3:,:,:])
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step,"epoch":epoch}
                logger.info(logs)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                del batch, noise, noisy_images, noise_pred
            
            if epoch % opt['trainning']['val_freq'] == 0:
                logger.info(f'Starting validation at epoch: {epoch}')
            # After each epoch you optionally sample some demo images with evaluate() and save the model
                if accelerator.is_main_process:
                    sampler = Pipeline.create_SR3Sampler(accelerator.unwrap_model(model),opt['scheduler'])
                    metrics_sum = {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
                    for count,  (images,index) in enumerate(val_loader):
                        logger.info(f'Validating image: {index}')
                        low_res_image = images[:,:3,:,:]
                        high_res_image = images[:,3:,:,:]
                        
                        samples = sampler.sample_high_res(low_res_image)
                        metrics = Logger.evaluate_metrics(high_res_image,samples[-1])
                        
                        logger.info(f'Metrics for image{index}:')
                        logger.info(metrics)
                        
                        
                        
                        Utils.save_image(samples[-1],f'{opt["path"]["results"]}/validation_{index[0]}_{epoch}.png')
                                     
                        wandb_tracker.log_images({f"validation_{index[0]}":[np.concatenate([Utils.tensor2img(low_res_image),
                                                                                            Utils.tensor2img(samples[-1])
                                                                                            ,Utils.tensor2img(high_res_image)])]})
                        
                        for key in metrics.keys():
                            metrics_sum[key] += metrics[key]
                            
                        wandb_tracker.log_table("validation_table", )
                        #Utils.tensors_to_grid_image(samples,low_res_image,f'{opt["path"]["results"]}/test_{index[0]}_{epoch}.png')
                        
                metrics_mean = {}
                for key in metrics_sum.keys():
                    metrics_mean[key] = metrics_sum[key] / (count + 1)
                logger.info(f'Mean metrics for epoch {epoch}:')
                logger.info(metrics_mean)
                accelerator.log(metrics_mean, step=global_step)
                
            logger.info(f'saving model for epoch {epoch}')
              
                        
                                                
"""                        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:

                            evaluate(config, epoch, pipeline)
                            if config.push_to_hub:
                                repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                            else:
                                pipeline.save_pretrained(config.output_dir)"""
