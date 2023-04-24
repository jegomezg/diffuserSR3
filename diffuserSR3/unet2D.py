from diffusers import UNet2DModel


def create_unet2D(opt):
    
    model = UNet2DModel(
        sample_size = opt['image_size'],  # the target image resolution
        in_channels = opt['in_channels'],  # the number of input channels, 3 for RGB images
        out_channels = opt['out_channels'],  # the number of output channels
        layers_per_block = opt['layers_per_block'],  # how many ResNet layers to use per UNet block
        block_out_channels=tuple(opt['block_out_channels']),  # the number of output channels for each UNet block
        down_block_types=tuple(opt['down_block_types']),
        up_block_types=tuple(opt['up_block_types'])
    )
    
    return model