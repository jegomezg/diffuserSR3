import numpy as np
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, normalized_root_mse

import diffuserSR3.util as Utils


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(img1, img2):
    # Convert RGB images to grayscale
    image1_gray = rgb2gray(img1)
    image2_gray = rgb2gray(img2)

    # Calculate SSIM
    ssim_value = ssim(image1_gray, image2_gray,
                      data_range=image1_gray.max() - image1_gray.min())
    return ssim_value


def evaluate_metrics(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate metrics
    img1 = Utils.tensor2img(img1)
    img2 = Utils.tensor2img(img2)

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
