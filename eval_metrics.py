# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.utils.metrics.psnr_ssim import calculate_ssim, calculate_psnr
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])


lpips_model = lpips.LPIPS(net='alex').cuda()  # 'alex', 'vgg', or 'squeeze'
lpips_model.eval()

folder1 = '/home/jthe/DeblurDM/dataset/test/GoPro/target'
folder2 = '/home/jthe/BlurDM/results/FFTformer/GoPro/results/GoPro'

files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

common_files = files1.intersection(files2)

lpips_scores = []
psnr_scores = []
ssim_scores = []
c = 0

for file_name in common_files:
    c += 1
    print(c, '/' ,len(common_files))
    img1_path = os.path.join(folder1, file_name)
    img2_path = os.path.join(folder2, file_name)

    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    img1_tensor = transform(img1).unsqueeze(0).cuda()
    img2_tensor = transform(img2).unsqueeze(0).cuda()

    with torch.no_grad():
        lpips_score = lpips_model(img1_tensor, img2_tensor)

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # psnr_score = psnr(img1_np, img2_np, data_range=255)
    psnr_score =  calculate_psnr(img1_np, img2_np, crop_border=0, test_y_channel=False)
    ssim_score = calculate_ssim(img1_np, img2_np, crop_border=0, test_y_channel=False)

    lpips_scores.append(lpips_score.item())
    psnr_scores.append(psnr_score)
    ssim_scores.append(ssim_score)

average_lpips = sum(lpips_scores) / len(lpips_scores)
average_psnr = sum(psnr_scores) / len(psnr_scores)
average_ssim = sum(ssim_scores) / len(ssim_scores)

print("Results:")
# for file_name, l_score, p_score, ssim_scores in zip(common_files, lpips_scores, psnr_scores, ssim_scores):
#     print(f"{file_name}: LPIPS={l_score:.4f}, PSNR={p_score:.2f}")

print(f"\nAverage LPIPS: {average_lpips:.4f}")
print(f"Average PSNR: {average_psnr:.4f}")
print(f"Average SSIM: {average_ssim:.4f}")