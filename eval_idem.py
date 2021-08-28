"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import time
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data, get_validation_data
from model.get_model import get_model
from skimage import img_as_ubyte
from utils.image_utils import torchPSNR

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./checkpoints-IdemNet/Deblurring/models/IdemNet-test/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()

model_restoration = get_model(name='idemnetstage4')

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset, 'test')
test_dataset = get_validation_data(rgb_dir_test, img_options={'patch_size': None})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

count_time, psnr_list, psnr_list_2, psnr_list_3, psnr_list_4, psnr_list_5, psnr_list_6, psnr_list_7 = [], [], [], [], [], [], [], []
psnr_list_8, psnr_list_9, psnr_list_10 = [], [], []
with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        # torch.cuda.ipc_collect()
        # torch.cuda.empty_cache()

        input_img = data_test[1].cuda()
        target_img = data_test[0].cuda()
        # import pdb; pdb.set_trace()

        # torch.cuda.synchronize()
        time_start = time.time()
        restored_img = model_restoration(input_img)
        restored_img_2 = model_restoration(restored_img[0])
        restored_img_3 = model_restoration(restored_img_2[0])
        restored_img_4 = model_restoration(restored_img_3[0])
        restored_img_5 = model_restoration(restored_img_4[0])
        restored_img_6 = model_restoration(restored_img_5[0])
        restored_img_7 = model_restoration(restored_img_6[0])
        restored_img_8 = model_restoration(restored_img_7[0])
        restored_img_9 = model_restoration(restored_img_8[0])
        restored_img_10 = model_restoration(restored_img_9[0])
        # torch.cuda.synchronize()
        time_end = time.time()

        count_time.append(time_end-time_start)
        restored_img = restored_img[0]
        restored_img_2 = restored_img_2[0]
        restored_img_3 = restored_img_3[0]
        restored_img_4 = restored_img_4[0]
        restored_img_5 = restored_img_5[0]
        restored_img_6 = restored_img_6[0]
        restored_img_7 = restored_img_7[0]
        restored_img_8 = restored_img_8[0]
        restored_img_9 = restored_img_9[0]
        restored_img_10 = restored_img_10[0]
        # restored_img = torch.clamp(restored_img[0], 0, 1)

        psnr_list.append(torchPSNR(restored_img, target_img).cpu().numpy())
        psnr_list_2.append(torchPSNR(restored_img_2, target_img).cpu().numpy())
        psnr_list_3.append(torchPSNR(restored_img_3, target_img).cpu().numpy())
        psnr_list_4.append(torchPSNR(restored_img_4, target_img).cpu().numpy())
        psnr_list_5.append(torchPSNR(restored_img_5, target_img).cpu().numpy())
        psnr_list_6.append(torchPSNR(restored_img_6, target_img).cpu().numpy())
        psnr_list_7.append(torchPSNR(restored_img_7, target_img).cpu().numpy())
        psnr_list_8.append(torchPSNR(restored_img_8, target_img).cpu().numpy())
        psnr_list_9.append(torchPSNR(restored_img_9, target_img).cpu().numpy())
        psnr_list_10.append(torchPSNR(restored_img_10, target_img).cpu().numpy())

        print(np.mean(psnr_list), np.mean(psnr_list_2), np.mean(psnr_list_3), np.mean(psnr_list_4), np.mean(psnr_list_5), np.mean(psnr_list_6), np.mean(psnr_list_7), np.mean(psnr_list_8), np.mean(psnr_list_9), np.mean(psnr_list_10))
