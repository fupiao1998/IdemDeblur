from glob import glob
import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data, get_validation_data_SubSet, get_train_data_SubSet
import losses
from model.get_model import get_model_and_optim
from tqdm import tqdm
from pdb import set_trace as stx
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION
model_name = opt.MODEL.MODEL_NAME
log_name = '{}_{}_{}'.format(mode, model_name, opt.TRAINING.LOG_NAME)

log_dir = os.path.join(opt.TRAINING.SAVE_BASE_DIR, log_name)
model_dir  = os.path.join(opt.TRAINING.SAVE_BASE_DIR, log_name, 'models')
print('[INFO]: Save model and logs in {}'.format(model_dir))
utils.mkdir(model_dir)

######### Model & Optimizer & Scheduler #######
model, optimizer, scheduler, start_epoch = get_model_and_optim(opt, model_dir)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("[INFO]: Using", torch.cuda.device_count(), "GPUs!")

if len(device_ids) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

######### Loss ###########
criterion_char, criterion_edge = losses.CharbonnierLoss(), losses.EdgeLoss()

######### DataLoaders ###########
train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR
train_dataset = get_train_data_SubSet(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=opt.OPTIM.BATCH_SIZE, drop_last=False, pin_memory=True)

val_dataset = get_validation_data_SubSet(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)
# stx()
######### Backup codes #########
utils.dir_utils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
utils.dir_utils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
utils.dir_utils.save_scripts(log_dir, scripts_to_save=glob('utils/*.py', recursive=True))

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr, best_epoch = 0, 0
log_writer = SummaryWriter(log_dir=log_dir)
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss, train_id = 0, 1

    model.train()
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, opt.OPTIM.NUM_EPOCHS + 1))
    for i, data in enumerate(progress_bar):

        # zero_grad
        for param in model.parameters():
            param.grad = None

        target, input_ = data[0].cuda(), data[1].cuda()

        restored_1 = model(input_)
        restored_2 = model(restored_1[0].detach())
 
        # Compute loss at each stage
        loss_char = (criterion_char(restored_1[0], target) + criterion_char(restored_2[0], target))/2
        loss_edge = (criterion_edge(restored_1[0], target) + criterion_edge(restored_2[0], target))/2

        loss_idem = criterion_char(restored_1[0], restored_2[0])
        loss = (loss_char) + (0.05*loss_edge) + (0.1*loss_idem)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target, input_ = data_val[0].cuda(), data_val[1].cuda()

            with torch.no_grad():
                restored = model(input_)
            restored = restored[0]

            for res,tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("==> Epoch %d PSNR: %.4f | best_epoch %d | Best_PSNR %.4f" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

        torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        log_writer.add_scalar('PSNR', psnr_val_rgb, epoch)
        log_writer.add_scalar('BestPSNR', best_psnr, epoch)

    print("Epoch:{} | Time:{:.4f} | Loss:{:.4f} | LearningRate:{:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    log_writer.add_scalar('loss', epoch_loss, epoch)
    log_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
    scheduler.step()

    torch.save({'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 

