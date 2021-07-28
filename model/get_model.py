from warmup_scheduler import GradualWarmupScheduler
from model.IdemNetStage3 import IdemNetStage3
from model.IdemNetStage4 import IdemNetStage4
import torch.optim as optim
import utils
import os


def get_model_and_optim(opt, model_dir):
    name = opt.MODEL.MODEL_NAME
    new_lr = opt.OPTIM.LR_INITIAL

    if name.lower() == 'idemnetstage3':
        model = IdemNetStage3().cuda()
    elif name.lower() == 'idemnetstage4':
        model = IdemNetStage4().cuda()
    else:
        print('[ERROR]: NotImplementednt!')
        exit()
    print("[INFO]: There are {:.3f}M parameters".format(sum(param.numel() for param in model.parameters()) / 10**6))

    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    start_epoch = 1
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    return model, optimizer, scheduler, start_epoch
