# from code.loss import FocalLoss
from utils import AverageMeter, AttrDict, acc, dice, iou
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import FocalLoss
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
import numpy as np
from datetime import datetime,date
import os
import numpy as np
from sklearn.utils import class_weight
from fastprogress.fastprogress import master_bar, progress_bar


class Learner:
    def __init__(self, model, opt_func="Adam", lr=1e-3, device = 'cpu', wandb = None):
        self.model = model
        self.opt_func = opt_func
        self.device = device
        self.model.to(self.device);
        ## init
        self.epoch_start = 0
        self.best_valid_loss = None
        self.best_valid_score = None
        # self.writer = SummaryWriter()
        self.wandb = wandb
    def get_dataloader(self, train_dl, valid_dl, mixup_fn, test_dl = None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.mixup_fn = mixup_fn
    def get_config(self, config):
        self.config = config        
        self.loss = FocalLoss(gamma=2)
        
    def train_one(self, epoch):
        self.model.train()
        summary_loss = AverageMeter()
        
        # tk0 = tqdm(self.train_dl, total=len(self.train_dl))
        num_steps = len(self.train_dl)
        d_ap = []
        d_an = []
        mean_triplet_loss = 0
        # for step, (images, targets) in enumerate(tk0):
        for step, (images, targets) in enumerate(progress_bar(self.train_dl, parent = self.mb)):
            outputs = self.model(images)
            losses = self.loss(outputs, targets)
            
            self.optimizer.zero_grad()

            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step_update(epoch * num_steps + step)
        
            self.model.zero_grad()

            summary_loss.update(losses.item(), self.config.DATA.BATCH_SIZE)
            
            # tk0.set_postfix(loss=summary_loss.avg)
        return summary_loss

    def evaluate_one(self, epoch):

        eval_model = self.model
        eval_model.eval()
        
        summary_loss = AverageMeter()
        list_outputs = []
        list_targets = []

        with torch.no_grad():
            
            # tk0 = tqdm(self.valid_dl, total=len(self.valid_dl))
            # for step, (images, targets) in enumerate(tk0):
            for images, targets in self.valid_dl:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = eval_model(images)
                losses = self.loss(outputs, targets)            
                summary_loss.update(losses.item(), self.config.DATA.BATCH_SIZE)
                # tk0.set_postfix(loss=summary_loss.avg)
                acc_score = acc(outputs, targets)
                dice_score = dice(outputs, targets)
                iou_score = iou(outputs, targets)
                metric ={"acc": acc, "dice": dice_score, "iou": iou}
            return summary_loss, metric

    def save_checkpoint(self, foldname):
        checkpoint = {}

        d = date.today().strftime("%m_%d_%Y") 
        h = datetime.now().strftime("%H_%M_%S").split('_')
        h_offset = int(datetime.now().strftime("%H_%M_%S").split('_')[0])+2
        h[0] = str(h_offset)
        h = '_'.join(h)
        filename =  d +'_' + h + '_epoch_' + str(self.epoch)

        checkpoint['epoch'] = self.epoch
        checkpoint['best_valid_loss'] = self.best_valid_loss
        checkpoint['best_valid_score'] = self.best_valid_score
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['scheduler'] = self.lr_scheduler.state_dict()

        f = os.path.join(foldname, filename + '.pth')
        torch.save(checkpoint, f)
        # print('Saved checkpoint')

    # def load_checkpoint(self, checkpoint_dir, is_train=False):
    #     checkpoint = torch.load(checkpoint_dir, map_location = {'cuda:0':'cpu'})    
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     if is_train:
    #         for parameter in self.model.parameters():
    #             parameter.requires_grad = True
    #     else:
    #         for parameter in self.model.parameters():
    #             parameter.requires_grad = False
    #     if self.config.TRAIN.USE_EMA:
    #         self.ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
    #         if is_train:
    #             for parameter in self.ema_model.ema.parameters():
    #                 parameter.requires_grad = True
    #         else:
    #             for parameter in self.ema_model.ema.parameters():
    #                 parameter.requires_grad = False
    #     self.epoch_start = checkpoint['epoch']
    #     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

    def fit(self):
        self.mb = master_bar(range(self.epoch_start, self.config.TRAIN.EPOCHS))
        count_early_stop = 0
        for epoch in self.mb:
            if count_early_stop > 5:
                print('Early stopping')
                break
            else:
                self.epoch = epoch
                # print(f'Training epoch: {self.epoch} | Current LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                train_loss = self.train_one(self.epoch)
                # self.writer.add_scalar("Loss/train", train_loss.avg, epoch)
                self.wandb.log({"Loss/train": train_loss.avg})
                # print(f'\tTrain Loss: {train_loss.avg:.3f}')
                if (self.epoch)% self.config.TRAIN.FREQ_EVAL == 0:
                    valid_loss, valid_metric = self.evaluate_one(self.epoch)
                    self.wandb.log({"Loss/valid": valid_loss.avg, 
                                    "Metric/dice": valid_metric['dice'],
                                    "Metric/acc": valid_metric['acc'],
                                    "Metric/iou": valid_metric['iou']})
                    # self.writer.add_scalar("Loss/valid", valid_loss.avg, epoch)
                    # self.writer.add_scalar("Metric/f1", valid_metric['macro/f1'], epoch)
                    if self.best_valid_loss and self.best_valid_score:
                        if self.best_valid_loss > valid_loss.avg and self.best_valid_score < float(valid_metric['dice']):
                            self.best_valid_loss = valid_loss.avg
                            self.best_valid_score = float(valid_metric['dice'])
                            self.save_checkpoint(self.config.TRAIN.SAVE_CP)
                        elif self.best_valid_loss < valid_loss.avg or self.best_valid_score > float(valid_metric['dice']):
                            # print('Early stopping')
                            count_early_stop += 1
                        else:
                            ## do nothing
                            pass
                    else:
                        self.best_valid_loss = valid_loss.avg
                        self.best_valid_score = float(valid_metric['dice'])
                        self.save_checkpoint(self.config.TRAIN.SAVE_CP)
                    # print(f'\tValid Loss: {valid_loss.avg:.3f}')
                    # f1_score = valid_metric['macro/f1']
                    # print(f'\tMacro F1-score: {f1_score}')
            
            # self.writer.flush()
            # self.writer.close()