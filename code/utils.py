from skimage.draw import polygon
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from datetime import datetime,date
import os
import yaml
from yaml.loader import SafeLoader
import cv2
from torch import Tensor

def generate_mask(annos, resolution):
    mask = np.zeros(shape=resolution, dtype=int)
    
    for coor in list(annos):
        poly = np.array(coor).reshape((int(len(coor)/2), 2))
        rr, cc = polygon(poly[:,0], poly[:,1], resolution)
        mask[cc,rr] = 1
    return mask

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    config = AttrDict(config)
    for k1 in config.keys():
        config[k1] = AttrDict(config[k1])
    return config

def acc(input, target):
    """
    segmentation accuracy
    """
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

def dice(input:Tensor, targs:Tensor):
    """
    DICE score is the F1-score for the segmentation problems.
    """
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.squeeze(1).view(n,-1)
    intersect = (input*targs).float().sum()
    union = (input+targs).float().sum()
    return 2. * intersect / union
    
def iou(input:Tensor, targs:Tensor):
    """
    Intersection over Union
    """
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.squeeze(1).view(n,-1)
    intersect = (input*targs).float().sum()
    union = (input+targs).float().sum()
    return intersect / (union-intersect+1.0)

def show_batch(inp, title=None):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig=plt.figure(figsize=(20, 7))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)