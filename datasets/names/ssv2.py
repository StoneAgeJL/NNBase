import random
import time

from os import path
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from utils import logging

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from einops import rearrange, repeat, reduce
import numpy as np
import ffmpeg
import cv2

from ..build import DATASETS

logger = logging.get_logger(__name__)

@DATASETS.register()
class Ssv2_webm(Dataset):
    def __init__(self, cfg):
        assert cfg.mode in ["train", "val", "test"], \
            f"dataset mode: {cfg.mode} not supported for Something-Something V2"
        self.cfg = cfg
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        pass
