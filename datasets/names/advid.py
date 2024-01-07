import random
# import time

from os import path
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from utils import logging

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms

# from einops import rearrange, repeat, reduce
import numpy as np
import random
import traceback

# from utils import translate_sentences
from utils import CNClip
from utils import load_json
from utils import DataBuffer
from utils import pad_sequence
from utils import select_shuffle_tensor
from utils import no_contains_chinese

from ..build import DATASETS

logger = logging.get_logger(__name__)

@DATASETS.register()
class Advis(Dataset):
    def __init__(self, cfg):
        '''cfg:
            mode
            max_num_shot
            use_preload
            gen_pos_neg
            data_path
            device
            seed
            buffer_sz
            shuffle_control_rate
            shuffle_len
        '''
        self.cfg = cfg
        self.mode = self.cfg.mode
        self.max_num_shot = self.cfg.max_num_shot
        self.use_preload = self.cfg.use_preload # False
        self.gen_pos_neg = self.cfg.gen_pos_neg # True-Train
        assert self.mode in ["train", "val"]
        
        # filesystem
        self.anno_path = f"metadata/v1_easy_title_{self.mode}.json"
        self.anno_path = path.join(self.cfg.data_path, self.anno_path)
        self.shot_path = "data/v1_frame/"
        self.shot_path = path.join(self.cfg.data_path, self.shot_path)
        
        self.info = load_json(self.anno_path)
        # filter only-1shot
        self.info = [s for i, s in enumerate(self.info) if len(self.get_shot_info(i)[1]) > 1]
        
        logger.info(f"Load {self.anno_path} with {len(self.info)} samples in {self.mode} mode.")

        # embedding
        assert self.cfg.device in ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())], \
            f"GPU device: {self.cfg.device} not supported."
        self.clip = CNClip(model_name="ViT-L-14-336", device=self.cfg.device) # imgsz:336,embsz:768
        
        # shuffle & buffer
        if self.use_preload:
            self.raw_seed = self.cfg.seed
            self._buffer_size = self.cfg.buffer_sz        
            self._buffer = DataBuffer(self._buffer_size)
            self.do_shuffle()
        
        # item_cnt
        self.item_count = 0
        
    def __len__(self):
        return len(self.info)

    def get_valid_len(self):
        return self.max_num_shot, self.clip.text_seq_len # num_shot, bert_seq_length

    def get_shot_info(self, index):
        item = self.info[index % len(self.info)]
        movie_id = item["movie_id"]
        keyframes = item["keyframe"]
        shot_list = [path.join(self.shot_path, movie_id, shot) for shot in keyframes]
        synopses = item["synopses"]
        # chinese_text = translate_sentences(synopses) # 英译中
        chinese_text = [s for s in synopses if not no_contains_chinese(s)]
        chinese_text = "，".join(chinese_text)
        return chinese_text, shot_list
        
    def get_item(self, index):
        chinese_text, shot_path_list = self.get_shot_info(index)
        
        # shot:
        shots_embedding = self.clip.encode_image(shot_path_list, pre_load=True) # B,K
        
        # text:
        text_embedding = self.clip.encode_text(text_list=chinese_text, return_seq=True) # 1,X,K (should be X,K if return_seq=True) 
        shots_embedding = shots_embedding.reshape(shots_embedding.shape[0], -1, shots_embedding.shape[-1]) # (shot_num, -1, emb_size)
        text_embedding = text_embedding.reshape(-1, text_embedding.shape[-1]) # (bert_seq_len, emb_size)
                
        shots_embedding, shot_mask = pad_sequence(shots_embedding, self.max_num_shot) # shot_mask: (shot_num)
        return shots_embedding, text_embedding, shot_mask
    
    def pre_load(self, s, e):
        for id in tqdm(range(s, e)):
            if id < 0 and id >= len(self.info): 
                continue
            if not self._buffer.is_exist(id):
                self._buffer.put(id, self.get_item(id))

    def epoch_stop(self):
        return self.item_count >= len(self.info)
    
    def do_shuffle(self):
        self.item_count = 0
        
        self.raw_seed = self.raw_seed % 1000000
        random.seed(self.raw_seed)
        random.shuffle(self.info)
        self._buffer.clear()
        self.raw_seed += 1
        
        logger.info(f"do dataset shuffle, seed: {self.raw_seed}")
        self.pre_load(0, self._buffer_size)
        logger.info(f"pre-loading dataset samples: {self._buffer_size}")
        
    def shuffle_sequence(self, shots_embedding, shot_mask):
        
        self.shuffle_control_rate = self.cfg.shuffle_control_rate
        self.shuffle_len = self.cfg.shuffle_len # 3,4,5

        assert self.shuffle_control_rate >= 0 and self.shuffle_control_rate <= 1, \
            "shuffle_control_rate should be [0, 1]"
        assert self.shuffle_len > 1, "shuffle_len should be [1, ∞)"        
        
        def is_shuffled(seq):
            if len(seq) < 2: return False
            for i in range(1, len(seq)):
                if seq[i] != seq[i - 1] + 1:
                    return True
            return False
        
        def empty_return_val():
            return torch.zeros_like(shots_embedding).expand(self.shuffle_len, -1, -1, -1), \
                torch.zeros((self.shuffle_len, shots_embedding.shape[0]), device=shots_embedding.device), \
                torch.zeros(self.shuffle_len, device=shots_embedding.device)
                    
        # shots_embedding, text_embedding, shot_mask = cur_sample
        valid_len = torch.sum(shot_mask)
        if valid_len < 2:
            return empty_return_val()
        
        shuffle_cnt = torch.arange(2, valid_len + 1, max(int(torch.floor(valid_len * self.shuffle_control_rate)), 1), dtype=torch.long)
        shuffle_cnt = shuffle_cnt[torch.randperm(len(shuffle_cnt))] # shuffle
        
        shuffle_embeds = []
        shuffle_seq_gt = []
        for n in shuffle_cnt:
            emb, shuffle_id = select_shuffle_tensor(shots_embedding, n, shot_mask)
            if not is_shuffled(shuffle_id[shot_mask.bool()]): continue
            shuffle_embeds.append(emb)
            shuffle_seq_gt.append(torch.argsort(shuffle_id))
        
        if len(shuffle_embeds) < 1:
            return empty_return_val()
        
        shuffle_embeds = torch.stack(shuffle_embeds, dim=0)
        shuffle_seq_gt = torch.stack(shuffle_seq_gt, dim=0)
        
        shuffle_embeds, shuffle_mask = pad_sequence(shuffle_embeds, self.shuffle_len)
        shuffle_seq_gt, _ = pad_sequence(shuffle_seq_gt, self.shuffle_len)
        return shuffle_embeds, shuffle_seq_gt, shuffle_mask
    
    def __getitem__(self, index):
        
        if self.use_preload:
            if self.epoch_stop():
                self.do_shuffle()
            
            if not self._buffer.is_exist(index):
                logger.info(f"pre-loading dataset samples: {index} ~ {index + self._buffer_size // 2 + 1}")
                self.pre_load(index, index + self._buffer_size // 2 + 1)
                
            self.item_count += 1

            return self._buffer.get(index)

        if not self.gen_pos_neg: # inference or evaluation
            return {
                "cur_sample": self.get_item(index),
                "shot_info": self.get_shot_info(index)
            }
        
        shots_embedding, text_embedding, shot_mask = self.get_item(index)
        random_pos = random.choice(list(range(0, index)) + list(range(index + 1, len(self.info))))
        pos_shots_embedding, pos_text_embedding, pos_shot_mask = self.get_item(random_pos)
        shuffle_embedding, shuffle_seq_gt, shuffle_mask = self.shuffle_sequence(shots_embedding, shot_mask)
        
        return { # training
            "cur_sample": (shots_embedding, text_embedding, shot_mask),
            "pos_sample": (pos_shots_embedding, pos_text_embedding, pos_shot_mask),
            "neg_sample": (shuffle_embedding, shuffle_seq_gt, shuffle_mask,
                           text_embedding.repeat(shuffle_embedding.shape[0], 1, 1),
                           shot_mask.repeat(shuffle_embedding.shape[0], 1))
        }
