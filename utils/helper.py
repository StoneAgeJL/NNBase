import json
import jsonlines
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

import numpy as np
import random
import tarfile

import io
import re

from tqdm import tqdm

import torch.nn.functional as F
import copy

import importlib

import pickle
import pandas as pd

from torch.utils.data.dataloader import default_collate

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def load_jsonl(jsonl_path):
    info = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            info.append(item)
    return info

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_jsonl(jsonl_path, list_data):
    with open(jsonl_path, "w", encoding="utf8") as f:
        with jsonlines.Writer(f) as writer:
            for x in list_data:
                writer.write(x)

def write_json(json_path, list_data):
    with open(json_path, 'w') as f:
        json.dump(list_data, f, indent=4, ensure_ascii=False)
        
def load_tar(tar_path, refer_names=None):
    rvl = {}
    with tarfile.open(tar_path, 'r') as ctar:
        for mem in tqdm(ctar.getmembers()):
            exfile = ctar.extractfile(mem)
            assert(exfile is not None)
            file_stem = mem.name.split('.')[-2][1:]
            if refer_names is None or file_stem in refer_names:
                npz = np.load(io.BytesIO(exfile.read()))
                rvl[file_stem] = npz["features"]
    return rvl
                

class DataBuffer(object):
    def __init__(self, buffer_size=-1, data_len=-1):
        self._buffer = OrderedDict() # 有序样本序列
        self._buffer_sz = buffer_size
        # self.freq_list = np.zeros(data_len, dtype=np.int32)
    
    def get(self, idx):
        # if idx in self._buffer:
        #     self._buffer.move_to_end(idx, last=True) # 移到尾部
        return self._buffer[idx]
    
    def put(self, idx, info):
        # if idx in self._buffer:
        #     self._buffer.move_to_end(idx, last=False) # 移到队头
        #     # self._buffer[idx] = info
        if not self.is_exist(idx):
            if len(self._buffer) >= self._buffer_sz:
                self._buffer.popitem(last=False) # 去除最早加入的元素(队头)
            self._buffer[idx] = info
    
    def clear(self):
        self._buffer.clear()
        
    def set_buffer_size(self, sz):
        self._buffer_sz = sz
        
    def is_exist(self, idx):
        return idx in self._buffer
    
    def get_range(self):
        if len(self._buffer) < 1:
            return -1, 0
        index_list = list(self._buffer.keys())
        return np.min(index_list), np.max(index_list) + 1

def pad_sequence(seq: torch.Tensor, seq_len):
    assert isinstance(seq, torch.Tensor), f"Error type of seq: {type(seq)}"
    pad_shape = list(seq.shape)
    pad_shape[0] = seq_len - seq.shape[0]
    if pad_shape[0] < 0:
        s = random.randint(0, abs(pad_shape[0]) - 1)
        return seq[s:s+seq_len], torch.ones(seq_len, device=seq.device, dtype=seq.dtype)
    elif pad_shape[0] > 0:
        return torch.cat([seq, torch.zeros(*pad_shape, device=seq.device, dtype=seq.dtype)]), \
            torch.cat([
                torch.ones(seq.shape[0], device=seq.device, dtype=seq.dtype),
                torch.zeros(pad_shape[0], device=seq.device, dtype=seq.dtype)
            ])
    return seq, torch.ones(seq.shape[0], device=seq.device, dtype=seq.dtype)

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def select_shuffle_arr(array: np.array, n):
    indices = np.arange(len(array)) # 获取数组的索引
    selected_indices_raw = np.random.choice(indices, size=n, replace=False) # 无放回地随机选择n个索引
    selected_indices_shuffle = np.copy(selected_indices_raw)
    np.random.shuffle(selected_indices_shuffle) # 打乱选中的索引
    rvl = np.copy(array)
    rvl[selected_indices_raw] = rvl[selected_indices_shuffle] # 使用打乱的索引更新对应的元素
    indices[selected_indices_raw] = indices[selected_indices_shuffle]
    return rvl, indices

def select_shuffle_tensor(array: torch.Tensor, n, mask: torch.Tensor):
    if n == -1: n = int(torch.sum(mask))
    indices = torch.arange(len(array), device=array.device) # 获取数组的索引
    selected_indices_raw = torch.randperm(len(indices[mask.bool()]))[:n] # 无放回地随机选择n个索引
    selected_indices_shuffle = selected_indices_raw.clone() #复制原数组
    selected_indices_shuffle = selected_indices_shuffle[torch.randperm(n)] # 打乱选中的索引
    rvl = array.clone()
    rvl[selected_indices_raw] = rvl[selected_indices_shuffle] # 使用打乱的索引更新对应的元素
    indices[selected_indices_raw] = indices[selected_indices_shuffle]
    return rvl, indices

# @torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

def average_all_gather(tensor):
    if isinstance(tensor, torch.Tensor):
        averaged = tensor.detach().clone()
        torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
        return averaged / torch.distributed.get_world_size()
    elif isinstance(tensor, dict):
        for k, v in tensor.items():
            tensor[k] = average_all_gather(v)
    return tensor # 一些数值的类型 如int等

def average_all(tensor):
    if isinstance(tensor, torch.Tensor):
        averaged = tensor.detach().item()
        return averaged
    elif isinstance(tensor, dict):
        for k, v in tensor.items():
            tensor[k] = average_all(v)
    elif not isinstance(tensor, list): # list的变量不读取
        return tensor
    return None

def delete_None(d: dict):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if v is not None}

def no_contains_chinese(check_str):
    for ch in check_str:
        if re.match(u'[\u4e00-\u9fa5]+',ch):
            return False
    return True

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
        (size_a,), (size_b,) -> (size_a, size_b)
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def mask_logits(inputs, mask, mask_value=-1e30): # 1 -> 0, 0 -> -1e30(无穷小)
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def read_gpu(dist_type, gpu_devices):
    if dist_type is None:
        if isinstance(gpu_devices, int):
            return gpu_devices if gpu_devices != -1 else 0
        elif isinstance(gpu_devices, (list,tuple)):
            return gpu_devices[0]
    else:
        return -1

def build_optim(params, cfg):
    optim_module = importlib.import_module("torch.optim")
    optim_func = getattr(optim_module, cfg["name"])
    cfgg = copy.deepcopy(cfg)
    cfgg.pop("name")
    return optim_func(params, **cfgg)

def build_lr_scheduler(optimizer, cfg):
    lr_scheduler = importlib.import_module("torch.optim.lr_scheduler")
    lr_scheduler_func = getattr(lr_scheduler, cfg["name"])
    cfgg = copy.deepcopy(cfg)
    cfgg.pop("name")
    return lr_scheduler_func(optimizer, **cfgg)

def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def custom_collate_fn(batch):
    elem = batch[0]
    batch_shot_info = None
    if "shot_info" in elem: # shotsranker
        batch_shot_info = [d["shot_info"][0] for d in batch], [d["shot_info"][1] for d in batch]
        for b in batch: b.pop("shot_info")
    rvl = default_collate(batch)
    if batch_shot_info:
        rvl["shot_info"] = batch_shot_info
    return rvl

def start_end_collate_mr(batch): # univtg?
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    
    batched_data = dict()
    
    for k in model_inputs_keys:
        
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue

        # batched_data[k] = pad_sequences_1d(
        # [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
        
    return batch_meta, batched_data

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    return data.to(device, non_blocking=True)

def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)
