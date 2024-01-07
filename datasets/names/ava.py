from torch.utils.data import Dataset
from os import path
from utils import logging
from utils import load_tar
from utils import load_json
import numpy as np

from ..build import DATASETS

logger = logging.get_logger(__name__)

@DATASETS.register()
class Avarate(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = self.cfg.mode # train / val
        assert self.mode in ["train", "val"]
        
        # filesystem
        self.anno_path = f"clip_b32_{self.mode}.json"
        self.data_path = f"clip_b32.tar"
        self.anno_path = path.join(self.cfg.data_path, self.anno_path)
        self.data_path = path.join(self.cfg.data_path, self.data_path)
        
        # embedding
        self.anno_info = load_json(self.anno_path)
        emb_info = load_tar(self.data_path, refer_names=self.anno_info)
        self.info = [] # 数据集加载
        for name in self.anno_info.keys():
            self.info.append({
                "img_id": name,
                "img_embed": emb_info[name].astype(np.float32).reshape(-1),
                "img_rate": np.array([self.anno_info[name]], dtype=np.float32).reshape(-1)
            })
            
        logger.info(f"Load {self.anno_path} with {len(self.info)} samples in {self.mode} mode.")
        # del emb_info # too big, clean it immediately
    
    def __len__(self):
        return len(self.info)
    
    def get_info(self, index):
        return self.info[index]
    
    def get_item(self, index):
        info = self.get_info(index)
        return info["img_embed"], info["img_rate"]
    
    def __getitem__(self, index):
        img_embed, img_rate = self.get_item(index)
        return {
            "img_embed": img_embed,
            "img_rate": img_rate
        }
    
        