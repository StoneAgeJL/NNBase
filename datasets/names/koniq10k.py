from torch.utils.data import Dataset
from os import path
import csv
from glob import glob
from utils import logging
from utils import load_tar
from utils import load_json
from ..build import DATASETS

logger = logging.get_logger(__name__)

@DATASETS.register()
class Koniq10k(Dataset):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        # self.mode = self.cfg.mode
        # assert self.mode in ["train", "val"]
        self.img_path = "512x384/*.jpg"
        self.img_path = path.join(self.cfg.data_path, self.img_path)
        self.csvfile = csv.DictReader(open("./koniq10k/koniq10k_scores_and_distributions.csv", mode='r', encoding='utf-8'))
        self.anno_info = {}
        for row in self.csvfile:
            self.anno_info[row["image_name"]] = float(row["MOS_zscore"]) / 10.
        self.img_path = glob(str(self.img_path))
        pass
    