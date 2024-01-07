import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils import logging
from ..build import MODELS

logger = logging.get_logger(__name__)

@MODELS.register()
class Rator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def single_inference(self, emb):
        return self.layers(emb)
    
    def cal_loss(self, *args, **kwargs):
        img_embed = kwargs["data_iter"].get("img_embed")
        img_rate = kwargs["data_iter"].get("img_rate")
        x_hat = self.single_inference(img_embed)
        loss = F.mse_loss(x_hat, img_rate, reduction="mean")
        return {
            "loss": loss
        }
    
    @torch.no_grad()
    def cal_metric(self, *args, **kwargs):
        img_embed = kwargs["data_iter"].get("img_embed")
        img_rate = kwargs["data_iter"].get("img_rate")
        x_hat = self.single_inference(img_embed)
        predict_score = x_hat.cpu().numpy().reshape(-1)
        gt_score = img_rate.cpu().numpy().reshape(-1)
        # rmse = np.sqrt(np.mean(np.power(predict_score - gt_score, 2)))
        mse = np.mean(np.power(predict_score - gt_score, 2))
        pearson_corr, _ = pearsonr(predict_score, gt_score)
        spearman_corr, _ = spearmanr(predict_score, gt_score)
        return {
            "mse": mse,
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr
        }

    def forward(self, *args, **kwargs):
        cal_loss = kwargs.get("cal_loss", False)
        cal_metric = kwargs.get("cal_metric", False)
        if not cal_loss and not cal_metric:
            if len(args) > 0:
                embeds = args
            else:
                embeds = kwargs["data_iter"]
            return self.single_inference(embeds)
        elif cal_loss:
            return self.cal_loss(*args, **kwargs)
        elif cal_metric:
            return self.cal_metric(*args, **kwargs)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer