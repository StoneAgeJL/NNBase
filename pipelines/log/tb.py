import torch
from tensorboardX import SummaryWriter as SummaryWriter

from utils import logging

from datetime import datetime

logger = logging.get_logger(__name__)

def is_master(rank):
    return rank == 0

class LoggerWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        kwargs["log_dir"] = "./tensorboard/" + TIMESTAMP
        super(LoggerWriter, self).__init__(*args, **kwargs)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, enable_print=False):
        super(LoggerWriter, self).add_scalar(tag, scalar_value, global_step, walltime)
        if enable_print:
            logger.info(f'TensorBoard: {tag} in step {global_step} - {scalar_value}')

    def add_image(self, tag, imgs: list, global_step, walltime=None, enable_print=False):
        if len(imgs) < 1: 
            logger.info("TensorBoard: empty image tensor list.")
            return
        super(LoggerWriter, self).add_image(tag, imgs, global_step, walltime)
        if enable_print:
            logger.info(f'TensorBoard: {tag} in step {global_step} - writedown image (shape: {imgs[0].shape.tolist()})')

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None, enable_print=False):
        super(LoggerWriter, self).add_histogram(tag, values, global_step, bins, walltime, max_bins)
        if enable_print:
            logger.info(f'TensorBoard Histogram: {tag}')
            
    def add_text(self, tag, text_string, global_step=None, walltime=None, enable_print=False):
        super(LoggerWriter, self).add_text(tag, text_string, global_step, walltime)
        if enable_print:
            logger.info(f"TensorBoard Text: tag is {tag}, {text_string}")
        