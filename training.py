import torch

from datasets import build_dataset
from models import build_model
from pipelines import build_pipe

from utils import logging
from utils import read_gpu
from utils.config import read_cfg
from utils.config import convert_to_namedtuple

import argparse

def main(cfg_path, dist_type="UnSet"):
    cfg_dict = read_cfg(cfg_path, dict_type=True)
    pipe_info = cfg_dict["pipe"]
    pipe_info.update(cfg_dict["log"])
    
    if dist_type in ["None", "ddp"]: # 调整
        pipe_info["dist_type"]["name"] = dist_type if dist_type != "None" else None
    
    # setup logger
    logging.setup_logging(output_dir=pipe_info["output_path"])
    
    # build dataset
    training_dataset = build_dataset(convert_to_namedtuple(cfg_dict["data"]["training"]))
    evaluate_dataset = build_dataset(convert_to_namedtuple(cfg_dict["data"]["evaluate"]))
    
    # build model
    model = build_model(convert_to_namedtuple(cfg_dict["model"]))
    
    # build pipeline
    gpu_device = read_gpu(pipe_info["dist_type"]["name"], pipe_info["dist_type"]["device"])
    if gpu_device != -1: torch.cuda.set_device(gpu_device) # 单卡模式下设定默认显卡设备编号
    # print(f"GPU-setting: {gpu_device}, CUDA device: {torch.cuda.current_device()}")
    
    pipe_parames = {
        "name": pipe_info["name"],
        "model": model,
        "training_data": training_dataset,
        "eval_data": evaluate_dataset,
        "config": pipe_info
    }
    pipe = build_pipe(pipe_parames)
    pipe.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline Codebase.')
    parser.add_argument('config_path', type=str, help='path of configuration file')
    parser.add_argument('--dist_type', default="UnSet", type=str, help='type of distrubution')
    args = parser.parse_args()
    main(args.config_path, args.dist_type)
