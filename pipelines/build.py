import torch
from utils.registry import Registry

PIPES = Registry("PIPE")

def build_pipe(cfg):
    name = cfg["name"]
    name = name.capitalize()
    model = PIPES.get(name)(**cfg)
    return model
