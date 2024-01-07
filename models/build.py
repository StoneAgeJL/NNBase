import torch
from utils.registry import Registry

MODELS = Registry("MODEL")

def build_model(cfg):
    name = cfg.name
    name = name.capitalize()
    model = MODELS.get(name)(cfg)
    return model
