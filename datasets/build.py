from utils import Registry

DATASETS = Registry("DATASET")

def build_dataset(cfg):
    data_name = cfg.name
    name = data_name.capitalize()
    return DATASETS.get(name)(cfg)
