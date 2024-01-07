import yaml
from types import SimpleNamespace
from collections import namedtuple

def convert_to_namedtuple(d):
   # 递归函数，用于处理嵌套字典
   def _convert(obj):
       if isinstance(obj, dict):
           field_names = [key if isinstance(key, str) and key.isidentifier() else f"field_{i}"
                          for i, key in enumerate(obj.keys())]
           NestedNamedTuple = namedtuple('NestedNamedTuple', field_names)
           return NestedNamedTuple(**{k: _convert(v) for k, v in obj.items()})
       elif isinstance(obj, list):
           return [_convert(item) for item in obj]
       else:
           return obj

   return _convert(d)

def namedtuple_to_dict(obj):
   # 递归函数，用于处理嵌套的namedtuple和列表
   if hasattr(obj, "_asdict"):  # 检查obj是否是namedtuple
       return {key: namedtuple_to_dict(value) for key, value in obj._asdict().items()}
   elif isinstance(obj, list):  # 检查obj是否是列表
       return [namedtuple_to_dict(item) for item in obj]
   else:
       return obj

def read_cfg(yaml_path, dict_type=True):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
        # return SimpleNamespace(**cfg)
        if dict_type: return cfg
        return convert_to_namedtuple(cfg)
    
if __name__ == "__main__":
    
    cfg = read_cfg("/home/hadoop-mtcv/dolphinfs_hdd_hadoop-mtcv/lijian134/Code/temporal_arrangement/configs/base.yaml", dict_type=False)
    print(cfg)
    cfg_dict = namedtuple_to_dict(cfg)
    print(cfg_dict)