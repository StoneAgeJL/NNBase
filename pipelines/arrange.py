import os
from typing import Optional, Union
from pathlib import Path
from glob import glob
from datetime import datetime

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from deepspeed.runtime.engine import DeepSpeedOptimizerCallable
from deepspeed.runtime.engine import DeepSpeedSchedulerCallable
# from deepspeed.utils import zero_to_fp32

from .dist.ddp import DDPHelper
from .log.tb import LoggerWriter as TbLogger
from .log.wb import LoggerWriter as WbLogger

from utils import logging
from utils import average_all_gather, average_all, delete_None
from utils import write_jsonl

from utils import build_optim
from utils import build_lr_scheduler

from utils import dict_to_markdown
from utils import custom_collate_fn
from utils import to_device

import copy

from .build import PIPES

logger = logging.get_logger(__name__)

@PIPES.register()
class Basepipe(object):
    """config = {
        snapshot_path
        tag
        logger_type: (tensorboard, wandb)
        batch_size
        num_epoch
        gradient_accumulation_steps
        save_every
        save_every_iter
        enable_epoch_log
        enable_iter_log
        eval_every
        log_every
        dist_type: (ddp, deepspeed)
    }
    """
    def __init__(
        self,
        model: torch.nn.Module,
        training_data: Optional[Dataset],
        eval_data = None,
        optimizer=None,
        lr_scheduler=None,
        model_parameters=None,
        mpu=None,
        dist_init_required: Optional[bool] = None,
        collate_fn=None,
        config: dict = {},
        config_params=None,
        ckpt_name=None, **kwargs):
        
        # if model_parameters is None:
        #     model_parameters = [p for p in model.parameters() if p.requires_grad]

        self.model = model
        # self.model_param = model_parameters
        self.train_data = training_data
        self.eval_data = eval_data
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.reset_config(config)
        self.model_data_wrapper(config)
        
        # resume or not
        if ckpt_name is None: ckpt_name = "latest"
        self.ckpt_io(config, name=ckpt_name, mode="load")

        logger.info(f"config of pipeline: {self.config}, start from {self.epoch_resum}")
        
        self.vis = [] # 保存一些eval的待可视化结果

    def reset_config(self, config):
        
        self.logger_type = config.get("logger_type", None)
        self.logger = None
        
        self.config = config
        self.batch_size = config.get('batch_size', 1)
        self.num_epoch = config.get('num_epoch', 1)
        self.num_epoch_step = len(self.train_data) // self.batch_size
        self.accu_steps = config.get('gradient_accumulation_steps', 1)     
           
        self.epoch = 0
        self.epoch_resum = 0
        self.global_iter = dict()
        
        self.save_every = config.get("save_every", None)
        self.save_every_iter = config.get("save_every_iter", None)
        self.max_num_save = config.get("max_num_save", None)
        
        self.enable_epoch_log = config.get("enable_epoch_log", False)
        self.enable_iter_log = config.get("enable_iter_log", False)
        
        self.eval_every = config.get("eval_every", None)
        
        self.log_every = config.get("log_every", None)
        
        
    def ckpt_io(self, config, name="latest", mode="save"):
        self.snapshot_path = config.get("snapshot_path", "")
        self.tag = config.get("tag", "")
        self.snapshot_path = Path(self.snapshot_path) / name
        if mode == "load":
            self.load_checkpoint(self.snapshot_path, self.tag)
        else:
            self.save_checkpoint(self.snapshot_path, self.tag)
        
        
    def model_data_wrapper(self, config):
        self.dist_type = config.get("dist_type", None)
        if isinstance(self.dist_type, dict):
            self.dist_type = self.dist_type["name"]
        
        if self.dist_type is None:
            self.model = self.model.cuda()
            # self.device = self.model.device
            self.train_data = DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
                                        # num_workers=cfg.num_worker, pin_memory=True, drop_last=True, 
                                        # prefetch_factor=10, persistent_workers=True
                                        
            if self.eval_data:
                self.eval_data = DataLoader(self.eval_data, self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
                                        # num_workers=cfg.num_worker, pin_memory=True, drop_last=True, 
                                        # prefetch_factor=10, persistent_workers=True
                                        
            # build optim
            self.optimizer = build_optim(self.model.parameters(), config["optim"])
                
        elif self.dist_type == "ddp":
            # DDPHelper.setup(backend="nccl", mode="pythonrun")
            DDPHelper.setup(backend="nccl", mode="torchrun")
            self.model = DDP(self.model.cuda())
            # self.device = self.model.module.device
            self.train_data = DataLoader(self.train_data, self.batch_size, shuffle=False, 
                                         sampler=DistributedSampler(self.train_data), collate_fn=custom_collate_fn)
                                        # num_workers=cfg.num_worker, pin_memory=True, drop_last=True, 
                                        # prefetch_factor=10, persistent_workers=True
                                        
            if self.eval_data:
                self.eval_data = DataLoader(self.eval_data, self.batch_size, shuffle=False,
                                            sampler=DistributedSampler(self.eval_data), collate_fn=custom_collate_fn)
                                        # num_workers=cfg.num_worker, pin_memory=True, drop_last=True, 
                                        # prefetch_factor=10, persistent_workers=True
                                        
            # build optim
            self.optimizer = build_optim(self.model.module.parameters(), config["optim"])
            
        elif self.dist_type == "deepspeed":
            pass
        
        # optimizer learning rate schedule
        self.lr_scheduler = build_lr_scheduler(self.optimizer, config["lr_scheduler"])
        
        
    @property
    def device(self):
        return torch.cuda.current_device() # 返回默认GPU序号
    
    @property
    def micro_batch_size(self):
        return self.batch_size # 单个GPU在每轮训练中所载入的数据样本量
    
    def __call__(self, *args, **kwargs):
        return self.one_iter(*args, **kwargs)
    
    def gradient_accumulation_steps(self):
        return self.accu_steps
    
    def is_first_stage(self):
        return True
    
    def is_second_stage(self):
        return True

    def is_master(self):
        if self.dist_type is None:
            return True
        elif self.dist_type == "ddp":
            return DDPHelper.is_master()

    def _save_snapshot(self, *args, **kwargs):
        self.save_checkpoint(*args, **kwargs)
    
    def _load_snapshot(self, *args, **kwargs):
        self.load_checkpoint(*args, **kwargs)
    
    def save_checkpoint(self, save_dir, tag=None,
                        client_state={}, save_latest=True, 
                        exclude_frozen_parameters=False):
        ''' client_state: {
                hparams: hyper-parameter
                weights: model.state_dict()
                opt_state: opt.state_dict()
                scheduler_state: lr_schedule.state_dict()
                epoch: epoch
            }
        '''
        
        if self.dist_type == "deepspeed":
            pass # ds封装后的模型等需要额外的处理
        # 如果没有zero策略将模型分片，可以类似于ddp，用self.model.module来取出模型参数
        # 否则需要使用 engine.save_checkpoint('/path/to/checkpoint/dir') 
        #   和 engine.load_checkpoint('/path/to/checkpoint/dir') 将分片模型保存成完整模型再拿

        client_state["epoch"] = self.epoch
        client_state["hparams"] = self.config
        
        if self.dist_type is None:
            client_state["weights"] = self.model.state_dict()
            
        elif self.dist_type == "ddp":
            client_state["weights"] = self.model.module.state_dict()
        
        client_state["opt_state"] = self.optimizer.state_dict()
        
        if self.lr_scheduler is not None:
            client_state["scheduler_state"] = self.lr_scheduler.state_dict()
            
        client_state["global_train_iter"] = self.global_iter.get("train_iter", 0)
        client_state["global_eval_iter"] = self.global_iter.get("eval_iter", 0)
        
        if tag:
            save_dir = Path(save_dir).parent / tag / Path(save_dir).name
        
        save_dir = Path(save_dir).with_suffix('.pth')
        
        if not os.path.exists(save_dir.parent):
            os.makedirs(save_dir.parent)
        
        if save_latest:
            dst_path = Path(save_dir).with_stem("latest")
            # if os.path.exists(dst_path):
            #     os.remove(dst_path)
            try:
                os.remove(dst_path)
            except:
                pass
        
        if self.max_num_save:
            # all_pth_path = save_dir.with_stem("*")
            # all_pth_path = glob(str(all_pth_path))
            # all_pth_path.sort()
            # while len(all_pth_path) > self.max_num_save:
            #     os.remove(all_pth_path[0])
            #     all_pth_path.pop(0)
            file_list = [(f.name, f.stat().st_mtime) for f in os.scandir(str(save_dir.parent)) if f.is_file()]
            while len(file_list) > self.max_num_save:
                file_list.sort(key=lambda x:x[1])
                os.remove(str(save_dir.parent / file_list[0][0]))
                file_list = [(f.name, f.stat().st_mtime) for f in os.scandir(str(save_dir.parent)) if f.is_file()]
                
        torch.save(client_state, save_dir)
        
        if save_latest:
            os.symlink(save_dir, dst_path)
    
    def load_checkpoint(self, load_dir, tag=None, 
                        load_module_strict=True, load_optimizer_states=True, 
                        load_lr_scheduler_states=True, load_module_only=False, 
                        custom_load_fn=None, auto_latest=False):
        ''' client_state: {
                hparams: hyper-parameter
                weights: model.state_dict()
                opt_state: opt.state_dict()
                scheduler_state: lr_schedule.state_dict()
                epoch: epoch
            }
            PS: 在修改学习率相关参数时,load_module_only需要设置为True
        '''
        def load_latest_module(path):
            file_list = [(f.name, f.stat().st_mtime) for f in os.scandir(str(path.parent)) if f.is_file()]
            return path.parent / file_list[-1][0]
        
        if tag:
            load_dir = Path(load_dir).parent / tag / Path(load_dir).name
        
        load_dir = Path(load_dir).with_suffix(".pth")
        
        if not os.path.exists(load_dir):
            logger.info(f"Loading model params from {load_dir}. NOT EXIST!")
            return
        
        if auto_latest:
            load_dir = load_latest_module(load_dir)
        
        if custom_load_fn is None:
            custom_load_fn = torch.load
        
        client_state = custom_load_fn(str(load_dir), map_location='cuda')
        # device = torch.cuda.current_device()
        # client_state = custom_load_fn(str(load_dir), map_location=lambda storage, loc: storage.cuda(device))
        
        logger.info(f"Loading model params from {load_dir}.")
        
        if not client_state:
            return
        
        if not load_module_only:
            self.config = client_state["hparams"]
            self.reset_config(config=self.config)
            
            self.epoch_resum = client_state["epoch"] + 1 # resum, +1
            
            self.global_iter = {
                "train_iter": client_state.get("global_train_iter", 0), 
                "eval_iter": client_state.get("global_eval_iter", 0)
            }
            
            if load_optimizer_states:
                opt_state = client_state.get("opt_state", None)
                if opt_state is not None:
                    self.optimizer.load_state_dict(opt_state)
                    
            if load_lr_scheduler_states:
                scheduler_state = client_state.get("scheduler_state", None)
                if scheduler_state is not None:
                    self.lr_scheduler.load_state_dict(scheduler_state)
                
        self.model.load_state_dict(client_state["weights"], load_module_strict)

    def train_batch(self, *args, **kwargs):
        iter_id = kwargs.get("train_iter_id", None)
        data_iter = kwargs.get("data_iter", None)
        assert iter_id is not None, "not iteration index."
        assert data_iter is not None, "not data in training process."
        
        self.model.train()
        
        kwargs["cal_loss"] = True
        kwargs["cal_metric"] = False
        output = self.one_iter(*args, **kwargs)

        output["loss"].backward()
        
        # logger.info(f"train batch output0: {output}")

        if iter_id % self.accu_steps == 0: # 累计梯度
            self.optimizer.step()
            self.optimizer.zero_grad()

        # rvl = {}
        if self.dist_type is None:
            average_all(output)
            # rvl["train_output"] = output
        else:
            average_all_gather(output)
            average_all(output)
            # rvl["train_output"] = output

        # logger.info(f"train batch output1: {output}")

        output = delete_None(output)
        
        # logger.info(f"train batch output2: {output}")
        
        # with torch.no_grad():
        #     print(output)
        #     output = self.one_iter(*args, **kwargs)
        #     print(output)

        # return rvl
        return output

    @torch.no_grad()
    def eval_batch(self, *args, **kwargs):
        iter_id = kwargs.get("eval_iter_id", None)
        data_iter = kwargs.get("data_iter", None)
        assert iter_id is not None, "not iteration index."
        assert data_iter is not None, "not data in eval process."
        
        self.model.eval()
        
        kwargs["cal_loss"] = False
        kwargs["cal_metric"] = True
        output = self.one_iter(*args, **kwargs)
        
        # logger.info(f"eval batch output0: {output}")
        
        if "eval_tab" in output: # 记录可视化的结果表
            assert isinstance(output["eval_tab"], list), \
                "type of evaluation's table-output should be LIST. "
            self.vis.extend(output["eval_tab"])
                
        # rvl = {}
        if self.dist_type is None:
            average_all(output)
            # rvl["eval_output"] = output
        else:
            average_all_gather(output)
            average_all(output)
            # rvl["eval_output"] = output
                        
        # logger.info(f"eval batch output1: {output}")

        output = delete_None(output)
        
        # logger.info(f"eval batch output2: {output}")
    
        # return rvl
        return output

    def one_epoch(self, *args, **kwargs): # 模型训练一个周期
        mode = kwargs.get("mode", "train")
        self.epoch = kwargs.get("epoch_id")
        
        if mode == "train":
            dataload = self.train_data
        elif mode == "eval":
            dataload = self.eval_data
            
        if dataload is None:
            logger.info(f"error input param in {kwargs}: data is NULL.")
        
        if self.dist_type == "ddp":
            dataload.sampler.set_epoch(self.epoch) # batch_sampler
        
        rvl_dict = dict()
        
        def run_mean_dict(cur_output_dict: dict):
            for k, v in cur_output_dict.items():
                if k in rvl_dict:
                    rvl_dict[k] = [rvl_dict[k][0] + v, rvl_dict[k][1] + 1]
                else:
                    rvl_dict[k] = [v, 1]
                    
        def get_average_lr(optimizer):
            total_lr = 0
            for param_group in optimizer.param_groups:
                total_lr += param_group['lr']
            average_lr = total_lr / len(optimizer.param_groups)
            return average_lr
        
        for iter_id, batch_data in enumerate(dataload):
            
            # if iter_id > self.num_epoch_step:
            if isinstance(batch_data, dict):
                isok = False
                for k, v in batch_data.items():
                    if isinstance(v, list) or isinstance(v, tuple):
                        for vv in v:
                            if len(vv) < 1:
                                isok = True
                                break
                    else:
                        if len(v) < 1:
                            isok = True
                            break
                    if isok:
                        break
                if isok:
                    continue
                    
            elif len(batch_data) < 1:
                continue
            
            kwargs["data_iter"] = to_device(batch_data, self.device)
            
            if mode == "train":
                kwargs["train_iter_id"] = iter_id
                output_dict = self.train_batch(*args, **kwargs)
                
                if self.is_master():
                    output_dict["lr"] = get_average_lr(self.optimizer)
                    
                    if "train_iter" not in self.global_iter:
                        self.global_iter["train_iter"] = 0
                        
                    output_dict = {"ITER/train/"+k: v for k, v in output_dict.items()}
                    
                    if self.enable_iter_log:
                        self.cal_log(self.global_iter["train_iter"], log_info=output_dict)
                        
                    self.global_iter["train_iter"] += 1
                    run_mean_dict(output_dict)
                
            elif mode == "eval":
                kwargs["eval_iter_id"] = iter_id
                output_dict = self.eval_batch(*args, **kwargs)
                
                if self.is_master():
                    output_dict["lr"] = get_average_lr(self.optimizer)
                                        
                    if "eval_iter" not in self.global_iter:
                        self.global_iter["eval_iter"] = 0
                        
                    output_dict = {"ITER/eval/"+k: v for k, v in output_dict.items()}
                    
                    if self.enable_iter_log:
                        self.cal_log(self.global_iter["eval_iter"], log_info=output_dict)
                        
                    self.global_iter["eval_iter"] += 1
                    run_mean_dict(output_dict)
            
            if mode == "train" and self.is_master() and self.save_every_iter and iter_id % self.save_every_iter == 0:
                self.ckpt_io(self.config, name="iter_"+str(iter_id).zfill(8), mode="save")
        
        if mode == "train" and self.is_master() and self.save_every and self.epoch % self.save_every == 0:
            self.ckpt_io(self.config, name="epoch_"+str(self.epoch).zfill(8), mode="save")
            
        if self.is_master() and self.enable_epoch_log:
            rvl = {}
            for k, v in rvl_dict.items():
                rvl[k.replace("ITER", "EPOCH")] = v[0] / v[1]
            self.cal_log(self.epoch, log_info=rvl, enable_print=True)
        
        # self.epoch += 1 # 训练轮次累计
    
    def train(self, *args, **kwargs):
        kwargs["mode"] = "train"
        
        for epoch_id in range(self.epoch_resum, self.epoch_resum + self.num_epoch):
            kwargs["epoch_id"] = epoch_id
            self.one_epoch(*args, **kwargs)
            
            if self.lr_scheduler: # epoch轮次结束-更新学习率
                self.lr_scheduler.step()
            
            if self.is_master() and self.eval_every and (epoch_id - self.epoch_resum) % self.eval_every == 0:
                self.eval(*args, **kwargs)
                    
        if self.is_master():
            if self.logger_type == "wandb" or self.logger_type == "tensorboard":
                
                if self.logger is None:
                    if self.logger_type == "tensorboard":
                        self.logger = TbLogger()
                    elif self.logger_type == "wandb":
                        self.logger = WbLogger()
                if self.logger_type == "wandb":
                    self.logger.rec(name="training hyper-parameters", config=self.config)
                else: # tensorboard
                    self.logger.add_text("training hyper-parameters", dict_to_markdown(self.config, max_str_len=None))
            else:
                # from pprint import pprint
                print(f"training hyper-parameters: {self.config}")
                
        if self.dist_type == "ddp":
            DDPHelper.cleanup()
            
    def eval(self, *args, **kwargs):
        kwargs["mode"] = "eval"
        self.vis = []
        self.one_epoch(*args, **kwargs)
        prt_enable = kwargs.get("enable_print", False)
        if prt_enable and len(self.vis) > 0:
            self.wrt_visualization(self.vis)
        self.vis = [] # self.vis仅在eval中被消耗 

    def cal_log(self, step_id, *args, **kwargs):
        
        def cal_tb_log(logger: TbLogger, step_id, tag="", scalar_value=-1):
            logger.add_scalar(tag, scalar_value, step_id)
        
        def cal_wb_log(logger: WbLogger, info=dict()):
            logger.step(info)
        
        info = kwargs.get("log_info", {})
        if not info: return
        
        if self.logger is None:
            if self.logger_type == "tensorboard":
                self.logger = TbLogger()
            elif self.logger_type == "wandb":
                self.logger = WbLogger()
        
        if self.logger_type == "tensorboard":
            for k, v in info.items():
                cal_tb_log(self.logger, step_id, k, v)

        elif self.logger_type == "wandb":
            cal_wb_log(self.logger, info)
        
        if self.is_master():
            enable_print = kwargs.get("enable_print", False)
            if enable_print or (self.log_every and step_id % self.log_every == 0):
                logger.info(f": #{step_id}: info: {info}")

    def one_iter(self, *args, **kwargs): # 模型迭代一次获取结果
        data_iter = kwargs.get("data_iter", None)
        if data_iter is None: return
        return self.model(*args, **kwargs)

    def wrt_visualization(self, result):
        root_path = Path(self.snapshot_path).parent.parent / "vis"
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        jsonl_path = str(root_path / TIMESTAMP) + ".jsonl"
        write_jsonl(jsonl_path, result)
